# -*- coding: utf-8 -*-
"""
단타 스캘핑 안전강화 버전 (풀 스크립트)
- 유동성/스프레드 필터
- SL 거리 기반 포지션 사이징 (프롬프트 의도 충실)
- reduceOnly SL/TP + 실패 시 재시도
- 일일 손실 한도(킬스위치) + 중복진입 방지
- 재시작 복구 (거래소 포지션 ↔ DB 동기화)
- 정밀도/최소주문/격리/레버리지 설정
- 모델이 과도하게 NO_POSITION만 내는 경우 작은 사이즈의 Fallback 진입 가드
"""

import os, time, json, sqlite3, traceback
from datetime import datetime, timezone
from typing import Dict, Any

import pandas as pd
import ccxt
from dotenv import load_dotenv
from openai import OpenAI

# =========================
# 환경/경로
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(BASE_DIR, "txt"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "db"), exist_ok=True)

# .env 로드 (상위 config 우선, 로컬 .env 보조)
load_dotenv(os.path.join(BASE_DIR, "..", "buffett-config", ".env"))
load_dotenv()

def require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"[ENV] {name} 누락. .env에 {name}=... 을 추가하세요.")
    return v

MIN_HOLD_SEC    = int(os.getenv("MIN_HOLD_SEC", "60"))   # 최소 보유
NO_POS_STREAK_N = int(os.getenv("NO_POS_STREAK_N", "2")) # NO_POSITION N틱 연속 시 청산
OPENAI_API_KEY  = require_env("OPENAI_API_KEY")
BINANCE_API_KEY = require_env("BINANCE_API_KEY")
BINANCE_SECRET  = require_env("BINANCE_SECRET_KEY")

# 파라미터 (필요시 .env에서 조정)
AI_MODEL      = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
RISK_PCT      = float(os.getenv("RISK_PCT", "0.004"))     # 트레이드당 계좌 리스크 0.4%
MAX_LEVERAGE  = int(os.getenv("MAX_LEVERAGE", "6"))
DAILY_MAX_LOSS_USDT = float(os.getenv("DAILY_MAX_LOSS_USDT", "100"))
HEDGE_MODE    = os.getenv("HEDGE_MODE", "false").lower() == "true"
ENABLE_FALLBACK = os.getenv("ENABLE_FALLBACK", "false").lower() == "true"
MIN_RR = float(os.getenv("MIN_RR", "1.8"))
MAX_NOTIONAL_FRAC = float(os.getenv("MAX_NOTIONAL_FRAC", "0.60"))
LIQ_MAX_SPREAD = float(os.getenv("LIQ_MAX_SPREAD", "0.0010"))         # 0.10%
LIQ_MIN_DEPTH_USDT = float(os.getenv("LIQ_MIN_DEPTH_USDT", "20000"))  # ETH 기준 최소 호가 유동성(완화)
MAX_RISK_USDT = float(os.getenv("MAX_RISK_USDT", "2.5"))              # 1회 트레이드 최대 허용 손실(USDT)
MIN_RISK_USDT = float(os.getenv("MIN_RISK_USDT", "0.06"))             # 1회 트레이드 최소 위험금액(체결 하한)
MAX_RISK_PCT_OF_FREE = float(os.getenv("MAX_RISK_PCT_OF_FREE", "0.02")) # 잔고 대비 1회 최대 리스크 비율(2.0%)
RANGE_BLOCK_ENTRY = os.getenv("RANGE_BLOCK_ENTRY", "true").lower() == "true"
FLIP_CONFIRM_TICKS = int(os.getenv("FLIP_CONFIRM_TICKS", "2"))
ENABLE_TIME_FILTER = os.getenv("ENABLE_TIME_FILTER", "false").lower() == "true"
ALLOWED_UTC_HOURS = os.getenv("ALLOWED_UTC_HOURS", "0,1,2,6,7,8,12,13,14,15,16,17,18,19,20,21,22,23")
MAX_CONSEC_LOSSES = int(os.getenv("MAX_CONSEC_LOSSES", "3"))
COOLDOWN_SEC_AFTER_LOSS_STREAK = int(os.getenv("COOLDOWN_SEC_AFTER_LOSS_STREAK", "1800"))
ENABLE_LOSS_STREAK_COOLDOWN = os.getenv("ENABLE_LOSS_STREAK_COOLDOWN", "false").lower() == "true"
ALGO_ORDER_UNSUPPORTED = False
ENABLE_LOSS_HOLD_ON_NO_POS = os.getenv("ENABLE_LOSS_HOLD_ON_NO_POS", "true").lower() == "true"
LOSS_HOLD_TRIGGER_PCT = float(os.getenv("LOSS_HOLD_TRIGGER_PCT", "0.8"))  # 미실현손실 -0.8% 이하일 때 유예 대상
LOSS_HOLD_MAX_SEC = int(os.getenv("LOSS_HOLD_MAX_SEC", "900"))            # 최대 유예 15분

TICKER_SEC    = int(os.getenv("TICKER_SEC", "240"))
POSITION_TICKER_SEC = int(os.getenv("POSITION_TICKER_SEC", "150"))
HEAVY_SEC     = 15
ANALYZE_SEC   = int(os.getenv("ANALYZE_SEC", "30"))
OPENAI_INPUT_COST_PER_1M = float(os.getenv("OPENAI_INPUT_COST_PER_1M", "0"))
OPENAI_OUTPUT_COST_PER_1M = float(os.getenv("OPENAI_OUTPUT_COST_PER_1M", "0"))
OPENAI_MONTHLY_BUDGET_USD = float(os.getenv("OPENAI_MONTHLY_BUDGET_USD", "0"))

# 코인명
COIN_NAME_PATH = os.path.join(BASE_DIR, "txt", "coinName.txt")
BASE = (os.getenv("COIN_NAME") or "").strip().replace(" ", "").upper()
if not BASE:
    if os.path.exists(COIN_NAME_PATH):
        BASE = open(COIN_NAME_PATH, "r", encoding="utf-8").read().strip().replace(" ", "").upper()
    else:
        BASE = "XCN"

SYMBOL  = f"{BASE}/USDT:USDT"   # USDT-M Perpetual
DB_FILE = os.path.join(BASE_DIR, "db", f"{BASE}_trading.db")
LEGACY_DB_FILE = os.path.join(BASE_DIR, "db", f"{BASE.lower()}_trading.db")
if not os.path.exists(DB_FILE) and os.path.exists(LEGACY_DB_FILE):
    try:
        os.replace(LEGACY_DB_FILE, DB_FILE)
    except Exception:
        DB_FILE = LEGACY_DB_FILE

ROTATE_ON_IDLE = os.getenv("ROTATE_ON_IDLE", "true").lower() == "true"
ROTATE_IDLE_STREAK_N = int(os.getenv("ROTATE_IDLE_STREAK_N", "2"))
ROTATE_COINS_RAW = os.getenv("ROTATE_COINS", "XRP,ETH,SOL")
ENABLE_GLOBAL_IDLE_CYCLE_COOLDOWN = os.getenv("ENABLE_GLOBAL_IDLE_CYCLE_COOLDOWN", "true").lower() == "true"
GLOBAL_IDLE_CYCLE_COOLDOWN_SEC = int(os.getenv("GLOBAL_IDLE_CYCLE_COOLDOWN_SEC", "900"))
ENABLE_BREAKOUT_WAKEUP = os.getenv("ENABLE_BREAKOUT_WAKEUP", "true").lower() == "true"
BREAKOUT_LOOKBACK = int(os.getenv("BREAKOUT_LOOKBACK", "20"))
BREAKOUT_VOL_MULT = float(os.getenv("BREAKOUT_VOL_MULT", "1.8"))


def parse_rotation_coins() -> list[str]:
    coins = []
    seen = set()
    for token in (ROTATE_COINS_RAW or "").split(","):
        coin = token.strip().replace(" ", "").upper()
        if not coin or coin in seen:
            continue
        seen.add(coin)
        coins.append(coin)
    if BASE not in seen:
        coins.insert(0, BASE)
    return coins


ROTATION_COINS = parse_rotation_coins()
rotation_idx = ROTATION_COINS.index(BASE) if BASE in ROTATION_COINS else 0
AI_USAGE_TOTAL_IN = 0
AI_USAGE_TOTAL_OUT = 0
AI_USAGE_TOTAL_ALL = 0
AI_USAGE_DAY = datetime.now().strftime("%Y-%m-%d")
GLOBAL_IDLE_COOLDOWN_UNTIL = 0.0

# =========================
# 프롬프트 로드
# =========================
SP_PATH = os.path.join(BASE_DIR, "txt", "system_prompt.txt")
if os.path.exists(SP_PATH):
    SYSTEM_PROMPT_RAW = open(SP_PATH, "r", encoding="utf-8").read()
else:
    SYSTEM_PROMPT_RAW = """You are the world’s top high-frequency crypto scalper, specializing in XCN/USDT perpetual futures trading on Binance using the ChatGPT API.
Use 1m/3m/5m momentum. Output JSON with fields: direction, recommended_position_size, recommended_leverage, stop_loss_percentage, take_profit_percentage, reasoning."""

def render_system_prompt(base: str) -> str:
    """현재 거래 코인 기준으로 시스템 프롬프트를 재생성한다."""
    b = (base or "").strip().upper()
    symbol_spot = f"{b}/USDT"
    p = SYSTEM_PROMPT_RAW
    p = p.replace("{BASE}", b).replace("{SYMBOL}", symbol_spot)
    # 기존 프롬프트에 하드코딩된 대표 심볼 치환
    p = p.replace("DOGE/USDT", symbol_spot).replace("XCN/USDT", symbol_spot).replace("ETH/USDT", symbol_spot)
    # 다른 코인명 하드코딩이 남아있어도 현재 심볼을 우선하도록 런타임 가드 추가
    runtime_guard = (
        f"Runtime trading target is strictly {symbol_spot} perpetual futures on Binance. "
        "All analysis, reasoning, and output must be for this symbol only."
    )
    return f"{runtime_guard}\n\n{p}"

SYSTEM_PROMPT = render_system_prompt(BASE)


def refresh_runtime_for_coin(new_base: str):
    global BASE, SYMBOL, DB_FILE, LEGACY_DB_FILE, SYSTEM_PROMPT, SYSTEM_PROMPT_RAW
    normalized = (new_base or "").strip().replace(" ", "").upper()
    if not normalized or normalized == BASE:
        return

    BASE = normalized
    SYMBOL = f"{BASE}/USDT:USDT"
    DB_FILE = os.path.join(BASE_DIR, "db", f"{BASE}_trading.db")
    LEGACY_DB_FILE = os.path.join(BASE_DIR, "db", f"{BASE.lower()}_trading.db")
    if not os.path.exists(DB_FILE) and os.path.exists(LEGACY_DB_FILE):
        try:
            os.replace(LEGACY_DB_FILE, DB_FILE)
        except Exception:
            DB_FILE = LEGACY_DB_FILE

    try:
        with open(COIN_NAME_PATH, "w", encoding="utf-8") as f:
            f.write(BASE)
    except Exception as e:
        log(f"[WARN] coinName.txt 업데이트 실패: {e}")

    sp = read_text(SP_PATH)
    if sp.strip():
        SYSTEM_PROMPT_RAW = sp
    SYSTEM_PROMPT = render_system_prompt(BASE)


def rotate_to_next_coin() -> bool:
    global rotation_idx
    if len(ROTATION_COINS) < 2:
        return False
    rotation_idx = (rotation_idx + 1) % len(ROTATION_COINS)
    prev_base = BASE
    next_base = ROTATION_COINS[rotation_idx]
    refresh_runtime_for_coin(next_base)
    setup_db()
    ensure_market()
    log(f"[ROTATE] idle NO_POSITION {ROTATE_IDLE_STREAK_N}회 이상 → {prev_base} -> {next_base} 전환")
    return True

# =========================
# 클라이언트/거래소
# =========================
client = OpenAI(api_key=OPENAI_API_KEY)

exchange = ccxt.binance({
    "apiKey": BINANCE_API_KEY,
    "secret": BINANCE_SECRET,
    "enableRateLimit": True,
    "options": {
        "defaultType": "future",
        "defaultMarket": "futures",
        "adjustForTimeDifference": True,
    }
})

# =========================
# 유틸
# =========================
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def now_iso():
    return datetime.now().isoformat(timespec="seconds")

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def log_ai_usage(usage):
    """OpenAI usage를 output.log에 누적 기록한다.
    NOTE: API에서 정확 잔액은 제공되지 않아 토큰/추정비용만 기록 가능.
    """
    global AI_USAGE_TOTAL_IN, AI_USAGE_TOTAL_OUT, AI_USAGE_TOTAL_ALL, AI_USAGE_DAY

    today = datetime.now().strftime("%Y-%m-%d")
    if today != AI_USAGE_DAY:
        AI_USAGE_DAY = today
        AI_USAGE_TOTAL_IN = 0
        AI_USAGE_TOTAL_OUT = 0
        AI_USAGE_TOTAL_ALL = 0

    in_tok = int(getattr(usage, "prompt_tokens", 0) or 0)
    out_tok = int(getattr(usage, "completion_tokens", 0) or 0)
    all_tok = int(getattr(usage, "total_tokens", (in_tok + out_tok)) or (in_tok + out_tok))

    AI_USAGE_TOTAL_IN += in_tok
    AI_USAGE_TOTAL_OUT += out_tok
    AI_USAGE_TOTAL_ALL += all_tok

    msg = (
        f"[API] OpenAI usage this_call(in={in_tok}, out={out_tok}, total={all_tok}) "
        f"today(in={AI_USAGE_TOTAL_IN}, out={AI_USAGE_TOTAL_OUT}, total={AI_USAGE_TOTAL_ALL})"
    )

    if OPENAI_INPUT_COST_PER_1M > 0 or OPENAI_OUTPUT_COST_PER_1M > 0:
        call_cost = (in_tok / 1_000_000.0) * OPENAI_INPUT_COST_PER_1M + (out_tok / 1_000_000.0) * OPENAI_OUTPUT_COST_PER_1M
        day_cost = (AI_USAGE_TOTAL_IN / 1_000_000.0) * OPENAI_INPUT_COST_PER_1M + (AI_USAGE_TOTAL_OUT / 1_000_000.0) * OPENAI_OUTPUT_COST_PER_1M
        msg += f", est_usd(call={call_cost:.6f}, today={day_cost:.4f})"
        if OPENAI_MONTHLY_BUDGET_USD > 0:
            remaining = max(0.0, OPENAI_MONTHLY_BUDGET_USD - day_cost)
            msg += f", est_budget_left={remaining:.4f}/{OPENAI_MONTHLY_BUDGET_USD:.2f}"
    else:
        msg += ", est_usd=off(set OPENAI_INPUT_COST_PER_1M/OPENAI_OUTPUT_COST_PER_1M)"

    log(msg)

def loop_wait_seconds() -> int:
    return POSITION_TICKER_SEC if fetch_current_position() else TICKER_SEC

def idle_no_position_wait_seconds(idle_streak: int) -> int:
    # 무포지션이 길어질수록 호출 빈도를 줄여 API 비용을 절감
    base = max(10, TICKER_SEC)
    backoff = min(240, max(0, idle_streak) * 30)
    return base + backoff

def _parse_allowed_hours(raw: str):
    hours = set()
    for token in (raw or "").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            h = int(token)
        except Exception:
            continue
        if 0 <= h <= 23:
            hours.add(h)
    return hours

def is_allowed_trading_time() -> bool:
    if not ENABLE_TIME_FILTER:
        return True
    allowed = _parse_allowed_hours(ALLOWED_UTC_HOURS)
    if not allowed:
        return True
    return datetime.now(timezone.utc).hour in allowed

# =========================
# DB
# =========================
def db_conn():
    conn = sqlite3.connect(DB_FILE, timeout=10, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    # DB 파일 전환(legacy -> new) 직후에도 핵심 테이블이 항상 존재하도록 보장
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            action TEXT NOT NULL,
            entry_price REAL NOT NULL,
            amount REAL NOT NULL,
            leverage INTEGER NOT NULL,
            sl_price REAL NOT NULL,
            tp_price REAL NOT NULL,
            sl_percentage REAL NOT NULL,
            tp_percentage REAL NOT NULL,
            position_size_percentage REAL NOT NULL,
            investment_amount REAL NOT NULL,
            status TEXT DEFAULT 'OPEN',
            exit_price REAL,
            exit_timestamp TEXT,
            profit_loss REAL,
            profit_loss_percentage REAL
        );
        CREATE TABLE IF NOT EXISTS ai_analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            current_price REAL NOT NULL,
            direction TEXT NOT NULL,
            recommended_position_size REAL NOT NULL,
            recommended_leverage INTEGER NOT NULL,
            stop_loss_percentage REAL NOT NULL,
            take_profit_percentage REAL NOT NULL,
            reasoning TEXT NOT NULL,
            trade_id INTEGER,
            FOREIGN KEY (trade_id) REFERENCES trades (id)
        );
    """)
    return conn

def read_text(path: str) -> str:
    return open(path, "r", encoding="utf-8").read() if os.path.exists(path) else ""

def setup_db():
    conn = db_conn(); cur = conn.cursor()
    # trades
    trades_sql = read_text(os.path.join(BASE_DIR, "txt", "create_trades_table.txt"))
    if not trades_sql.strip():
        trades_sql = """
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            action TEXT NOT NULL,                -- 'long' | 'short'
            entry_price REAL NOT NULL,
            amount REAL NOT NULL,
            leverage INTEGER NOT NULL,
            sl_price REAL NOT NULL,
            tp_price REAL NOT NULL,
            sl_percentage REAL NOT NULL,
            tp_percentage REAL NOT NULL,
            position_size_percentage REAL NOT NULL,
            investment_amount REAL NOT NULL,
            status TEXT DEFAULT 'OPEN',          -- 'OPEN' | 'CLOSED'
            exit_price REAL,
            exit_timestamp TEXT,
            profit_loss REAL,
            profit_loss_percentage REAL
        );
        """
    # ai_analysis
    ai_sql = read_text(os.path.join(BASE_DIR, "txt", "create_table_ai_analysis.txt"))
    if not ai_sql.strip():
        ai_sql = """
        CREATE TABLE IF NOT EXISTS ai_analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            current_price REAL NOT NULL,
            direction TEXT NOT NULL,
            recommended_position_size REAL NOT NULL,
            recommended_leverage INTEGER NOT NULL,
            stop_loss_percentage REAL NOT NULL,
            take_profit_percentage REAL NOT NULL,
            reasoning TEXT NOT NULL,
            trade_id INTEGER,
            FOREIGN KEY (trade_id) REFERENCES trades (id)
        );
        """
    cur.executescript(trades_sql)
    cur.executescript(ai_sql)
    cur.executescript("""
        CREATE INDEX IF NOT EXISTS idx_trades_status_time ON trades(status, timestamp);
        CREATE INDEX IF NOT EXISTS idx_ai_trade_id ON ai_analysis(trade_id);
    """)
    conn.commit(); conn.close()
    log("DB 준비 완료")

def save_ai_analysis(data: Dict[str, Any], trade_id=None) -> int:
    conn = db_conn(); cur = conn.cursor()
    insert_sql = read_text(os.path.join(BASE_DIR, "txt", "insert_ai_analysis.txt"))
    if not insert_sql.strip():
        insert_sql = """
        INSERT INTO ai_analysis (
            timestamp, current_price, direction, recommended_position_size,
            recommended_leverage, stop_loss_percentage, take_profit_percentage,
            reasoning, trade_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
    cur.execute(insert_sql, (
        now_iso(),
        data.get("current_price", 0.0),
        data.get("direction", "NO_POSITION"),
        data.get("recommended_position_size", 0.0),
        data.get("recommended_leverage", 0),
        data.get("stop_loss_percentage", 0.0),
        data.get("take_profit_percentage", 0.0),
        data.get("reasoning", ""),
        trade_id
    ))
    rid = cur.lastrowid
    conn.commit(); conn.close()
    return rid

def save_trade(data: Dict[str, Any]) -> int:
    conn = db_conn(); cur = conn.cursor()
    cur.execute("""
        INSERT INTO trades (
            timestamp, action, entry_price, amount, leverage,
            sl_price, tp_price, sl_percentage, tp_percentage,
            position_size_percentage, investment_amount
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        now_iso(),
        data.get("action",""),
        data.get("entry_price",0.0),
        data.get("amount",0.0),
        data.get("leverage",0),
        data.get("sl_price",0.0),
        data.get("tp_price",0.0),
        data.get("sl_percentage",0.0),
        data.get("tp_percentage",0.0),
        data.get("position_size_percentage",0.0),
        data.get("investment_amount",0.0)
    ))
    rid = cur.lastrowid
    conn.commit(); conn.close()
    return rid

def update_trade_status(trade_id: int, status: str, exit_price=None, exit_timestamp=None,
                        profit_loss=None, profit_loss_percentage=None):
    conn = db_conn(); cur = conn.cursor()
    fields, vals = ["status = ?"], [status]
    if exit_price is not None: fields.append("exit_price = ?"); vals.append(exit_price)
    if exit_timestamp is not None: fields.append("exit_timestamp = ?"); vals.append(exit_timestamp)
    if profit_loss is not None: fields.append("profit_loss = ?"); vals.append(profit_loss)
    if profit_loss_percentage is not None: fields.append("profit_loss_percentage = ?"); vals.append(profit_loss_percentage)
    sql = f"UPDATE trades SET {', '.join(fields)} WHERE id = ?"
    vals.append(trade_id)
    cur.execute(sql, vals)
    conn.commit(); conn.close()

def update_open_trade_levels(sl_price: float, tp_price: float, sl_pct: float = None, tp_pct: float = None):
    conn = db_conn(); cur = conn.cursor()
    fields, vals = ["sl_price = ?", "tp_price = ?"], [sl_price, tp_price]
    if sl_pct is not None:
        fields.append("sl_percentage = ?"); vals.append(sl_pct)
    if tp_pct is not None:
        fields.append("tp_percentage = ?"); vals.append(tp_pct)
    vals.append("OPEN")
    cur.execute(f"""
        UPDATE trades
        SET {", ".join(fields)}
        WHERE id = (
            SELECT id FROM trades
            WHERE status = ?
            ORDER BY timestamp DESC
            LIMIT 1
        )
    """, vals)
    conn.commit(); conn.close()

def get_latest_open_trade():
    conn = db_conn(); cur = conn.cursor()
    cur.execute("""
        SELECT id, action, entry_price, amount, leverage, sl_price, tp_price
        FROM trades WHERE status='OPEN' ORDER BY timestamp DESC LIMIT 1
    """)
    r = cur.fetchone(); conn.close()
    if r:
        return {"id":r[0], "action":r[1], "entry_price":r[2], "amount":r[3],
                "leverage":r[4], "sl_price":r[5], "tp_price":r[6]}
    return None

def today_realized_pnl() -> float:
    conn = db_conn(); cur = conn.cursor()
    cur.execute("""
        SELECT COALESCE(SUM(profit_loss),0)
        FROM trades
        WHERE status='CLOSED' AND date(exit_timestamp)=date('now','localtime')
    """)
    v = float(cur.fetchone()[0] or 0.0)
    conn.close()
    return v

def get_recent_consecutive_losses(limit: int = 10) -> int:
    conn = db_conn(); cur = conn.cursor()
    cur.execute("""
        SELECT profit_loss
        FROM trades
        WHERE status='CLOSED' AND profit_loss IS NOT NULL
        ORDER BY exit_timestamp DESC, id DESC
        LIMIT ?
    """, (limit,))
    rows = cur.fetchall()
    conn.close()

    streak = 0
    for (pl,) in rows:
        if float(pl or 0.0) < 0:
            streak += 1
        else:
            break
    return streak

# =========================
# 마켓/정밀도/유동성
# =========================
def ensure_market():
    exchange.load_markets();  return exchange.market(SYMBOL)

def set_margin_and_leverage(leverage: int):
    try:
        exchange.set_margin_mode('isolated', SYMBOL)
    except Exception:
        pass
    lev = int(clamp(leverage, 1, MAX_LEVERAGE))
    exchange.set_leverage(lev, SYMBOL)
    return lev

def liquidity_ok(symbol: str, max_spread=0.002, min_depth_usdt=2000):
    """
    max_spread: 허용 최대 스프레드(0.002=0.2%)
    min_depth_usdt: 상/하단 10호가 합산 최소 유동성 USDT
    """
    ob = exchange.fetch_order_book(symbol, limit=10)
    if not ob.get('bids') or not ob.get('asks'):
        return False, 0.0, 0.0, 0.0
    best_bid = float(ob['bids'][0][0]); best_ask = float(ob['asks'][0][0])
    spread = (best_ask - best_bid) / ((best_ask + best_bid) / 2.0)
    depth_bid = sum([p * a for p,a in ob['bids']])
    depth_ask = sum([p * a for p,a in ob['asks']])
    depth_usdt = min(depth_bid, depth_ask)
    return (spread <= max_spread and depth_usdt >= min_depth_usdt), spread, depth_usdt, best_bid

def cancel_all_orders_for_symbol():
    """해당 심볼 모든 미체결 주문 취소 (실패해도 다음 단계 진행)"""
    try:
        open_orders = exchange.fetch_open_orders(SYMBOL)
        for o in open_orders:
            try:
                exchange.cancel_order(o['id'], SYMBOL)
            except Exception as e:
                log(f"주문 취소 실패(무시): {e}")
    except Exception:
        pass

def flatten_position_if_any():
    """보유 포지션이 있으면 시장가로 전량 청산"""
    try:
        pos = fetch_current_position()
        if not pos:
            return False
        amt = float(pos["amount"])
        if amt <= 0:
            return False
        side = 'sell' if pos["side"]=="long" else 'buy'
        exchange.create_order(SYMBOL, 'market', side, amt, None, {"reduceOnly": True})
        log("보유 포지션 평탄화 완료")
        return True
    except Exception as e:
        log(f"평탄화 실패(계속 진행): {e}")
        return False

# =========================
# 캔들 스냅샷 → AI 결정
# =========================
def fetch_candles(tf: str, limit=120):
    o = exchange.fetch_ohlcv(SYMBOL, timeframe=tf, limit=limit)
    df = pd.DataFrame(o, columns=["ts","open","high","low","close","volume"]).astype(
        {"open":float,"high":float,"low":float,"close":float,"volume":float}
    )
    return df


def breakout_wakeup_signal(symbol: str) -> bool:
    """쿨다운 중 재개 트리거용: 직전 구간 고/저 돌파 + 거래량 급증."""
    if not ENABLE_BREAKOUT_WAKEUP:
        return False
    try:
        limit = max(10, BREAKOUT_LOOKBACK + 2)
        o = exchange.fetch_ohlcv(symbol, timeframe="1m", limit=limit)
        if not o or len(o) < (BREAKOUT_LOOKBACK + 2):
            return False
        df = pd.DataFrame(o, columns=["ts", "open", "high", "low", "close", "volume"]).astype(
            {"open": float, "high": float, "low": float, "close": float, "volume": float}
        )
        prev = df.iloc[-(BREAKOUT_LOOKBACK + 1):-1]
        last = df.iloc[-1]
        prev_high = float(prev["high"].max())
        prev_low = float(prev["low"].min())
        avg_vol = float(prev["volume"].mean())
        last_close = float(last["close"])
        last_vol = float(last["volume"])
        vol_ok = avg_vol > 0 and last_vol >= (avg_vol * BREAKOUT_VOL_MULT)
        break_up = last_close > prev_high
        break_dn = last_close < prev_low
        return bool(vol_ok and (break_up or break_dn))
    except Exception:
        return False

def compute_indicator_pack(df: pd.DataFrame) -> Dict[str, Any]:
    if df is None or df.empty:
        return {}

    d = df.copy()
    d["close"] = d["close"].astype(float)
    d["volume"] = d["volume"].astype(float)
    d["pv"] = d["close"] * d["volume"]

    ema9 = float(d["close"].ewm(span=9, adjust=False).mean().iloc[-1])
    v_sum = float(d["volume"].sum())
    vwap = float(d["pv"].sum() / v_sum) if v_sum > 0 else float(d["close"].iloc[-1])

    # 간단 볼륨 프로파일: 최근 구간을 12개 가격 구간으로 나눠 고거래량 노드(HVN) 추출
    sub = d.tail(100).copy()
    pmin = float(sub["close"].min())
    pmax = float(sub["close"].max())
    hvn_levels = []
    if pmax > pmin:
        bins = pd.cut(sub["close"], bins=12, include_lowest=True)
        vp = sub.groupby(bins, observed=False)["volume"].sum().sort_values(ascending=False).head(3)
        for interval, _vol in vp.items():
            if pd.isna(interval):
                continue
            hvn_levels.append(float((interval.left + interval.right) / 2.0))

    last = float(d["close"].iloc[-1])
    regime = "RANGE"
    if last > vwap and last > ema9:
        regime = "TRENDING_UP"
    elif last < vwap and last < ema9:
        regime = "TRENDING_DOWN"

    return {
        "last_close": last,
        "ema9": ema9,
        "vwap": vwap,
        "above_vwap": bool(last > vwap),
        "above_ema9": bool(last > ema9),
        "hvn_levels": hvn_levels,
        "regime_hint": regime,
    }

def build_snapshot():
    data = {}
    indicators = {}
    for tf in ["1m","3m","5m"]:
        try:
            df = fetch_candles(tf)
            data[tf] = df.to_dict(orient="records")
            indicators[tf] = compute_indicator_pack(df)
        except Exception as e:
            log(f"캔들 수집 실패 {tf}: {e}")
            data[tf] = []
            indicators[tf] = {}
    price = float(exchange.fetch_ticker(SYMBOL)["last"])
    return {
        "timestamp": now_iso(),
        "symbol": SYMBOL,
        "current_price": price,
        "timeframes": data,
        "indicator_pack": indicators
    }

def is_range_regime(snapshot: Dict[str, Any]) -> bool:
    pack = snapshot.get("indicator_pack", {}) or {}
    r1 = str((pack.get("1m", {}) or {}).get("regime_hint", ""))
    r3 = str((pack.get("3m", {}) or {}).get("regime_hint", ""))
    return r1 == "RANGE" and r3 == "RANGE"

def ai_decide(snapshot: Dict[str,Any]) -> Dict[str,Any]:
    try:
        resp = client.chat.completions.create(
            model=AI_MODEL,
            response_format={"type":"json_object"},
            messages=[
                {"role":"system","content":SYSTEM_PROMPT},
                {"role":"user","content":json.dumps(snapshot, ensure_ascii=False)}
            ],
        )
        log_ai_usage(resp.usage)
        raw = resp.choices[0].message.content or "{}"
        d = json.loads(raw)
    except Exception as e:
        log(f"[AI] 응답 파싱 실패 → NO_POSITION로 대체: {e}")
        d = {}

    # 안전 파싱 + 기본값
    direction = str((d.get("direction") or "NO_POSITION")).upper()
    if direction not in ("LONG","SHORT","NO_POSITION"):
        direction = "NO_POSITION"

    pos_scale = clamp(_to_float(d.get("recommended_position_size"), 1.0), 0.5, 2.0)
    lev       = int(clamp(_to_float(d.get("recommended_leverage"), 5), 1, MAX_LEVERAGE))

    sl_pct = _to_float(d.get("stop_loss_percentage"), 0.003)
    tp_pct = _to_float(d.get("take_profit_percentage"), 0.006)

    # 1보다 큰 값은 %로 간주해서 보정
    if sl_pct > 1: sl_pct = sl_pct / 100.0
    if tp_pct > 1: tp_pct = tp_pct / 100.0

    sl_pct = clamp(sl_pct, 0.001, 0.05)
    tp_pct = clamp(tp_pct, 0.002, 0.2)
    # 모델이 낮은 RR(예: 1.6)을 반환해도 실행 필터(MIN_RR)와 충돌하지 않도록 TP 하한 보정
    min_tp_pct = sl_pct * max(MIN_RR, 1.0)
    if tp_pct + 1e-12 < min_tp_pct:
        tp_pct = min(0.2, min_tp_pct)

    return {
        "direction": direction,
        "recommended_position_size": pos_scale,
        "recommended_leverage": lev,
        "stop_loss_percentage": sl_pct,
        "take_profit_percentage": tp_pct,
        "reasoning": d.get("reasoning") or ""
    }

def has_enough_data(snap) -> bool:
    tfs = snap.get("timeframes", {})
    return all(len(tfs.get(tf, [])) >= 60 for tf in ["1m","3m","5m"])

def fallback_decision_from_snapshot(snap: Dict[str,Any]) -> Dict[str,Any]:
    """
    데이터·유동성 정상인데 모델이 NO_POSITION만 내면,
    아주 작은 크기의 추세 추종 진입으로 안전하게 대체.
      - 1m 단순 모멘텀: 최근20 평균 위면 LONG, 아니면 SHORT
      - size 0.6, lev 5, SL 0.35%, TP 0.8% (R:R > 2)
    """
    tfs = snap.get("timeframes", {})
    df1 = pd.DataFrame(tfs.get("1m", []))
    if df1.empty or "close" not in df1.columns:
        return {
            "direction":"NO_POSITION","recommended_position_size":1.0,"recommended_leverage":5,
            "stop_loss_percentage":0.0035,"take_profit_percentage":0.008,
            "reasoning":"[DATA_GAP] Fallback failed: no 1m data."
        }

    closes = df1["close"].astype(float).tail(60).reset_index(drop=True)
    if len(closes) < 20:
        return {
            "direction":"NO_POSITION","recommended_position_size":1.0,"recommended_leverage":5,
            "stop_loss_percentage":0.0035,"take_profit_percentage":0.008,
            "reasoning":"[DATA_GAP] Fallback failed: <20 closes."
        }

    sma20 = closes.tail(20).mean()
    last  = float(closes.iloc[-1])
    direction = "LONG" if last > sma20 else "SHORT"

    return {
        "direction": direction,
        "recommended_position_size": 0.6,
        "recommended_leverage": 5,
        "stop_loss_percentage": 0.0035,
        "take_profit_percentage": 0.008,
        "reasoning": f"[FALLBACK] simple-1m-momentum last={last:.6f} vs sma20={sma20:.6f}; enter small size with RR>2."
    }

def _to_float(x, default):
    """모델 응답의 숫자/문자/None/공백/'0.3%' 등을 안전하게 float으로 변환"""
    try:
        if x is None:
            return default
        if isinstance(x, str):
            s = x.strip()
            if not s:
                return default
            s = s.replace('%', '')  # '0.3%' -> '0.3'
            return float(s)
        return float(x)
    except Exception:
        return default
# =========================
# 포지션/주문
# =========================
def fetch_current_position():
    positions = exchange.fetch_positions([SYMBOL])
    for p in positions:
        if p.get("symbol")==SYMBOL:
            amt = float(p["info"].get("positionAmt", 0))
            entry = float(p["info"].get("entryPrice", 0))
            if amt>0:  return {"side":"long","amount":amt,"entry":entry}
            if amt<0:  return {"side":"short","amount":abs(amt),"entry":entry}
    return None

def unrealized_pnl_pct(pos: Dict[str, Any], current_price: float):
    entry = float(pos.get("entry") or 0.0)
    if entry <= 0:
        return None
    if pos.get("side") == "long":
        return (current_price / entry - 1.0) * 100.0
    return (1.0 - current_price / entry) * 100.0

def sync_position_db():
    on_ex = fetch_current_position()
    in_db = get_latest_open_trade()
    if on_ex and not in_db:
        entry = on_ex["entry"] or float(exchange.fetch_ticker(SYMBOL)["last"])
        save_trade({
            "action": on_ex["side"],
            "entry_price": entry,
            "amount": on_ex["amount"],
            "leverage": 5,
            "sl_price": 0.0, "tp_price": 0.0,
            "sl_percentage": 0.0, "tp_percentage": 0.0,
            "position_size_percentage": 0.0, "investment_amount": 0.0
        })
        log("재시작 복구: 거래소 포지션을 DB에 동기화")
    elif (not on_ex) and in_db:
        price = float(exchange.fetch_ticker(SYMBOL)["last"])
        entry = float(in_db["entry_price"]); amt = float(in_db["amount"])
        if in_db["action"]=="long":
            pl = (price-entry)*amt; pl_pct=(price/entry-1)*100 if entry else 0.0
        else:
            pl = (entry-price)*amt; pl_pct=(1-price/entry)*100 if entry else 0.0
        update_trade_status(in_db["id"], "CLOSED", price, now_iso(), pl, pl_pct)
        log("무포지션 동기화: DB 포지션 종결")

def place_protective_orders(direction: str, amount: float, sl_price: float, tp_price: float):
    global ALGO_ORDER_UNSUPPORTED
    if ALGO_ORDER_UNSUPPORTED:
        return False

    # reduceOnly 보호주문 + 재시도
    side_sl = 'sell' if direction=="LONG" else 'buy'
    side_tp = side_sl
    params = lambda pos: {
        "stopPrice": pos,
        "reduceOnly": True,
        "workingType": "MARK_PRICE",
        **({"positionSide":"LONG"} if HEDGE_MODE and direction=="LONG" else {}),
        **({"positionSide":"SHORT"} if HEDGE_MODE and direction=="SHORT" else {}),
    }
    # 재시도 3회
    for i in range(3):
        try:
            exchange.create_order(SYMBOL, 'STOP_MARKET', side_sl, amount, None, params(sl_price))
            break
        except Exception as e:
            log(f"SL 주문 실패 재시도({i+1}/3): {e}"); time.sleep(1)
            if i==2:
                if "code\":-4120" in str(e) or "Algo Order API" in str(e):
                    ALGO_ORDER_UNSUPPORTED = True
                    log("[INFO] 이 계정은 기본 엔드포인트의 조건부 주문 미지원으로 감지됨. 이후 보호주문 재시도 생략")
                return False
    for i in range(3):
        try:
            exchange.create_order(SYMBOL, 'TAKE_PROFIT_MARKET', side_tp, amount, None, params(tp_price))
            break
        except Exception as e:
            log(f"TP 주문 실패 재시도({i+1}/3): {e}"); time.sleep(1)
            if i==2:
                if "code\":-4120" in str(e) or "Algo Order API" in str(e):
                    ALGO_ORDER_UNSUPPORTED = True
                    log("[INFO] 이 계정은 기본 엔드포인트의 조건부 주문 미지원으로 감지됨. 이후 보호주문 재시도 생략")
                return False
    return True

def enforce_local_sl_tp(price: float):
    """
    거래소 조건부 주문이 불가능한 계정에서도 동작하도록
    DB의 OPEN 포지션 SL/TP를 로컬에서 감시해 시장가 청산.
    """
    t = get_latest_open_trade()
    p = fetch_current_position()
    if not t or not p:
        return

    sl = float(t.get("sl_price") or 0.0)
    tp = float(t.get("tp_price") or 0.0)
    if sl <= 0 or tp <= 0:
        return

    action = str(t.get("action") or "").lower()
    hit = None
    if action == "long":
        if price <= sl:
            hit = "SL(local)"
        elif price >= tp:
            hit = "TP(local)"
    elif action == "short":
        if price >= sl:
            hit = "SL(local)"
        elif price <= tp:
            hit = "TP(local)"

    if hit:
        log(f"[LOCAL EXIT] {hit} 충족 → 시장가 청산 시도")
        flatten_position_if_any()

def place_orders(decision: Dict[str,Any], price: float, market: Dict[str,Any]):
    if not is_allowed_trading_time():
        log("[TIME] 허용 거래 시간대 아님(UTC) → 신규 진입 스킵")
        return None

    # 킬스위치
    if today_realized_pnl() <= -DAILY_MAX_LOSS_USDT:
        log("[KILL] 일일 손실 한도 도달. 신규 진입 금지"); return None

    # 유동성/스프레드 필터
    ok, spread, depth_usdt, _ = liquidity_ok(SYMBOL, max_spread=LIQ_MAX_SPREAD, min_depth_usdt=LIQ_MIN_DEPTH_USDT)
    if not ok:
        log(f"유동성/스프레드 부족: spread={spread:.4%}, depth≈{int(depth_usdt)}USDT → 대기")
        return None

    # 레버리지/마진 모드
    lev = set_margin_and_leverage(decision["recommended_leverage"])

    # 잔고
    bal = exchange.fetch_balance()
    print(bal.get("USDT"))
    free = float(bal.get("USDT",{}).get("free",0.0))

    # SL/TP 가격
    sl_pct = float(decision["stop_loss_percentage"])
    tp_pct = float(decision["take_profit_percentage"])
    if decision["direction"]=="LONG":
        sl_price = price * (1 - sl_pct)
        tp_price = price * (1 + tp_pct)
        stop_distance = price - sl_price
    else:
        sl_price = price * (1 + sl_pct)
        tp_price = price * (1 - tp_pct)
        stop_distance = sl_price - price
    stop_distance_pct = stop_distance / price
    if stop_distance_pct <= 0:
        log("SL 계산 오류 → 스킵"); return None
    rr = tp_pct / max(sl_pct, 1e-9)
    if rr + 1e-6 < MIN_RR:
        log(f"RR 부족(rr={rr:.2f} < {MIN_RR:.2f}) → 스킵")
        return None

    # 정밀도/최소주문
    exchange.load_markets()
    px_entry = float(exchange.price_to_precision(SYMBOL, price))
    mkt = market
    min_cost = (mkt.get('limits',{}).get('cost',{}) or {}).get('min') or 5

    # === 잔고 적응형 리스크 사이징 ===
    # 위험금액 = Equity * RISK_PCT * AI스케일, 최소주문 체결 가능성과 잔고 상한을 동시에 반영
    ai_scale = float(decision["recommended_position_size"])
    risk_amount = free * clamp(RISK_PCT * ai_scale, 0.003, 0.015)  # 0.3%~1.5% 사이

    # 최소주문(min_cost)이 성립하려면 필요한 최소 위험금액(근사):
    # min_risk_needed ≈ min_cost * stop_distance_pct
    min_risk_needed = float(min_cost) * float(stop_distance_pct)
    risk_amount = max(risk_amount, MIN_RISK_USDT, min_risk_needed * 1.2)  # 20% 버퍼

    # 과도한 리스크 상한: 절대값 + 잔고비율 중 더 보수적인 값 적용
    risk_cap = min(MAX_RISK_USDT, free * clamp(MAX_RISK_PCT_OF_FREE, 0.005, 0.05))
    risk_amount = min(risk_amount, max(0.0, risk_cap))
    if risk_amount <= 0:
        log("유효 리스크 금액이 0 이하 → 스킵")
        return None

    # 수량 = 위험금액 / 가격변동폭
    # (주의) 손실은 레버리지 배수가 아니라 가격변동폭*수량으로 결정됨
    raw_amount = risk_amount / max(1e-9, stop_distance)
    amount   = float(exchange.amount_to_precision(SYMBOL, raw_amount))
    notional = amount * px_entry
    if notional < min_cost:
        amount = float(exchange.amount_to_precision(SYMBOL, (min_cost/px_entry)))
        notional = amount * px_entry
        if notional < min_cost:
            log("최소 주문 금액 미만 → 스킵"); return None

    # === 마진 캡: 가용자금 대비 최대 명목가 제한 ===
    #   초기마진 ≈ 명목가 / 레버리지, 수수료/버퍼 고려해 80~90% 정도로 캡을 둡니다.
    margin_cushion = clamp(MAX_NOTIONAL_FRAC, 0.1, 0.9)
    max_notional_by_margin = free * lev * margin_cushion  # 예: free=20, lev=5 → 85% * 100 = 85 USDT
    if notional > max_notional_by_margin:
        capped_amount = max_notional_by_margin / px_entry
        amount = float(exchange.amount_to_precision(SYMBOL, capped_amount))
        notional = amount * px_entry
        if notional < min_cost:
            log("마진 캡 적용 후 최소주문 미만 → 스킵")
            return None
        log(f"마진 캡 적용: amount 조정, notional≈{notional:.2f}USDT (max≈{max_notional_by_margin:.2f}USDT)")
    
    if amount <= 0:
        log("정밀도/캡 적용 후 수량=0 → 스킵")
        return None
        
    # 중복진입 방지
    if fetch_current_position():
        log("이미 포지션 보유 중 → 신규 진입 생략")
        return None

    # === 진입 ===
    log(f"진입 시도: {decision['direction']} amt={amount} px={px_entry} SL={sl_price:.6f} TP={tp_price:.6f} lev={lev}")
    if decision["direction"]=="LONG":
        order = exchange.create_market_buy_order(SYMBOL, amount, params={
            **({"positionSide":"LONG"} if HEDGE_MODE else {})
        })
    else:
        order = exchange.create_market_sell_order(SYMBOL, amount, params={
            **({"positionSide":"SHORT"} if HEDGE_MODE else {})
        })
    entry_price = float(order.get("average") or order.get("price") or px_entry)

    # 보호주문(재시도 포함)
    sl_price = float(exchange.price_to_precision(SYMBOL, sl_price))
    tp_price = float(exchange.price_to_precision(SYMBOL, tp_price))
    protected = place_protective_orders(decision["direction"], amount, sl_price, tp_price)
    if not protected:
        log("[WARN] 거래소 보호주문 생성 실패. 로컬 SL/TP 감시 모드로 운영")

    # DB 저장
    trade_id = save_trade({
        "action": decision["direction"].lower(),
        "entry_price": entry_price,
        "amount": amount,
        "leverage": lev,
        "sl_price": sl_price, "tp_price": tp_price,
        "sl_percentage": sl_pct, "tp_percentage": tp_pct,
        "position_size_percentage": (risk_amount / max(1e-9, free)),
        "investment_amount": notional
    })
    return trade_id, entry_price, amount

def close_if_no_position_update_db():
    in_db = get_latest_open_trade()
    on_ex = fetch_current_position()
    if in_db and not on_ex:
        price = float(exchange.fetch_ticker(SYMBOL)["last"])
        entry = float(in_db["entry_price"]); amt = float(in_db["amount"])
        if in_db["action"]=="long":
            pl = (price-entry)*amt; pl_pct=(price/entry-1)*100 if entry else 0.0
        else:
            pl = (entry-price)*amt; pl_pct=(1-price/entry)*100 if entry else 0.0
        update_trade_status(in_db["id"], "CLOSED", price, now_iso(), pl, pl_pct)
        log("무포지션 동기화: DB 포지션 종결")

# =========================
# 메인 루프
# =========================
def main():
    global GLOBAL_IDLE_COOLDOWN_UNTIL
    setup_db()
    market = ensure_market()
    log("=== Scalper Safe Bot Started ===")
    log(f"Symbol: {SYMBOL}")
    log(f"[COIN] active={BASE}, symbol={SYMBOL}")

    last_heavy = 0

    # ✅ 런타임 상태 변수
    last_entry_time = 0.0
    last_entry_side = None
    no_pos_streak   = 0
    cooldown_until = 0.0
    idle_no_pos_streak = 0
    flip_signal_streak = 0
    idle_cycle_seen_coins = set()
    while True:
        try:
            now = time.time()

            # 재시작 복구
            sync_position_db()

            if last_entry_time == 0:
                _pos = fetch_current_position()
                if _pos:
                    last_entry_time = time.time()

            # 최신가
            ticker = exchange.fetch_ticker(SYMBOL)
            price = float(ticker.get("last") or ticker.get("close") or 0)
            if price <= 0:
                time.sleep(TICKER_SEC);  continue

            # 보호주문 미지원 계정 대응: 로컬 SL/TP 강제청산
            enforce_local_sl_tp(price)

            # 주기적 마켓정보 갱신
            if now - last_heavy > HEAVY_SEC:
                market = ensure_market()
                last_heavy = now

            # ✅ 보호주문 체결 감지 → DB 닫기 반영
            close_if_no_position_update_db()

            # === 오더북 체크(프롬프트/로그용) ===
            liq_ok, spread, depth_usdt, best_bid = liquidity_ok(SYMBOL, max_spread=LIQ_MAX_SPREAD, min_depth_usdt=LIQ_MIN_DEPTH_USDT)
            log(f"[COIN] active={BASE}, symbol={SYMBOL}")
            log(f"OB check → ok={liq_ok}, spread={spread:.4%}, depth≈{int(depth_usdt)} USDT")

            # 유동성 미달 + 무포지션이면 AI 호출을 생략해 비용/노이즈를 줄임
            if (not liq_ok) and (not fetch_current_position()):
                wait_sec = max(60, TICKER_SEC)
                log(f"[LIQ] 유동성 미달 & 무포지션 → AI 호출 생략, 대기 {wait_sec}s")
                time.sleep(wait_sec)
                continue

            # 허용 시간대 밖 + 무포지션이면 AI 호출 자체를 생략해 비용 절감
            if not is_allowed_trading_time() and not fetch_current_position():
                wait_sec = 60
                log(f"[TIME] 비거래 시간대(UTC) & 무포지션 → AI 호출 생략, 대기 {wait_sec}s")
                time.sleep(wait_sec)
                continue

            # 쿨다운 중이고 무포지션이면 AI 호출 자체를 생략해 비용을 절감
            if time.time() < cooldown_until and not fetch_current_position():
                remaining = int(cooldown_until - time.time())
                wait_sec = min(60, max(30, remaining))
                log(f"[COOLDOWN] 무포지션 쿨다운 중({remaining}s 남음) → AI 호출 생략, 대기 {wait_sec}s")
                time.sleep(wait_sec)
                continue

            # 모든 로테이션 코인이 무포지션으로 한 바퀴 돈 경우 글로벌 쿨다운
            if ENABLE_GLOBAL_IDLE_CYCLE_COOLDOWN and (time.time() < GLOBAL_IDLE_COOLDOWN_UNTIL) and (not fetch_current_position()):
                remaining = int(GLOBAL_IDLE_COOLDOWN_UNTIL - time.time())
                if breakout_wakeup_signal(SYMBOL):
                    GLOBAL_IDLE_COOLDOWN_UNTIL = 0.0
                    log(f"[WAKEUP] breakout 감지({SYMBOL}) → 글로벌 무포지션 쿨다운 해제")
                else:
                    wait_sec = min(120, max(60, remaining))
                    log(f"[GLOBAL_COOLDOWN] 전 코인 무포지션 사이클 보호 중({remaining}s 남음) → AI 호출 생략, 대기 {wait_sec}s")
                    time.sleep(wait_sec)
                    continue

            # === 분석 → 의사결정 (항상 먼저) ===
            snapshot = build_snapshot()
            snapshot["orderbook"] = {
                "ok": bool(liq_ok),
                "spread": float(spread),
                "depth_usdt": float(depth_usdt),
            }
            decision = ai_decide(snapshot)
            log(f"[SIGNAL] coin={BASE}, symbol={SYMBOL}, direction={decision.get('direction', 'NO_POSITION')}")

            # RANGE 구간에서는 신규 진입을 기본적으로 차단
            if RANGE_BLOCK_ENTRY and decision["direction"] in ("LONG", "SHORT") and is_range_regime(snapshot) and not fetch_current_position():
                decision["direction"] = "NO_POSITION"
                decision["reasoning"] = f"{decision.get('reasoning','')} [RANGE_BLOCK]"

            if time.time() < cooldown_until and decision["direction"] != "NO_POSITION":
                remaining = int(cooldown_until - time.time())
                log(f"[COOLDOWN] 연속 손실 보호 중({remaining}s 남음) → NO_POSITION으로 강제")
                decision["direction"] = "NO_POSITION"
                decision["reasoning"] = f"{decision.get('reasoning','')} [COOLDOWN_ACTIVE]"

            # ---- Fallback Guard: 모델이 근거 없이 NO_POSITION만 내면 소형 추세추종 진입으로 대체
            ob = snapshot.get("orderbook", {})
            liq_ok_flag = bool(ob.get("ok", False))
            data_ok = has_enough_data(snapshot)
            dd_ok = today_realized_pnl() > -DAILY_MAX_LOSS_USDT  # 일일 손실 한도 미도달

            if ENABLE_FALLBACK and decision["direction"] == "NO_POSITION":
                if liq_ok_flag and data_ok and dd_ok:
                    fb = fallback_decision_from_snapshot(snapshot)
                    if fb["direction"] != "NO_POSITION":
                        log(f"[GUARD] Model said NO_POSITION, but conditions look fine → using fallback: {fb['direction']} (small size)")
                        decision = fb
                    else:
                        log("[GUARD] Fallback도 NO_POSITION → 그대로 대기")
                else:
                    # 진짜로 무진입이어야 하는 상황이면 이유 로그
                    log(f"NO_POSITION 이유: {(decision.get('reasoning') or '').strip()}")
            elif decision["direction"] == "NO_POSITION":
                log(f"NO_POSITION 이유: {(decision.get('reasoning') or '').strip()}")

            # 분석 기록 (trade_id는 체결 후 연결)
            analysis_id = save_ai_analysis({
                "current_price": snapshot["current_price"],
                "direction": decision["direction"],
                "recommended_position_size": decision["recommended_position_size"],
                "recommended_leverage": decision["recommended_leverage"],
                "stop_loss_percentage": decision["stop_loss_percentage"],
                "take_profit_percentage": decision["take_profit_percentage"],
                "reasoning": decision.get("reasoning","")
            })

            # === NO_POSITION: 전량 취소/평탄화 후 대기 ===
            if decision["direction"] == "NO_POSITION":
                onpos = fetch_current_position()

                if not onpos:
                    cancel_all_orders_for_symbol()
                    no_pos_streak = 0
                    idle_no_pos_streak += 1
                    idle_cycle_seen_coins.add(BASE)
                    log("AI: NO_POSITION (보유 없음) → 주문만 정리하고 대기")

                    if ENABLE_GLOBAL_IDLE_CYCLE_COOLDOWN and len(idle_cycle_seen_coins) >= len(ROTATION_COINS):
                        GLOBAL_IDLE_COOLDOWN_UNTIL = max(
                            GLOBAL_IDLE_COOLDOWN_UNTIL,
                            time.time() + max(60, GLOBAL_IDLE_CYCLE_COOLDOWN_SEC)
                        )
                        log(f"[GLOBAL_COOLDOWN] 전 코인 무포지션 확인({len(ROTATION_COINS)}개) → {GLOBAL_IDLE_CYCLE_COOLDOWN_SEC}s 보호")
                        idle_cycle_seen_coins.clear()

                    if ROTATE_ON_IDLE and idle_no_pos_streak >= max(1, ROTATE_IDLE_STREAK_N):
                        switched = rotate_to_next_coin()
                        if switched:
                            idle_no_pos_streak = 0
                            no_pos_streak = 0
                            flip_signal_streak = 0
                            last_heavy = 0
                            log(f"[ROTATE] 현재 심볼: {SYMBOL}")

                    wait_sec = idle_no_position_wait_seconds(idle_no_pos_streak)
                    log(f"[COST] 무포지션 연속 {idle_no_pos_streak}회 → 대기 {wait_sec}s")
                    time.sleep(wait_sec)
                    continue

                # 보유 중이면: 최소 홀드 확인
                age = time.time() - (last_entry_time or 0)
                if age < MIN_HOLD_SEC:
                    sl_pct = float(decision["stop_loss_percentage"])
                    tp_pct = float(decision["take_profit_percentage"])
                    cancel_all_orders_for_symbol()

                    if onpos["side"] == "long":
                        new_sl_price = price * (1 - sl_pct)
                        new_tp_price = price * (1 + tp_pct)
                        dir_for_protect = "LONG"
                    else:
                        new_sl_price = price * (1 + sl_pct)
                        new_tp_price = price * (1 - tp_pct)
                        dir_for_protect = "SHORT"

                    new_sl_price = float(exchange.price_to_precision(SYMBOL, new_sl_price))
                    new_tp_price = float(exchange.price_to_precision(SYMBOL, new_tp_price))
                    ok_protect = place_protective_orders(dir_for_protect, onpos["amount"], new_sl_price, new_tp_price)
                    if not ok_protect:
                        log("[WARN] 보호주문 재설정 실패. 로컬 SL/TP 감시 계속")
                    update_open_trade_levels(new_sl_price, new_tp_price, sl_pct, tp_pct)

                    log(f"AI: NO_POSITION (보유 중, age={age:.1f}s < {MIN_HOLD_SEC}s) → SL/TP만 재설정하고 유지")
                    time.sleep(loop_wait_seconds())
                    continue

                # 최소 홀드 지난 경우: 연속 NO_POSITION 체크
                no_pos_streak += 1
                log(f"AI: NO_POSITION (연속 {no_pos_streak}/{NO_POS_STREAK_N})")

                if no_pos_streak < NO_POS_STREAK_N:
                    sl_pct = float(decision["stop_loss_percentage"])
                    tp_pct = float(decision["take_profit_percentage"])
                    cancel_all_orders_for_symbol()
                    if onpos["side"] == "long":
                        new_sl_price = price * (1 - sl_pct)
                        new_tp_price = price * (1 + tp_pct)
                        dir_for_protect = "LONG"
                    else:
                        new_sl_price = price * (1 + sl_pct)
                        new_tp_price = price * (1 - tp_pct)
                        dir_for_protect = "SHORT"
                    new_sl_price = float(exchange.price_to_precision(SYMBOL, new_sl_price))
                    new_tp_price = float(exchange.price_to_precision(SYMBOL, new_tp_price))
                    ok_protect = place_protective_orders(dir_for_protect, onpos["amount"], new_sl_price, new_tp_price)
                    if not ok_protect:
                        log("[WARN] 보호주문 재설정 실패. 로컬 SL/TP 감시 계속")
                    update_open_trade_levels(new_sl_price, new_tp_price, sl_pct, tp_pct)

                    time.sleep(loop_wait_seconds())
                    continue

                # 임계치 도달 → 평탄화
                if ENABLE_LOSS_HOLD_ON_NO_POS:
                    pnl_pct = unrealized_pnl_pct(onpos, price)
                    age = time.time() - (last_entry_time or time.time())
                    if pnl_pct is not None and pnl_pct <= -abs(LOSS_HOLD_TRIGGER_PCT) and age < LOSS_HOLD_MAX_SEC:
                        log(f"AI: NO_POSITION 연속 임계 도달 but 손실 유예 적용(pnl={pnl_pct:.3f}%, age={age:.0f}s) → 유지")
                        no_pos_streak = max(0, NO_POS_STREAK_N - 1)
                        time.sleep(loop_wait_seconds())
                        continue

                log("AI: NO_POSITION 연속 임계 도달 → 모든 미체결 취소 및 보유 포지션 평탄화")
                cancel_all_orders_for_symbol()
                flatten_position_if_any()
                no_pos_streak = 0
                time.sleep(ANALYZE_SEC)
                continue

            # === 보유 중 로직: 동일 방향이면 SL/TP만 재설정, 반대면 FLIP ===
            onpos = fetch_current_position()
            if onpos:
                cancel_all_orders_for_symbol()
                sl_pct = float(decision["stop_loss_percentage"])
                tp_pct = float(decision["take_profit_percentage"])
                want_long = (decision["direction"] == "LONG")

                if want_long:
                    new_sl_price = price * (1 - sl_pct)
                    new_tp_price = price * (1 + tp_pct)
                else:
                    new_sl_price = price * (1 + sl_pct)
                    new_tp_price = price * (1 - tp_pct)

                new_sl_price = float(exchange.price_to_precision(SYMBOL, new_sl_price))
                new_tp_price = float(exchange.price_to_precision(SYMBOL, new_tp_price))

                same_dir = (onpos["side"] == "long" and want_long) or (onpos["side"] == "short" and not want_long)
                if same_dir:
                    flip_signal_streak = 0
                    # set_margin_and_leverage(decision["recommended_leverage"])  # 필요시
                    ok_protect = place_protective_orders(decision["direction"], onpos["amount"], new_sl_price, new_tp_price)
                    if not ok_protect:
                        log("[WARN] 보호주문 재설정 실패. 로컬 SL/TP 감시 계속")
                    update_open_trade_levels(new_sl_price, new_tp_price, sl_pct, tp_pct)
                    log(f"보유 중 동일 방향: SL/TP 재설정 완료 (SL={new_sl_price}, TP={new_tp_price})")
                    time.sleep(loop_wait_seconds())
                    continue
                else:
                    flip_signal_streak += 1
                    if flip_signal_streak < max(1, FLIP_CONFIRM_TICKS):
                        log(f"보유 중 반대 방향 신호 감지({flip_signal_streak}/{FLIP_CONFIRM_TICKS}) → 확인 대기")
                        time.sleep(loop_wait_seconds())
                        continue
                    log("보유 중 반대 방향 신호 확인 완료 → FLIP: 평탄화 후 신규 진입")
                    flip_signal_streak = 0
                    flatten_position_if_any()
                    for _ in range(5):
                        if not fetch_current_position():
                            break
                        time.sleep(0.2)

            # === 신규 진입 전 공통 정리 ===
            cancel_all_orders_for_symbol()

            cur = fetch_current_position()
            want_long = (decision["direction"] == "LONG")
            if cur:
                is_long = (cur["side"] == "long")
                if is_long != want_long:
                    log("방향 반대(안전망) → 평탄화")
                    flatten_position_if_any()
                    for _ in range(5):
                        if not fetch_current_position():
                            break
                        time.sleep(0.2)

            # === 신규 진입 실행 ===
            if ENABLE_LOSS_STREAK_COOLDOWN:
                loss_streak = get_recent_consecutive_losses(limit=max(5, MAX_CONSEC_LOSSES + 2))
                if loss_streak >= MAX_CONSEC_LOSSES:
                    cooldown_until = max(cooldown_until, time.time() + COOLDOWN_SEC_AFTER_LOSS_STREAK)
                    log(f"[RISK] 최근 연속 손실 {loss_streak}회 → {COOLDOWN_SEC_AFTER_LOSS_STREAK}s 쿨다운")
                    time.sleep(ANALYZE_SEC)
                    continue

            placed = place_orders(decision, price, market)
            if placed:
                trade_id, entry_price, amount = placed
                conn = db_conn(); cur = conn.cursor()
                cur.execute("UPDATE ai_analysis SET trade_id=? WHERE id=?", (trade_id, analysis_id))
                conn.commit(); conn.close()
                log(f"진입 완료: trade_id={trade_id}, entry={entry_price}, amt={amount}")

                last_entry_time = time.time()
                last_entry_side = decision["direction"]
                no_pos_streak = 0
                idle_no_pos_streak = 0
                flip_signal_streak = 0
                idle_cycle_seen_coins.clear()
            else:
                log("주문 미체결/스킵")

            time.sleep(loop_wait_seconds())

        except ccxt.BaseError as ex:
            log(f"[CCXT] {type(ex).__name__}: {ex}"); time.sleep(2)
        except Exception as e:
            log(f"[ERROR] {e}\n{traceback.format_exc()}"); time.sleep(2)

if __name__ == "__main__":
    main()
