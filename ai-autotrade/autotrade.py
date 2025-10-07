# -*- coding: utf-8 -*-
"""
단타 스캘핑 안전강화 버전 (풀 스크립트)
- 유동성/스프레드 필터
- SL 거리 기반 포지션 사이징 (프롬프트 의도 충실)
- reduceOnly SL/TP + 실패 시 재시도
- 일일 손실 한도(킬스위치) + 중복진입 방지
- 재시작 복구 (거래소 포지션 ↔ DB 동기화)
- 정밀도/최소주문/격리/레버리지 설정
"""

import os, time, math, json, sqlite3, traceback
from datetime import datetime
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
MIN_HOLD_SEC    = int(os.getenv("MIN_HOLD_SEC", "60"))  # 최소 보유 25초 예시
NO_POS_STREAK_N = int(os.getenv("NO_POS_STREAK_N", "5"))  # NO_POSITION 5틱 연속 시 청산
OPENAI_API_KEY   = require_env("OPENAI_API_KEY")
BINANCE_API_KEY  = require_env("BINANCE_API_KEY")
BINANCE_SECRET   = require_env("BINANCE_SECRET_KEY")

# 파라미터 (필요시 .env에서 조정)
AI_MODEL      = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
RISK_PCT      = float(os.getenv("RISK_PCT", "0.012"))          # 트레이드당 계좌 리스크 1.2%
MAX_LEVERAGE  = int(os.getenv("MAX_LEVERAGE", "15"))
DAILY_MAX_LOSS_USDT = float(os.getenv("DAILY_MAX_LOSS_USDT", "100"))
HEDGE_MODE    = os.getenv("HEDGE_MODE", "false").lower() == "true"

TICKER_SEC    = 30
HEAVY_SEC     = 15
ANALYZE_SEC   = 10

# 코인명
COIN_NAME_PATH = os.path.join(BASE_DIR, "txt", "coinName.txt")
if os.path.exists(COIN_NAME_PATH):
    BASE = open(COIN_NAME_PATH, "r", encoding="utf-8").read().strip().replace(" ", "").upper()
else:
    BASE = "XCN"

SYMBOL = f"{BASE}/USDT:USDT"   # USDT-M Perpetual
DB_FILE = os.path.join(BASE_DIR, "db", f"{BASE.lower()}_trading.db")

# =========================
# 프롬프트 로드
# =========================
SP_PATH = os.path.join(BASE_DIR, "txt", "system_prompt.txt")
if os.path.exists(SP_PATH):
    SYSTEM_PROMPT = open(SP_PATH, "r", encoding="utf-8").read()
else:
    SYSTEM_PROMPT = """You are the world’s top high-frequency crypto scalper, specializing in XCN/USDT perpetual futures trading on Binance using the ChatGPT API.
Use 1m/3m/5m momentum. Output JSON with fields: direction, recommended_position_size, recommended_leverage, stop_loss_percentage, take_profit_percentage, reasoning."""

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

# =========================
# DB
# =========================
def db_conn():
    conn = sqlite3.connect(DB_FILE, timeout=10, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
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
        pos = fetch_current_position()  # 이미 너 코드에 있음
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

def build_snapshot():
    data = {}
    for tf in ["1m","3m","5m"]:
        try: data[tf] = fetch_candles(tf).to_dict(orient="records")
        except Exception as e:
            log(f"캔들 수집 실패 {tf}: {e}"); data[tf] = []
    price = float(exchange.fetch_ticker(SYMBOL)["last"])
    # 최근 20개 성과(보조)
    conn = db_conn(); cur = conn.cursor()
    cur.execute("""
        SELECT profit_loss, profit_loss_percentage FROM trades
        WHERE status='CLOSED' ORDER BY timestamp DESC LIMIT 20
    """); rows = cur.fetchall(); conn.close()
    perf = {
        "closed_count": len(rows),
        "win_count": sum(1 for r in rows if (r[0] or 0)>0),
        "loss_count": sum(1 for r in rows if (r[0] or 0)<0),
        "avg_pl_pct": (sum((r[1] or 0) for r in rows)/len(rows)) if rows else 0.0
    }
    return {"timestamp":now_iso(), "symbol":SYMBOL, "current_price":price,
            "timeframes":data, "recent_performance":perf}

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
    lev       = int(clamp(_to_int(d.get("recommended_leverage"), 5), 1, MAX_LEVERAGE))

    sl_pct = _to_float(d.get("stop_loss_percentage"), 0.003)
    tp_pct = _to_float(d.get("take_profit_percentage"), 0.006)

    # 만약 퍼센트를 3(=300%) 같이 보냈다면 1보다 큰 값은 %로 간주해서 보정
    if sl_pct > 1: sl_pct = sl_pct / 100.0
    if tp_pct > 1: tp_pct = tp_pct / 100.0

    sl_pct = clamp(sl_pct, 0.001, 0.05)
    tp_pct = clamp(tp_pct, 0.002, 0.2)

    return {
        "direction": direction,
        "recommended_position_size": pos_scale,
        "recommended_leverage": lev,
        "stop_loss_percentage": sl_pct,
        "take_profit_percentage": tp_pct,
        "reasoning": d.get("reasoning") or ""
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

def _to_int(x, default):
    """모델 응답의 숫자/문자/None을 안전하게 int로 변환"""
    try:
        if x is None:
            return default
        if isinstance(x, str):
            s = x.strip()
            if not s:
                return default
            return int(float(s))
        return int(x)
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
            if i==2: raise
    for i in range(3):
        try:
            exchange.create_order(SYMBOL, 'TAKE_PROFIT_MARKET', side_tp, amount, None, params(tp_price))
            break
        except Exception as e:
            log(f"TP 주문 실패 재시도({i+1}/3): {e}"); time.sleep(1)
            if i==2: raise

def place_orders(decision: Dict[str,Any], price: float, market: Dict[str,Any]):
    # 킬스위치
    if today_realized_pnl() <= -DAILY_MAX_LOSS_USDT:
        log("[KILL] 일일 손실 한도 도달. 신규 진입 금지"); return None

    # 유동성/스프레드 필터
    ok, spread, depth_usdt, _ = liquidity_ok(SYMBOL, max_spread=0.002, min_depth_usdt=2000)
    if not ok:
        log(f"유동성/스프레드 부족: spread={spread:.4%}, depth≈{int(depth_usdt)}USDT → 대기")
        return None

    # 레버리지/마진 모드
    lev = set_margin_and_leverage(decision["recommended_leverage"])

    # 잔고
    bal = exchange.fetch_balance()
    print(bal["USDT"])
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

    # === SL 거리 기반 포지션 사이징 ===
    # 위험금액 = Equity * RISK_PCT * AI스케일 (0.5~2.0)
    ai_scale = float(decision["recommended_position_size"])
    risk_amount = free * clamp(RISK_PCT * ai_scale, 0.005, 0.03)  # 0.5%~3% 사이
    # 수량 = (위험금액 / (stop_distance_pct * price)) * 레버리지
    raw_amount = (risk_amount / (stop_distance_pct * price)) * lev

    # 정밀도/최소주문
    exchange.load_markets()
    px_entry = float(exchange.price_to_precision(SYMBOL, price))
    amount   = float(exchange.amount_to_precision(SYMBOL, raw_amount))
    mkt = market
    min_cost = (mkt.get('limits',{}).get('cost',{}) or {}).get('min') or 5
    notional = amount * px_entry
    if notional < min_cost:
        amount = float(exchange.amount_to_precision(SYMBOL, (min_cost/px_entry)))
        notional = amount * px_entry
        if notional < min_cost:
            log("최소 주문 금액 미만 → 스킵"); return None

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
    place_protective_orders(decision["direction"], amount, sl_price, tp_price)

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
    setup_db()
    market = ensure_market()
    log("=== Scalper Safe Bot Started ===")
    log(f"Symbol: {SYMBOL}")

    last_heavy = 0

    # ✅ 런타임 상태 변수
    last_entry_time = 0.0
    last_entry_side = None
    no_pos_streak   = 0
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

            # 주기적 마켓정보 갱신
            if now - last_heavy > HEAVY_SEC:
                market = ensure_market()
                last_heavy = now

            # ✅ 보호주문 체결 감지 → DB 닫기 반영 (추가)
            close_if_no_position_update_db()

            # === 분석 → 의사결정 (항상 먼저) ===
            snapshot = build_snapshot()
            decision = ai_decide(snapshot)
            if decision["direction"] == "NO_POSITION":
                log(f"NO_POSITION 이유: { (decision.get('reasoning') or '')[:180] }")

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
                    # 무포지션이면 주문만 정리하고 대기
                    cancel_all_orders_for_symbol()
                    no_pos_streak = 0
                    log("AI: NO_POSITION (보유 없음) → 주문만 정리하고 대기")
                    time.sleep(ANALYZE_SEC)
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
                    place_protective_orders(dir_for_protect, onpos["amount"], new_sl_price, new_tp_price)

                    log(f"AI: NO_POSITION (보유 중, age={age:.1f}s < {MIN_HOLD_SEC}s) → SL/TP만 재설정하고 유지")
                    time.sleep(TICKER_SEC)
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
                    place_protective_orders(dir_for_protect, onpos["amount"], new_sl_price, new_tp_price)

                    time.sleep(TICKER_SEC)
                    continue

                # 임계치 도달 → 평탄화
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
                    # (선택) 레버리지 재설정
                    # set_margin_and_leverage(decision["recommended_leverage"])
                    place_protective_orders(decision["direction"], onpos["amount"], new_sl_price, new_tp_price)
                    log(f"보유 중 동일 방향: SL/TP 재설정 완료 (SL={new_sl_price}, TP={new_tp_price})")
                    time.sleep(TICKER_SEC)
                    continue
                else:
                    log("보유 중 반대 방향 신호 → FLIP: 평탄화 후 신규 진입")
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
            else:
                log("주문 미체결/스킵")
            time.sleep(TICKER_SEC)
        except ccxt.BaseError as ex:
            log(f"[CCXT] {type(ex).__name__}: {ex}"); time.sleep(2)
        except Exception as e:
            log(f"[ERROR] {e}\n{traceback.format_exc()}"); time.sleep(2)

if __name__ == "__main__":
    main()