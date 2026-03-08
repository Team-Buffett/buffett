import os
import sys
import time
import signal
import subprocess
from datetime import datetime


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUTOTRADE_SCRIPT = os.path.join(BASE_DIR, "autotrade.py")
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def parse_coins():
    raw = os.getenv("TRADING_COINS", "BTC,ETH,XRP,SOL,DOGE")
    coins = []
    for c in raw.split(","):
        cc = c.strip().upper()
        if cc and cc not in coins:
            coins.append(cc)
    return coins[:5]


def spawn_bot(coin):
    env = os.environ.copy()
    env["COIN_NAME"] = coin
    log_path = os.path.join(LOG_DIR, f"{coin.lower()}_autotrade.log")
    log_file = open(log_path, "a", encoding="utf-8")
    p = subprocess.Popen(
        [sys.executable, AUTOTRADE_SCRIPT],
        cwd=BASE_DIR,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    print(f"[{now()}] started {coin} pid={p.pid} log={log_path}", flush=True)
    return {"proc": p, "log": log_file, "coin": coin, "log_path": log_path}


def stop_bot(bot):
    p = bot["proc"]
    if p.poll() is None:
        p.terminate()
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()
    bot["log"].close()


def main():
    coins = parse_coins()
    if not coins:
        raise RuntimeError("No coins configured. Set TRADING_COINS, e.g. BTC,ETH,XRP,SOL,DOGE")

    print(f"[{now()}] multi bot start coins={coins}", flush=True)
    bots = {coin: spawn_bot(coin) for coin in coins}
    running = True

    def handle_signal(_sig, _frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        while running:
            time.sleep(5)
            for coin, bot in list(bots.items()):
                p = bot["proc"]
                code = p.poll()
                if code is not None:
                    bot["log"].close()
                    print(f"[{now()}] {coin} exited code={code}; restarting", flush=True)
                    bots[coin] = spawn_bot(coin)
    finally:
        print(f"[{now()}] stopping all bots", flush=True)
        for bot in bots.values():
            stop_bot(bot)


if __name__ == "__main__":
    main()
