import ccxt
import os
import math
import time
import pandas as pd
from dotenv import load_dotenv
load_dotenv("buffett-config/.env")
from openai import OpenAI
from datetime import datetime

# 바이낸스 세팅
api_key = os.getenv("BINANCE_API_KEY")
secret = os.getenv("BINANCE_SECRET_KEY")
exchange = ccxt.binance({
    'apiKey': api_key,
    'secret': secret,
    'enableRateLimit': True,
    'options': {
        'defaultType': 'future',
        'adjustForTimeDifference': True
    }
})
symbol = "BTC/USDT"
client = OpenAI()

print("\n=== Bitcoin Trading Bot Started ===")
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("Trading Pair:", symbol)
print("Leverage: 5x")
print("SL/TP: ±0.5%")
print("Multi Timeframe Analysis: 15m, 1h, 4h")
print("===================================\n")

# 멀티 타임프레임 데이터 수집 함수
def fetch_multi_timeframe_data():
    # 타임프레임별 데이터 수집
    timeframes = {
        "15m": {"timeframe": "15m", "limit": 96},  # 24시간 (15분 * 96)
        "1h": {"timeframe": "1h", "limit": 48},    # 48시간 (1시간 * 48)
        "4h": {"timeframe": "4h", "limit": 30}     # 5일 (4시간 * 30)
    }
    multi_tf_data = {}
    for tf_name, tf_params in timeframes.items():
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=tf_params["timeframe"], limit=tf_params["limit"])
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            multi_tf_data[tf_name] = df
            print(f"Collected {tf_name} data: {len(df)} candles")
        except Exception as e:
            print(f"Error fetching {tf_name} data: {e}")
    return multi_tf_data

while True:
    try:
        # 현재 시간 및 가격 조회
        current_time = datetime.now().strftime('%H:%M:%S')
        current_price = exchange.fetch_ticker(symbol)['last']
        print(f"\n[{current_time}] Current BTC Price: ${current_price:,.2f}")

        # 포지션 확인
        current_side = None
        amount = 0
        positions = exchange.fetch_positions([symbol])
        for position in positions:
            if position['symbol'] == 'BTC/USDT:USDT':
                amt = float(position['info']['positionAmt'])
                if amt > 0:
                    current_side = 'long'
                    amount = amt
                elif amt < 0:
                    current_side = 'short'
                    amount = abs(amt)
        if current_side:
            print(f"Current Position: {current_side.upper()} {amount} BTC")
        else:
            # 포지션이 없을 경우, 남아있는 미체결 주문 취소
            try:
                open_orders = exchange.fetch_open_orders(symbol)
                if open_orders:
                    for order in open_orders:
                        exchange.cancel_order(order['id'], symbol)
                    print("Cancelled remaining open orders for", symbol)
                else:
                    print("No remaining open orders to cancel.")
            except Exception as e:
                print("Error cancelling orders:", e)
            time.sleep(5)
            print("No position. Analyzing market...")

            # 멀티 타임프레임 차트 데이터 수집
            multi_tf_data = fetch_multi_timeframe_data()

            # AI 분석을 위한 데이터 준비
            market_analysis = {
                "timestamp": datetime.now().isoformat(),
                "current_price": current_price,
                "timeframes": {}
            }

            # 각 타임프레임 데이터를 dict로 변환하여 저장
            for tf_name, df in multi_tf_data.items():
                market_analysis["timeframes"][tf_name] = df.to_dict(orient="records")

            # AI에게 분석 요청
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": """
                    You are a crypto trading expert specializing in multi-timeframe analysis.
                    Analyze the market data across different timeframes (15m, 1h, 4h) and provide your trading decision.
                    
                    Consider the following:
                    - Short-term trend (15m): Recent price action and momentum
                    - Medium-term trend (1h): Intermediate market direction
                    - Long-term trend (4h): Overall market bias
                    
                    Respond with only 'long' or 'short'.
                    """},
                    {"role": "user", "content": str(market_analysis)}
                ]
            )
            action = response.choices[0].message.content.lower().strip()
            print(f"AI Decision (Multi-Timeframe Analysis): {action.upper()}")

            # 주문 수량 계산 (최소 100 USDT 이상 주문)
            amount = math.ceil((100 / current_price) * 1000) / 1000
            print(f"Order Amount: {amount} BTC")

            # 레버리지 설정
            exchange.set_leverage(5, symbol)

            # 포지션 진입 및 SL/TP 주문 (버퍼 0.5% 적용)
            if action == "long":
                order = exchange.create_market_buy_order(symbol, amount)
                entry_price = current_price
                sl_price = round(entry_price * 0.995, 2)   # 0.5% 하락
                tp_price = round(entry_price * 1.005, 2)   # 0.5% 상승

                # SL/TP 주문 생성
                exchange.create_order(symbol, 'STOP_MARKET', 'sell', amount, None, {'stopPrice': sl_price})
                exchange.create_order(symbol, 'TAKE_PROFIT_MARKET', 'sell', amount, None, {'stopPrice': tp_price})

                print(f"\n=== LONG Position Opened ===")
                print(f"Entry: ${entry_price:,.2f}")
                print(f"Stop Loss: ${sl_price:,.2f} (-0.5%)")
                print(f"Take Profit: ${tp_price:,.2f} (+0.5%)")
                print("===========================")

            elif action == "short":
                order = exchange.create_market_sell_order(symbol, amount)
                entry_price = current_price
                sl_price = round(entry_price * 1.005, 2)   # 0.5% 상승
                tp_price = round(entry_price * 0.995, 2)   # 0.5% 하락

                # SL/TP 주문 생성
                exchange.create_order(symbol, 'STOP_MARKET', 'buy', amount, None, {'stopPrice': sl_price})
                exchange.create_order(symbol, 'TAKE_PROFIT_MARKET', 'buy', amount, None, {'stopPrice': tp_price})

                print(f"\n=== SHORT Position Opened ===")
                print(f"Entry: ${entry_price:,.2f}")
                print(f"Stop Loss: ${sl_price:,.2f} (+0.5%)")
                print(f"Take Profit: ${tp_price:,.2f} (-0.5%)")
                print("============================")
            else:
                print("Action이 'long' 또는 'short'가 아니므로 주문을 실행하지 않습니다.")

        time.sleep(1000)

    except Exception as e:
        print(f"\n Error: {e}")
        time.sleep(5000)