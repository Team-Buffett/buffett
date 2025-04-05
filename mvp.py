import ccxt
import os
from dotenv import load_dotenv
load_dotenv()
import pandas as pd
from openai import OpenAI
import math

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

# 2. 차트 데이터 가져오기 (15분봉 최근 24시간)
ohlcv = exchange.fetch_ohlcv("BTC/USDT", timeframe="15m", limit=96)
df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

# 3. OpenAI를 통한 AI 투자 판단 받기
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "system",
            "content": "You are a crypto trading expert. Analyze the market data and respond with only 'long' or 'short'."
        },
        {
            "role": "user",
            "content": df.to_json()
        }
    ]
)
# AI의 응답(문자열)을 소문자로 변환하여 action 변수에 저장
action = response.choices[0].message.content.lower().strip()
print("AI Action:", action)

# 4. 최소 주문금액(100 USDT 이상) 만족하는 주문 수량 계산
current_price = exchange.fetch_ticker(symbol)['last']
# 100 USDT 이상 주문하도록 수량 산출 (소수점 3자리 반올림)
amount = math.ceil((100 / current_price) * 1000) / 1000
print("Order Amount:", amount)

# 5. 레버리지 5배 설정
exchange.set_leverage(5, symbol)

# 6. 진입 주문 후 Stop Loss / Take Profit 주문 지정 - 진입 가격 기준으로 ±0.5% 설정
if action == "long":
    # 롱 포지션 진입 (시장가 매수)
    order = exchange.create_market_buy_order(symbol, amount)
    print("Long order executed:", order)
    entry_price = current_price  # 진입가: 현재 가격 사용
    # 롱 포지션의 경우, 손절은 진입가의 0.5% 하락, 익절은 0.5% 상승
    stop_loss_price = round(entry_price * 0.995, 2)      # 0.5% 하락
    take_profit_price = round(entry_price * 1.005, 2)      # 0.5% 상승

    # Stop Loss 주문 (롱 포지션 청산을 위한 STOP_MARKET 매도 주문)
    sl_order = exchange.create_order(
        symbol=symbol,
        type='STOP_MARKET',
        side='sell',
        amount=amount,
        price=None,
        params={'stopPrice': stop_loss_price}
    )
    # Take Profit 주문 (롱 포지션 청산을 위한 TAKE_PROFIT_MARKET 매도 주문)
    tp_order = exchange.create_order(
        symbol=symbol,
        type='TAKE_PROFIT_MARKET',
        side='sell',
        amount=amount,
        price=None,
        params={'stopPrice': take_profit_price}
    )
    print("Stop Loss order:", sl_order)
    print("Take Profit order:", tp_order)

elif action == "short":
    # 숏 포지션 진입 (시장가 매도)
    order = exchange.create_market_sell_order(symbol, amount)
    print("Short order executed:", order)
    entry_price = current_price
    # 숏 포지션의 경우, 손절은 진입가의 0.5% 상승, 익절은 0.5% 하락
    stop_loss_price = round(entry_price * 1.005, 2)      # 0.5% 상승
    take_profit_price = round(entry_price * 0.995, 2)      # 0.5% 하락

    # Stop Loss 주문 (숏 포지션 청산을 위한 STOP_MARKET 매수 주문)
    sl_order = exchange.create_order(
        symbol=symbol,
        type='STOP_MARKET',
        side='buy',
        amount=amount,
        price=None,
        params={'stopPrice': stop_loss_price}
    )
    # Take Profit 주문 (숏 포지션 청산을 위한 TAKE_PROFIT_MARKET 매수 주문)
    tp_order = exchange.create_order(
        symbol=symbol,
        type='TAKE_PROFIT_MARKET',
        side='buy',
        amount=amount,
        price=None,
        params={'stopPrice': take_profit_price}
    )
    print("Stop Loss order:", sl_order)
    print("Take Profit order:", tp_order)

else:
    print("Action is neither 'long' nor 'short'. No orders executed.")
