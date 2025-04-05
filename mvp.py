import ccxt
import os
from dotenv import load_dotenv
import pandas as pd
from openai import OpenAI
import math

### 수정 가능한 변수 ###

# _openai_model_name = "gpt-4o"
_openai_model_name = "gpt-3.5-turbo"

_symbol = "BTC/USDT"        # 거래할 심볼
_min_order_usdt = 7         # 최소 주문 금액 (예: 7 USDT)
_reverage = 1               # 레버리지 배율

# 포지션별 손절/익절 비율 설정 (단위: 0.01, 즉 0.5 -> 0.5%)
_long_stop_loss_rate = 1.0
_long_take_profit_rate = 1.0
_short_stop_loss_rate = 1.0
_short_take_profit_rate = 1.0

### 수정 가능한 변수 종료 ###

load_dotenv("buffett-config/.env")

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

# 2. 차트 데이터 가져오기 (15분봉 최근 24시간)
ohlcv = exchange.fetch_ohlcv(_symbol, timeframe="15m", limit=96)
df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

# 3. OpenAI를 통한 AI 투자 판단 받기
client = OpenAI()
response = client.chat.completions.create(
    model=_openai_model_name,
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

# 4. 최소 주문금액(_min_order_usdt USDT 이상) 만족하는 주문 수량 계산
current_price = exchange.fetch_ticker(_symbol)['last']
# 100 USDT 이상 주문하도록 수량 산출 (소수점 3자리 반올림)
amount = math.ceil((_min_order_usdt / current_price) * 1000) / 1000
print("Order Amount:", amount)

# 5. 레버리지 _REVERAGE배 설정
exchange.set_leverage(_reverage, _symbol)

if action == "long":
    # 롱 포지션 진입 (시장가 매수)
    order = exchange.create_market_buy_order(_symbol, amount)
    print("Long order executed:", order)
    entry_price = current_price  # 진입가: 현재 가격 사용

    # 롱 포지션의 경우: 손절은 진입가의 _long_stop_loss_rate% 하락, 익절은 _long_take_profit_rate% 상승
    stop_loss_price = round(entry_price * (1.0 - (_long_stop_loss_rate * 0.01)), 2)
    take_profit_price = round(entry_price * (1.0 + (_long_take_profit_rate * 0.01)), 2)

    # Stop Loss 주문 (롱 포지션 청산을 위한 STOP_MARKET 매도 주문)
    sl_order = exchange.create_order(
        symbol=_symbol,
        type='STOP_MARKET',
        side='sell',
        amount=amount,
        price=None,
        params={'stopPrice': stop_loss_price}
    )
    # Take Profit 주문 (롱 포지션 청산을 위한 TAKE_PROFIT_MARKET 매도 주문)
    tp_order = exchange.create_order(
        symbol=_symbol,
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
    order = exchange.create_market_sell_order(_symbol, amount)
    print("Short order executed:", order)
    entry_price = current_price

    # 숏 포지션의 경우: 손절은 진입가의 _short_stop_loss_rate% 상승, 익절은 _short_take_profit_rate% 하락
    stop_loss_price = round(entry_price * (1.0 + (_short_stop_loss_rate * 0.01)), 2)
    take_profit_price = round(entry_price * (1.0 - (_short_take_profit_rate * 0.01)), 2)

    # Stop Loss 주문 (숏 포지션 청산을 위한 STOP_MARKET 매수 주문)
    sl_order = exchange.create_order(
        symbol=_symbol,
        type='STOP_MARKET',
        side='buy',
        amount=amount,
        price=None,
        params={'stopPrice': stop_loss_price}
    )
    # Take Profit 주문 (숏 포지션 청산을 위한 TAKE_PROFIT_MARKET 매수 주문)
    tp_order = exchange.create_order(
        symbol=_symbol,
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
