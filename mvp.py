import ccxt
import os
from dotenv import load_dotenv
import pandas as pd
from openai import OpenAI
import math

load_dotenv()
_openai_model_name = "gpt-3.5-turbo"

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

# 차트 데이터 가져오기
ohlcv = exchange.fetch_ohlcv("BTC/USDT", timeframe="15m", limit=96)
df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

# AI 투자 판단 받기
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
action = response.choices[0].message.content.lower()
print(action)

# 100 USDT 가치의 비트코인 수량 계산
current_price = exchange.fetch_ticker("BTC/USDT")['last']
amount = math.ceil((100 / current_price) * 1000) / 1000
print(amount)

# Long / Short 레버리지 5배 실행
exchange.set_leverage(5, "BTC/USDT")

# AI 판단에 따른 포지션 진입
if action == "long":
    order = exchange.create_market_buy_order("BTC/USDT", amount)
    print(order)
elif action == "short":
    order = exchange.create_market_sell_order("BTC/USDT", amount)
    print(order)