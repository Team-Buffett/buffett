import ccxt
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv("buffett-config/.env")

# binance 로 초기화
exchange = ccxt.binance({
    'options': {
        'defaultType': 'future' # 선물
    }
})

# 차트 데이터 가져오기
symbol = 'BTC/USDT'
ohlcv = exchange.fetch_ohlcv(symbol, '15m', limit=96)
df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')


client = OpenAI()

response = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {
      "role": "system",
      "content": [
        {
          "type": "text",
          "text": "you are a cryptocurrency trading expert. Analyze the market data and respond with 'Long' or 'Short'. Do not use any other characters except long or short"
        }
      ]
    },
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": df.to_json()
        }
      ]
    }
  ],
  response_format={
    "type": "text"
  }
)

final_result = response.choices[0].message.content

print(final_result)
