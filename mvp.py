import ccxt
import pandas as pd

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
print(df)