from dotenv import load_dotenv
import os

load_dotenv("buffett-config/.env")

openai_api_key = os.getenv("OPANAI_API_KEY")
api_key = os.getenv("BINANCE_API_KEY")
secret_key = os.getenv("BINANCE_SECRET_KEY")

print(openai_api_key)
print(api_key)
print(secret_key)
