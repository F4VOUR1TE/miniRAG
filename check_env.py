import os
from dotenv import load_dotenv

load_dotenv()  

key = os.getenv("OPENROUTER_API_KEY")
if key:
    print(f"Ключ найден, начинается с: {key[:12]}...")
else:
    print("Ключ НЕ найден. Проверь .env")