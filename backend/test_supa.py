import requests
import os
from dotenv import load_dotenv

load_dotenv()
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_URL = "https://hjzqywjtssveipriurgn.supabase.co/rest/v1/User_database"

headers = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}"
}

# Just get one row to see columns
resp = requests.get(SUPABASE_URL + "?limit=1", headers=headers)
print("Status:", resp.status_code)
if resp.ok and len(resp.json()) > 0:
    print("Columns:", list(resp.json()[0].keys()))
else:
    print("Response:", resp.text)
