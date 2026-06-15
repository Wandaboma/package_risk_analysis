import os
import requests
from datetime import date, timedelta

BASE_URL = "https://static.crates.io/archive/version-downloads"
SAVE_DIR = "../data/version_downloads"

os.makedirs(SAVE_DIR, exist_ok=True)

# start / end date
start_date = date(2023, 11, 1)
end_date = date(2025, 11, 30)

current = start_date

while current <= end_date:
    date_str = current.isoformat()

    url = f"{BASE_URL}/{date_str}.csv"
    save_path = os.path.join(SAVE_DIR, f"{date_str}.csv")

    # skip if already downloaded
    if os.path.exists(save_path):
        print(f"[SKIP] {date_str}")
        current += timedelta(days=1)
        continue

    try:
        print(f"[DOWNLOAD] {url}")

        response = requests.get(url, timeout=60)

        if response.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(response.content)

            print(f"[OK] saved -> {save_path}")

        else:
            print(f"[MISS] {date_str} status={response.status_code}")

    except Exception as e:
        print(f"[ERROR] {date_str}: {e}")

    current += timedelta(days=1)