import os
import getpass
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

PRODUCT_ID = "db017667-e895-4586-a3d1-c019a5ef43a7"
OUTPUT = "data/raw/sentinel1/S1D_Trishuli_20260909_COG.SAFE"

TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/"
    "CDSE/protocol/openid-connect/token"
)
DOWNLOAD_URL = (
    "https://download.dataspace.copernicus.eu/odata/v1/Products("
    f"{PRODUCT_ID})/$value"
)

def make_session():
    session = requests.Session()
    retries = Retry(
        total=4,
        connect=4,
        read=2,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session

def main():
    session = make_session()

    username = input("Copernicus account email: ").strip()
    password = getpass.getpass("Copernicus password: ")

    print("\nAuthenticating with Copernicus Data Space...")
    response = session.post(
        TOKEN_URL,
        data={
            "client_id": "cdse-public",
            "username": username,
            "password": password,
            "grant_type": "password",
        },
        timeout=60,
    )

    if not response.ok:
        print("Authentication failed:", response.status_code)
        print(response.text[:500])
        return

    token = response.json().get("access_token")
    if not token:
        print("Authentication response did not contain an access token.")
        return

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

    # Resume a partial download if one exists.
    existing_size = os.path.getsize(OUTPUT) if os.path.exists(OUTPUT) else 0
    headers = {"Authorization": f"Bearer {token}"}
    if existing_size:
        headers["Range"] = f"bytes={existing_size}-"

    print("\nStarting download...")
    try:
        with session.get(
            DOWNLOAD_URL,
            headers=headers,
            stream=True,
            timeout=(30, 180),
        ) as r:
            r.raise_for_status()

            # If the server ignores Range, restart rather than append.
            if existing_size and r.status_code != 206:
                existing_size = 0

            mode = "ab" if existing_size and r.status_code == 206 else "wb"
            content_length = int(r.headers.get("Content-Length", 0))
            downloaded = existing_size

            with open(OUTPUT, mode) as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        print(
                            f"\rDownloaded: {downloaded / 1024**2:.1f} MB",
                            end="",
                            flush=True,
                        )

        print("\n\nDownload finished.")
        print("Saved path:", OUTPUT)
        print(f"File size: {os.path.getsize(OUTPUT) / 1024**3:.2f} GB")

    except requests.RequestException as exc:
        print("\nDownload failed:", exc)
        print("If a partial file exists, rerun the script to retry.")

if __name__ == "__main__":
    main()
