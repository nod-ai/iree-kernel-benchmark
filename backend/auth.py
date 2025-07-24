import time
import json
import os
import jwt
from pathlib import Path
from typing import Literal
import requests
from dotenv import load_dotenv

load_dotenv()

PEM_FILE = Path(os.getenv("PEM_FILE"))
CLIENT_ID = os.getenv("CLIENT_ID")
JWT_CACHE_FILE = os.getenv("JWT_CACHE_FILE")
ACCESS_TOKEN_CACHE_FILE = os.getenv("ACCESS_TOKEN_CACHE_FILE")


def load_cache(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}

def save_cache(file_path, data):
    with open(file_path, 'w') as f:
        json.dump(data, f)

def get_jwt():
    current_time = int(time.time())
    cache = load_cache(JWT_CACHE_FILE)

    if cache.get("token") and cache.get("expires_at", 0) > current_time:
        return cache["token"]

    with open(PEM_FILE, 'rb') as pem_file:
        signing_key = pem_file.read()

    payload = {
        'iat': current_time,
        'exp': current_time + 400,
        'iss': CLIENT_ID
    }

    encoded_jwt = jwt.encode(payload, signing_key, algorithm='RS256')
    cache = {
        "token": encoded_jwt,
        "expires_at": current_time + 400
    }
    save_cache(JWT_CACHE_FILE, cache)

    return encoded_jwt

def get_access_token(installation: Literal['WAVE', 'BENCH', 'TEST'] = 'WAVE'):
    current_time = int(time.time())
    cache_file = f'{installation}_{ACCESS_TOKEN_CACHE_FILE}'

    cache = load_cache(cache_file)

    if cache.get("token") and cache.get("expires_at", 0) > current_time:
        return cache["token"]

    jwt_token = get_jwt()
    installation_id = int(os.getenv(f'{installation}_INSTALLATION_ID'))
    url = f"https://api.github.com/app/installations/{installation_id}/access_tokens"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {jwt_token}",
        "X-GitHub-Api-Version": "2022-11-28"
    }

    response = requests.post(url, headers=headers)
    response_data = response.json()

    if response.status_code == 201:
        expires_at_struct = time.strptime(response_data["expires_at"], "%Y-%m-%dT%H:%M:%SZ")
        expires_at_epoch = int(time.mktime(expires_at_struct))
        cache = {
            "token": response_data["token"],
            "expires_at": expires_at_epoch
        }
        save_cache(cache_file, cache)
        return response_data["token"]
    else:
        raise Exception(f"Failed to get access token: {response_data}")

if __name__ == '__main__':
    token = get_access_token()
    print(token)
