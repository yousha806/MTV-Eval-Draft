import requests


def check_url(url: str, timeout=5) -> bool:
    try:
        r = requests.head(url, timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False
