"""Post a short synthetic wearable stream to the doctor-only ingestion endpoint."""
import argparse
import json
import time
import urllib.error
import urllib.request


def post(url, cookie, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json", "Cookie": cookie})
    with urllib.request.urlopen(req) as response:
        return json.load(response)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--patient", type=int, default=3)
    ap.add_argument("--password", default="Twin#2026")
    ap.add_argument("--interval", type=float, default=0.0)
    a = ap.parse_args()
    # urllib does not expose the cookie jar from a request, so obtain it explicitly.
    request = urllib.request.Request(a.url + "/api/auth/login",
                                     data=json.dumps({"username": "doctor", "password": a.password}).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request) as response:
        cookie = response.headers.get("Set-Cookie", "").split(";", 1)[0]
    readings = [
        {"glucose": 132, "heart_rate": 78, "steps": 20, "sleep_hours": 7.0},
        {"glucose": 148, "heart_rate": 84, "steps": 110, "sleep_hours": 7.0, "meal": "2 idli, sambar"},
        {"glucose": 176, "heart_rate": 86, "steps": 150, "sleep_hours": 7.0},
    ]
    for item in readings:
        item["patient_id"] = a.patient
        try:
            print(json.dumps(post(a.url + "/api/ingest", cookie, item), indent=2))
        except urllib.error.HTTPError as exc:
            raise SystemExit(f"ingestion failed: HTTP {exc.code} {exc.read().decode()}") from exc
        if a.interval:
            time.sleep(a.interval)


if __name__ == "__main__":
    main()
