"""
Access control for the demo: roles, signed expiring session cookie, scrypt password check,
login throttling and an audit trail.

Demo scope: accounts are `doctor` and `patient<id>`, all sharing ONE demo password whose scrypt hash is
computed at startup. A real deployment would keep per-user salted hashes in a database, use HTTPS
(set COOKIE_SECURE=1) and an identity provider. This is access control for a proof of concept,
not a claim of DPDP or HIPAA compliance. All data is synthetic.
"""
import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import time
from dataclasses import dataclass

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel

SESSION_COOKIE = "gt_session"
SESSION_SECONDS = 8 * 3600
SECRET = os.environ.get("GLUCOTWIN_SECRET", "").encode() or secrets.token_bytes(32)  # random => sessions end on restart
DEMO_PASSWORD = os.environ.get("GLUCOTWIN_DEMO_PASSWORD", "Twin#2026")
COOKIE_SECURE = os.environ.get("COOKIE_SECURE", "0") == "1"
AUDIT_FILE = os.environ.get("AUDIT_LOG", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "audit.log"))

_SALT = secrets.token_bytes(16)
_scrypt = lambda pw: hashlib.scrypt(pw.encode(), salt=_SALT, n=2 ** 14, r=8, p=1, dklen=32)
_DEMO_HASH = _scrypt(DEMO_PASSWORD)

patient_exists = lambda pid: True   # replaced by api.main once the cohort is loaded


@dataclass
class User:
    username: str
    role: str                 # "doctor" | "patient"
    patient_id: int | None = None

    @property
    def display(self):
        return "Doctor" if self.role == "doctor" else f"Patient {self.patient_id}"


def lookup(username: str) -> User | None:
    username = username.strip().lower()
    if username == "doctor":
        return User("doctor", "doctor")
    m = re.fullmatch(r"patient(\d{1,4})", username)
    if m and patient_exists(int(m.group(1))):
        return User(username, "patient", int(m.group(1)))
    return None


# ---------------------------------------------------------------- signed session token
def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode().rstrip("=")


def make_token(u: User) -> str:
    body = _b64(json.dumps({"u": u.username, "exp": int(time.time()) + SESSION_SECONDS}).encode())
    return body + "." + _b64(hmac.new(SECRET, body.encode(), hashlib.sha256).digest())


def read_token(token: str | None) -> User | None:
    try:
        body, sig = token.split(".")
        good = _b64(hmac.new(SECRET, body.encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(sig, good):
            return None
        data = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
        if data["exp"] < time.time():
            return None
        return lookup(data["u"])
    except Exception:
        return None


def session_user(request: Request) -> User | None:
    return read_token(request.cookies.get(SESSION_COOKIE))


def current_user(request: Request) -> User:
    u = session_user(request)
    if not u:
        raise HTTPException(401, "not signed in")
    return u


# ---------------------------------------------------------------- throttling
_fails: dict[tuple, list] = {}
MAX_FAILS, WINDOW, LOCK = 5, 300, 60


def _throttled(key) -> int:
    now = time.time()
    f = [t for t in _fails.get(key, []) if now - t < WINDOW]
    _fails[key] = f
    if len(f) >= MAX_FAILS and now - f[-1] < LOCK:
        return int(LOCK - (now - f[-1])) + 1
    return 0


# ---------------------------------------------------------------- audit trail
_audit: list[dict] = []


def audit(user: User, action: str, patient_id: int | None = None):
    now = time.time()
    for e in reversed(_audit[-20:]):                       # collapse bursts (e.g. slider drags)
        if e["user"] == user.username and e["action"] == action and e["patient"] == patient_id and now - e["_t"] < 60:
            return
    entry = dict(time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)), user=user.username, role=user.role,
                 action=action, patient=patient_id, _t=now)
    _audit.append(entry)
    try:
        with open(AUDIT_FILE, "a") as f:
            f.write(json.dumps({k: v for k, v in entry.items() if k != "_t"}) + "\n")
    except OSError:
        pass


def read_audit(n=30):
    return [{k: v for k, v in e.items() if k != "_t"} for e in _audit[-n:]][::-1]


# ---------------------------------------------------------------- routes
router = APIRouter(prefix="/api/auth", tags=["auth"])


class Login(BaseModel):
    username: str
    password: str


@router.post("/login")
def login(body: Login, request: Request, response: Response):
    key = (request.client.host if request.client else "?", body.username.strip().lower())
    wait = _throttled(key)
    if wait:
        raise HTTPException(429, f"Too many attempts. Try again in {wait} s.")
    user = lookup(body.username)
    ok = hmac.compare_digest(_scrypt(body.password), _DEMO_HASH) and user is not None   # scrypt always runs: no timing hint
    if not ok:
        _fails.setdefault(key, []).append(time.time())
        raise HTTPException(401, "Wrong username or password.")
    _fails.pop(key, None)
    response.set_cookie(SESSION_COOKIE, make_token(user), max_age=SESSION_SECONDS, httponly=True, samesite="lax", secure=COOKIE_SECURE)
    audit(user, "sign_in")
    return {"username": user.username, "role": user.role, "patient_id": user.patient_id}


@router.post("/logout")
def logout(response: Response, user: User = Depends(current_user)):
    audit(user, "sign_out")
    response.delete_cookie(SESSION_COOKIE)
    return {"ok": True}


@router.get("/me")
def me(user: User = Depends(current_user)):
    return {"username": user.username, "role": user.role, "patient_id": user.patient_id, "display": user.display}
