"""Smoke + security tests. Run after `sh scripts/run_all.sh`:  pytest -q"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["AUDIT_LOG"] = "/tmp/glucotwin_test_audit.log"
from fastapi.testclient import TestClient
from api.main import app
from api import auth

PW = auth.DEMO_PASSWORD


def client(user=None):
    c = TestClient(app)
    if user:
        assert c.post("/api/auth/login", json={"username": user, "password": PW}).status_code == 200
    return c


def test_health_is_public():
    assert client().get("/api/health").json()["patients"] == 300


def test_unauthenticated_is_rejected():
    c = client()
    for path in ["/api/patients", "/api/patients/3", "/api/model-card", "/api/meals", "/api/audit", "/api/auth/me"]:
        assert c.get(path).status_code == 401, path
    r = c.get("/", follow_redirects=False)
    assert r.status_code == 302 and r.headers["location"] == "/login"


def test_wrong_password_and_unknown_user():
    c = client()
    assert c.post("/api/auth/login", json={"username": "doctor", "password": "nope"}).status_code == 401
    assert c.post("/api/auth/login", json={"username": "patient9999", "password": PW}).status_code == 401


def test_patient_sees_only_own_record():
    c = client("patient3")
    assert c.get("/api/patients/3").status_code == 200
    assert c.get("/api/patients/4").status_code == 403
    assert c.post("/api/patients/4/whatif", json={"scenarios": [{"label": "x", "meal": "poha"}]}).status_code == 403
    assert c.get("/api/patients").status_code == 403          # cohort list: doctors only
    assert c.get("/api/model-card").status_code == 403
    assert c.get("/api/audit").status_code == 403


def test_doctor_sees_everything_and_audit_records_it():
    c = client("doctor")
    assert len(c.get("/api/patients").json()) > 0
    assert c.get("/api/patients/7").status_code == 200
    assert any(e["action"] == "view_patient" and e["patient"] == 7 for e in c.get("/api/audit").json())


def test_tampered_or_expired_token_is_rejected():
    c = client("doctor")
    good = c.cookies.get(auth.SESSION_COOKIE)
    body, sig = good.split(".")
    forged = TestClient(app); forged.cookies.set(auth.SESSION_COOKIE, body + "." + sig[:-2] + "AA")
    assert forged.get("/api/patients").status_code == 401
    # forge a doctor token for a patient session by swapping the body: signature no longer matches
    import base64, json
    fake = base64.urlsafe_b64encode(json.dumps({"u": "doctor", "exp": 9999999999}).encode()).decode().rstrip("=")
    f2 = TestClient(app); f2.cookies.set(auth.SESSION_COOKIE, fake + "." + sig)
    assert f2.get("/api/patients").status_code == 401


def test_login_throttle_after_repeated_failures():
    c = client()
    codes = [c.post("/api/auth/login", json={"username": "patient8", "password": "bad"}).status_code for _ in range(7)]
    assert codes[:5] == [401] * 5 and codes[5] == 429


def test_twin_learns_from_stream():
    c = client("doctor"); c.post("/api/patients/11/reset")
    n0 = c.get("/api/patients/11").json()["twin"]["meals_observed"]
    c.post("/api/patients/11/advance", json={"hours": 12})
    assert c.get("/api/patients/11").json()["twin"]["meals_observed"] > n0


def test_walk_and_smaller_portion_reduce_peak():
    c = client("doctor")
    base = dict(meal="white_rice_dal", portion=1.0, walk=False, sleep_hours=7)
    r = c.post("/api/patients/3/whatif", json={"scenarios": [
        {**base, "label": "a"}, {**base, "walk": True, "label": "b"}, {**base, "portion": 0.5, "label": "c"}]}).json()["scenarios"]
    assert r[1]["peak"] < r[0]["peak"] and r[2]["peak"] < r[0]["peak"]


def test_low_gi_swap_reduces_spike_probability():
    c = client("doctor")
    r = c.post("/api/patients/3/whatif", json={"scenarios": [
        {"label": "rice", "meal": "white_rice_curry", "sleep_hours": 7},
        {"label": "millet", "meal": "millet_roti_sabzi", "sleep_hours": 7}]}).json()["scenarios"]
    assert r[1]["p_spike"] < r[0]["p_spike"]


# ---------------------------------------------------------------- typed foods
def test_food_parser_quantities_typos_and_multiple_items():
    from twin import foods
    r = foods.parse_meal("2 idli, sambar, coconut chutney")
    assert r["carbs_g"] == 30.0 and [i["name"] for i in r["items"]] == ["idli", "sambar", "coconut chutney"]
    assert foods.parse_meal("idly and sambar")["items"][0]["name"] == "idli"
    assert foods.parse_meal("gulab jamoon")["items"][0]["name"] == "gulab jamun"
    assert foods.parse_meal("1.5 cup white rice, fish curry")["items"][0]["carbs_g"] == 67.5
    assert foods.parse_meal("pizza 40 g")["carbs_g"] == 40.0


def test_unknown_food_is_reported_not_guessed():
    from twin import foods
    r = foods.parse_meal("xyzfood")
    assert r["ok"] is False and r["unmatched"] == ["xyzfood"] and r["carbs_g"] == 0


def test_swap_lowers_gi_and_keeps_quantity():
    from twin import foods
    s = foods.suggest_swap("2 plain dosa, sambar and white rice")
    assert s["text"] == "2 ragi dosa, sambar and millet rice" and len(s["changes"]) == 2


def test_food_endpoints_need_login_and_work():
    assert client().get("/api/foods/search?q=id").status_code == 401
    c = client("patient3")
    assert c.get("/api/foods/search?q=masa").json()[0]["name"] == "masala dosa"
    assert c.post("/api/foods/parse", json={"text": "2 roti, dal"}).json()["carbs_g"] == 56.0


def test_whatif_accepts_typed_foods_and_rejects_unknown():
    c = client("doctor")
    r = c.post("/api/patients/3/whatif", json={"scenarios": [
        {"label": "A", "foods": "white rice, dal, curd"},
        {"label": "B", "foods": "millet rice, dal, curd", "walk": True}]})
    a, b = r.json()["scenarios"]
    assert a["parsed"]["carbs_g"] == 73.0 and b["peak"] < a["peak"]
    bad = c.post("/api/patients/3/whatif", json={"scenarios": [{"label": "A", "foods": "xyzfood"}]})
    assert bad.status_code == 422 and "xyzfood" in bad.json()["detail"]


def test_doctor_triage_list():
    rows = client("doctor").get("/api/patients").json()
    assert len(rows) == 300 and {"tir_24h", "spikes_24h", "attention", "score"} <= set(rows[0])
    flagged = sum(r["attention"] for r in rows)
    assert 10 <= flagged <= 70                      # a worklist, not everyone


def test_doctor_can_ingest_wearable_reading_and_update_twin():
    c = client("doctor")
    c.post("/api/patients/12/reset")
    before = c.get("/api/patients/12").json()["twin"]["meals_observed"]
    r = c.post("/api/ingest", json={"patient_id": 12, "glucose": 214, "heart_rate": 88,
                                    "steps": 120, "sleep_hours": 6.5, "meal": "2 idli"})
    assert r.status_code == 200
    body = r.json()
    assert body["reading"]["glucose"] == 214.0
    assert body["update"]["meals_observed"] == before + 1
    assert any(e["action"] == "ingest_reading" and e["patient"] == 12
               for e in c.get("/api/audit").json())


def test_ingest_rejects_bad_input_and_unknown_food():
    c = client("doctor")
    assert c.post("/api/ingest", json={"patient_id": 12, "glucose": 5, "heart_rate": 88, "steps": 0}).status_code == 422
    assert c.post("/api/ingest", json={"patient_id": 12, "glucose": 120, "heart_rate": 88,
                                       "steps": 0, "meal": "not a food"}).status_code == 422


def test_patient_cannot_ingest_for_any_patient():
    c = client("patient3")
    assert c.post("/api/ingest", json={"patient_id": 3, "glucose": 120, "heart_rate": 80, "steps": 0}).status_code == 403
