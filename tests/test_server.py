"""Integration tests for FastAPI server (v0.4.0)."""
import pytest
from fastapi.testclient import TestClient
from server.app import app

@pytest.fixture
def client():
    return TestClient(app)

class TestHealth:
    def test_health(self, client):
        assert client.get("/health").json()["status"] == "healthy"

    def test_metadata(self, client):
        assert client.get("/metadata").json()["name"] == "email-triage-env"

    def test_schema(self, client):
        data = client.get("/schema").json()
        assert "action" in data and "observation" in data and "state" in data
        assert "legal_compliance" in str(data["action"])

    def test_root(self, client):
        assert client.get("/").status_code == 200

class TestReset:
    def test_all_tasks(self, client):
        for t in ["easy", "medium", "hard", "enterprise"]:
            r = client.post("/reset", json={"task_id": t})
            assert r.status_code == 200
            assert r.json()["done"] is False

    def test_seed_reproducible(self, client):
        r1 = client.post("/reset", json={"task_id": "hard", "seed": 42})
        r2 = client.post("/reset", json={"task_id": "hard", "seed": 42})
        assert r1.json()["observation"]["current_email"]["id"] == r2.json()["observation"]["current_email"]["id"]

    def test_invalid_task(self, client):
        assert client.post("/reset", json={"task_id": "x"}).status_code == 422

class TestStep:
    def test_valid(self, client):
        client.post("/reset", json={"task_id": "easy"})
        r = client.post("/step", json={"action": {"action_type": "archive"}})
        assert r.status_code == 200 and "reward" in r.json()

    def test_defer(self, client):
        client.post("/reset", json={"task_id": "easy"})
        r = client.post("/step", json={"action": {"action_type": "defer", "confidence": 0.2}})
        assert r.json()["reward"] < 0

    def test_legal_compliance(self, client):
        client.post("/reset", json={"task_id": "easy"})
        r = client.post("/step", json={"action": {"action_type": "route", "department": "legal_compliance"}})
        assert r.status_code == 200

    def test_invalid_action(self, client):
        client.post("/reset", json={"task_id": "easy"})
        assert client.post("/step", json={"action": {"action_type": "fly"}}).status_code == 422

class TestState:
    def test_new_fields(self, client):
        client.post("/reset", json={"task_id": "hard"})
        s = client.get("/state").json()
        assert "customer_satisfaction" in s
        assert "calibration_score" in s
        assert "escalation_budget_remaining" in s
        assert "consequence_stage_reached" in s
        assert 0 < s["current_score"] < 1

    def test_csat_decays(self, client):
        client.post("/reset", json={"task_id": "hard", "seed": 0})
        for _ in range(5):
            r = client.post("/step", json={"action": {"action_type": "archive"}})
            if r.json()["done"]:
                break
        s = client.get("/state").json()
        assert s["customer_satisfaction"] < 1.0

    def test_score_bounded(self, client):
        client.post("/reset", json={"task_id": "hard", "seed": 0})
        for _ in range(40):
            r = client.post("/step", json={"action": {"action_type": "archive"}})
            if r.json()["done"]:
                break
        assert 0 < client.get("/state").json()["current_score"] < 1
