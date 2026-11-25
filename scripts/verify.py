import os
import sys
import json
import time
from unittest.mock import patch
from fastapi.testclient import TestClient
from litellm.types.utils import ModelResponse, Choices, Message, Usage

# Setup paths
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
os.environ["LLM_API_KEYS"] = "eval_key_1,eval_key_2"
os.environ["ENABLE_FAILURE_SIMULATION"] = "true"
os.environ["DATABASE_URL"] = "sqlite:///./data/verify_test.db"

import app.database as db_module
from app.database import init_db
from app.main import app
import app.llm_service as llm_mod

# Initialize test DB
test_db = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "verify_test.db"))
os.makedirs(os.path.dirname(test_db), exist_ok=True)
db_module.DB_PATH = test_db
init_db()

client = TestClient(app)

def mock_llm_response(content="The capital of France is Paris.", in_tokens=42, out_tokens=9, model="gemini/gemini-3.8-flash"):
    return ModelResponse(
        id="cmpl-eval-mock",
        choices=[Choices(finish_reason="stop", index=0, message=Message(content=content, role="assistant"))],
        usage=Usage(prompt_tokens=in_tokens, completion_tokens=out_tokens, total_tokens=in_tokens + out_tokens),
        model=model
    )

results = {}

print("=== STARTING COMPLETE AUDIT & VERIFICATION ===")

# TEST 1 — Health
try:
    r = client.get("/health")
    data = r.json()
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    assert data["status"] == "ok", f"Expected ok, got {data['status']}"
    assert data["keys_configured"] >= 2, f"Expected >=2 keys, got {data['keys_configured']}"
    assert data["cache_enabled"] is True, f"Expected True, got {data['cache_enabled']}"
    results["TEST 1 - Health"] = "PASS"
    print("TEST 1 - Health: PASS", data)
except Exception as e:
    results["TEST 1 - Health"] = f"FAIL: {e}"
    print("TEST 1 - Health: FAIL", e)

# TEST 2 — Live Inference
try:
    with patch("litellm.completion") as mock_comp:
        mock_comp.return_value = mock_llm_response("The capital of France is Paris.", in_tokens=42, out_tokens=9)
        s1 = "s-verif-1"
        r = client.post("/chat", json={"session_id": s1, "message": "What is the capital of France?"})
        assert r.status_code == 200, f"Status {r.status_code}"
        d = r.json()
        assert d["reply"] == "The capital of France is Paris."
        m = d["metadata"]
        assert m["resolution_source"] == "LIVE_INFERENCE"
        assert isinstance(m["latency_ms"], int)
        assert m["input_tokens"] == 42
        assert m["output_tokens"] == 9
        assert m["key_id"] == "key_1"
        assert m["fallback_attempts"] == 0
        assert "model" in m
        results["TEST 2 - Live inference"] = "PASS"
        print("TEST 2 - Live inference: PASS", m)
except Exception as e:
    results["TEST 2 - Live inference"] = f"FAIL: {e}"
    print("TEST 2 - Live inference: FAIL", e)

# TEST 3 — Cache Hit
try:
    with patch("litellm.completion") as mock_comp:
        # Same first message in completely new Session B
        s2 = "s-verif-2"
        r = client.post("/chat", json={"session_id": s2, "message": "What is the capital of France?"})
        assert r.status_code == 200, f"Status {r.status_code}"
        d = r.json()
        assert mock_comp.call_count == 0, "Provider was called on cache hit!"
        assert d["reply"] == "The capital of France is Paris."
        m = d["metadata"]
        assert m["resolution_source"] == "LLM_CACHE"
        assert m["key_id"] is None
        assert m["fallback_attempts"] == 0
        assert m["input_tokens"] == 42
        assert m["output_tokens"] == 9
        assert isinstance(m["latency_ms"], int)
        results["TEST 3 - Cache hit"] = "PASS"
        print("TEST 3 - Cache hit: PASS", m)
except Exception as e:
    results["TEST 3 - Cache hit"] = f"FAIL: {e}"
    print("TEST 3 - Cache hit: FAIL", e)

# TEST 4 — Different Cache Key
try:
    with patch("litellm.completion") as mock_comp:
        mock_comp.return_value = mock_llm_response("Rome is the capital of Italy.", in_tokens=30, out_tokens=7)
        s3 = "s-verif-3"
        r = client.post("/chat", json={"session_id": s3, "message": "What is the capital of Italy?"})
        assert r.status_code == 200
        d = r.json()
        assert d["reply"] == "Rome is the capital of Italy."
        assert d["metadata"]["resolution_source"] == "LIVE_INFERENCE"
        assert mock_comp.call_count == 1
        results["TEST 4 - Different cache key"] = "PASS"
        print("TEST 4 - Different cache key: PASS")
except Exception as e:
    results["TEST 4 - Different cache key"] = f"FAIL: {e}"
    print("TEST 4 - Different cache key: FAIL", e)

# TEST 5 — Multi-Turn Conversation
try:
    with patch("litellm.completion") as mock_comp:
        def side_effect(*args, **kwargs):
            msgs = kwargs.get("messages", [])
            contents = [m["content"] for m in msgs]
            if any("My name is Alex" in c for c in contents) and any("What is my name?" in c for c in contents):
                return mock_llm_response("Your name is Alex.", in_tokens=55, out_tokens=6)
            return mock_llm_response("Nice to meet you, Alex!", in_tokens=25, out_tokens=7)
        mock_comp.side_effect = side_effect

        s_alex = "s-alex-turn"
        r1 = client.post("/chat", json={"session_id": s_alex, "message": "My name is Alex."})
        assert r1.status_code == 200

        r2 = client.post("/chat", json={"session_id": s_alex, "message": "What is my name?"})
        assert r2.status_code == 200
        assert r2.json()["reply"] == "Your name is Alex."
        results["TEST 5 - Multi-turn conversation"] = "PASS"
        print("TEST 5 - Multi-turn conversation: PASS")
except Exception as e:
    results["TEST 5 - Multi-turn conversation"] = f"FAIL: {e}"
    print("TEST 5 - Multi-turn conversation: FAIL", e)

# TEST 6 — Session Isolation
try:
    with patch("litellm.completion") as mock_comp:
        call_histories = []
        def se(*args, **kwargs):
            call_histories.append(kwargs.get("messages", []))
            return mock_llm_response("Color answer", in_tokens=20, out_tokens=4)
        mock_comp.side_effect = se

        client.post("/chat", json={"session_id": "session-color-A", "message": "My favorite color is blue."})
        client.post("/chat", json={"session_id": "session-color-B", "message": "What is my favorite color?"})

        hist_b = [m["content"] for m in call_histories[1]]
        assert not any("blue" in c for c in hist_b), "Session B leaked history from Session A!"
        results["TEST 6 - Session isolation"] = "PASS"
        print("TEST 6 - Session isolation: PASS")
except Exception as e:
    results["TEST 6 - Session isolation"] = f"FAIL: {e}"
    print("TEST 6 - Session isolation: FAIL", e)

# TEST 7 — History Endpoint
try:
    hist_r = client.get("/sessions/s-alex-turn/messages")
    assert hist_r.status_code == 200
    hdata = hist_r.json()
    assert hdata["session_id"] == "s-alex-turn"
    msgs = hdata["messages"]
    assert len(msgs) == 4 # user, assistant, user, assistant
    assert msgs[0]["role"] == "user"
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["metadata"]["resolution_source"] == "LIVE_INFERENCE"
    assert "input_tokens" in msgs[1]["metadata"]
    assert "output_tokens" in msgs[1]["metadata"]
    assert not any(m["role"] == "system" for m in msgs), "System prompt appeared in history!"

    # 404 for unknown session
    unknown = client.get("/sessions/s-nonexistent-unknown/messages")
    assert unknown.status_code == 404
    assert unknown.json() == {"detail": "session not found"}
    results["TEST 7 - History endpoint"] = "PASS"
    print("TEST 7 - History endpoint: PASS")
except Exception as e:
    results["TEST 7 - History endpoint"] = f"FAIL: {e}"
    print("TEST 7 - History endpoint: FAIL", e)

# TEST 8 — Failover 429
try:
    with patch("litellm.completion") as mock_comp:
        mock_comp.return_value = mock_llm_response("Key 2 recovered", in_tokens=30, out_tokens=5)
        r = client.post(
            "/chat",
            json={"session_id": "s-failover-1", "message": "Failover 429 test"},
            headers={"X-Simulate-Failure": "key_1=429"}
        )
        assert r.status_code == 200
        d = r.json()
        assert d["metadata"]["key_id"] == "key_2"
        assert d["metadata"]["fallback_attempts"] == 1
        results["TEST 8 - Failover 429"] = "PASS"
        print("TEST 8 - Failover 429: PASS", d["metadata"])
except Exception as e:
    results["TEST 8 - Failover 429"] = f"FAIL: {e}"
    print("TEST 8 - Failover 429: FAIL", e)

# TEST 9 — Failover 500
try:
    with patch("litellm.completion") as mock_comp:
        mock_comp.return_value = mock_llm_response("Key 2 recovered after 500", in_tokens=30, out_tokens=5)
        r = client.post(
            "/chat",
            json={"session_id": "s-failover-2", "message": "Failover 500 test"},
            headers={"X-Simulate-Failure": "key_1=500"}
        )
        assert r.status_code == 200
        d = r.json()
        assert d["metadata"]["key_id"] == "key_2"
        assert d["metadata"]["fallback_attempts"] == 1
        results["TEST 9 - Failover 500"] = "PASS"
        print("TEST 9 - Failover 500: PASS", d["metadata"])
except Exception as e:
    results["TEST 9 - Failover 500"] = f"FAIL: {e}"
    print("TEST 9 - Failover 500: FAIL", e)

# TEST 10 — Every Key Fails (503 and nothing in history)
try:
    s_fail = "s-every-fail"
    r = client.post(
        "/chat",
        json={"session_id": s_fail, "message": "All keys fail test"},
        headers={"X-Simulate-Failure": "*=429"}
    )
    assert r.status_code == 503
    assert r.json() == {"detail": "all providers unavailable"}

    # Verify not stored
    check_hist = client.get(f"/sessions/{s_fail}/messages")
    assert check_hist.status_code == 404
    assert check_hist.json() == {"detail": "session not found"}
    results["TEST 10 - Every key fails"] = "PASS"
    print("TEST 10 - Every key fails: PASS")
except Exception as e:
    results["TEST 10 - Every key fails"] = f"FAIL: {e}"
    print("TEST 10 - Every key fails: FAIL", e)

# TEST 11 — Validation
try:
    v1 = client.post("/chat", json={"message": "no session id"})
    assert v1.status_code == 422
    assert "detail" in v1.json()

    v2 = client.post("/chat", json={"session_id": "s-val", "message": "   "})
    assert v2.status_code == 422
    assert "detail" in v2.json()

    v3 = client.post("/chat", json={"session_id": "s-val", "message": "a" * 4001})
    assert v3.status_code == 422
    assert "detail" in v3.json()

    # None stored
    val_hist = client.get("/sessions/s-val/messages")
    assert val_hist.status_code == 404
    results["TEST 11 - Validation"] = "PASS"
    print("TEST 11 - Validation: PASS")
except Exception as e:
    results["TEST 11 - Validation"] = f"FAIL: {e}"
    print("TEST 11 - Validation: FAIL", e)

# TEST 12 — Frontend contract test IDs
try:
    app_tsx = open(os.path.join(os.path.dirname(__file__), "..", "frontend", "src", "App.tsx"), encoding="utf-8").read()
    required_test_ids = [
        'data-testid="chat-input"',
        'data-testid="send-button"',
        'data-testid="assistant-message"',
        'data-testid="meta-source"',
        'data-testid="meta-model"',
        'data-testid="meta-latency"',
        'data-testid="meta-input-tokens"',
        'data-testid="meta-output-tokens"',
        'data-testid="system-prompt-input"'
    ]
    for tid in required_test_ids:
        assert tid in app_tsx, f"Missing test id: {tid}"
    results["TEST 12 - Frontend contract"] = "PASS"
    print("TEST 12 - Frontend contract: PASS")
except Exception as e:
    results["TEST 12 - Frontend contract"] = f"FAIL: {e}"
    print("TEST 12 - Frontend contract: FAIL", e)

print("\n=== SUMMARY OF ALL CHECKS ===")
for test_name, status in results.items():
    print(f"{test_name}: {status}")

# Cleanup
if os.path.exists(test_db):
    try:
        os.remove(test_db)
    except Exception:
        pass
