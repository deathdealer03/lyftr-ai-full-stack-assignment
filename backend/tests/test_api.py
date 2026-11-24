import json
from unittest.mock import patch, MagicMock
import pytest
from litellm.types.utils import ModelResponse, Choices, Message, Usage
from app.config import settings


def create_mock_completion(content="The capital of France is Paris.", in_tokens=42, out_tokens=9, model=None):
    model = model or settings.LLM_MODEL
    return ModelResponse(
        id="chatcmpl-mock-123",
        choices=[Choices(
            finish_reason="stop",
            index=0,
            message=Message(content=content, role="assistant")
        )],
        usage=Usage(
            prompt_tokens=in_tokens,
            completion_tokens=out_tokens,
            total_tokens=in_tokens + out_tokens
        ),
        model=model
    )


# TEST 1 — Health
def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["keys_configured"] >= 2
    assert data["cache_enabled"] is True


# TEST 11 — Validation
def test_validation_errors(client):
    # Missing session_id
    r = client.post("/chat", json={"message": "Hello"})
    assert r.status_code == 422
    assert "detail" in r.json()

    # Empty session_id
    r = client.post("/chat", json={"session_id": "   ", "message": "Hello"})
    assert r.status_code == 422
    assert "detail" in r.json()

    # Missing message
    r = client.post("/chat", json={"session_id": "s-123"})
    assert r.status_code == 422
    assert "detail" in r.json()

    # Empty message
    r = client.post("/chat", json={"session_id": "s-123", "message": ""})
    assert r.status_code == 422
    assert "detail" in r.json()

    # Whitespace message
    r = client.post("/chat", json={"session_id": "s-123", "message": "   \n\t  "})
    assert r.status_code == 422
    assert "detail" in r.json()

    # Message over 4000 characters
    r = client.post("/chat", json={"session_id": "s-123", "message": "a" * 4001})
    assert r.status_code == 422
    assert "detail" in r.json()

    # Malformed JSON
    r = client.post(
        "/chat",
        content="not valid json",
        headers={"Content-Type": "application/json"}
    )
    assert r.status_code == 422
    assert "detail" in r.json()

    # Ensure none of these created a session history
    hist = client.get("/sessions/s-123/messages")
    assert hist.status_code == 404
    assert hist.json()["detail"] == "session not found"


# TEST 2 — Live inference
@patch("litellm.completion")
def test_live_inference(mock_completion, client):
    mock_completion.return_value = create_mock_completion("Paris is the capital of France.", in_tokens=35, out_tokens=8)

    session_id = "s-test-live-1"
    response = client.post("/chat", json={
        "session_id": session_id,
        "message": "What is the capital of France?"
    })

    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == session_id
    assert data["reply"] == "Paris is the capital of France."

    meta = data["metadata"]
    assert meta["resolution_source"] == "LIVE_INFERENCE"
    assert meta["model"] == settings.LLM_MODEL
    assert isinstance(meta["latency_ms"], int)
    assert meta["latency_ms"] >= 0
    assert meta["input_tokens"] == 35
    assert meta["output_tokens"] == 8
    assert meta["key_id"] == "key_1"
    assert meta["fallback_attempts"] == 0


# TEST 3 — Cache hit
@patch("litellm.completion")
def test_response_cache_hit(mock_completion, client):
    mock_completion.return_value = create_mock_completion("Berlin is the capital of Germany.", in_tokens=40, out_tokens=7)

    # First session (Live inference)
    r1 = client.post("/chat", json={
        "session_id": "session-germany-1",
        "message": "What is the capital of Germany?"
    })
    assert r1.status_code == 200
    assert r1.json()["metadata"]["resolution_source"] == "LIVE_INFERENCE"
    assert mock_completion.call_count == 1

    # Second completely new session, exact same first message and context
    r2 = client.post("/chat", json={
        "session_id": "session-germany-2",
        "message": "What is the capital of Germany?"
    })
    assert r2.status_code == 200
    data2 = r2.json()

    # LLM completion should NOT be called a second time
    assert mock_completion.call_count == 1
    assert data2["reply"] == "Berlin is the capital of Germany."
    meta2 = data2["metadata"]
    assert meta2["resolution_source"] == "LLM_CACHE"
    assert meta2["key_id"] is None
    assert meta2["fallback_attempts"] == 0
    assert meta2["input_tokens"] == 40
    assert meta2["output_tokens"] == 7
    assert isinstance(meta2["latency_ms"], int)


# TEST 4 — Different cache key
@patch("litellm.completion")
def test_different_message_not_cached(mock_completion, client):
    mock_completion.side_effect = [
        create_mock_completion("Answer 1", in_tokens=20, out_tokens=5),
        create_mock_completion("Answer 2", in_tokens=25, out_tokens=6),
    ]

    r1 = client.post("/chat", json={
        "session_id": "s-diff-1",
        "message": "First question"
    })
    assert r1.status_code == 200
    assert r1.json()["reply"] == "Answer 1"

    r2 = client.post("/chat", json={
        "session_id": "s-diff-2",
        "message": "A completely different question"
    })
    assert r2.status_code == 200
    assert r2.json()["reply"] == "Answer 2"
    assert r2.json()["metadata"]["resolution_source"] == "LIVE_INFERENCE"


# TEST 5 — Multi-turn conversation
@patch("litellm.completion")
def test_multi_turn_conversation(mock_completion, client):
    def side_effect_cb(*args, **kwargs):
        messages = kwargs.get("messages", [])
        # Check if earlier message exists in conversation sent to LLM
        history_contents = [m["content"] for m in messages]
        if any("My name is Alex" in c for c in history_contents) and any("What is my name?" in c for c in history_contents):
            return create_mock_completion("Your name is Alex.", in_tokens=50, out_tokens=5)
        return create_mock_completion("Hello Alex, nice to meet you!", in_tokens=30, out_tokens=8)

    mock_completion.side_effect = side_effect_cb

    session_id = "s-multiturn-alex"

    # Turn 1
    r1 = client.post("/chat", json={
        "session_id": session_id,
        "message": "My name is Alex."
    })
    assert r1.status_code == 200
    assert "Hello Alex" in r1.json()["reply"]

    # Turn 2: frontend sends ONLY the new message
    r2 = client.post("/chat", json={
        "session_id": session_id,
        "message": "What is my name?"
    })
    assert r2.status_code == 200
    assert r2.json()["reply"] == "Your name is Alex."

    # Verify history sent to LiteLLM contained system prompt + 2 user messages + 1 assistant message
    last_call_messages = mock_completion.call_args.kwargs["messages"]
    roles = [m["role"] for m in last_call_messages]
    assert roles == ["system", "user", "assistant", "user"]
    assert last_call_messages[1]["content"] == "My name is Alex."
    assert last_call_messages[3]["content"] == "What is my name?"


# TEST 6 — Session isolation
@patch("litellm.completion")
def test_session_isolation(mock_completion, client):
    recorded_messages = []

    def mock_cb(*args, **kwargs):
        msgs = kwargs.get("messages", [])
        recorded_messages.append(msgs)
        return create_mock_completion("Reply", in_tokens=20, out_tokens=4)

    mock_completion.side_effect = mock_cb

    # Session A
    client.post("/chat", json={
        "session_id": "session-A",
        "message": "My favorite color is blue."
    })

    # Session B
    client.post("/chat", json={
        "session_id": "session-B",
        "message": "What is my favorite color?"
    })

    # Check session B messages did NOT contain Session A content
    session_b_msgs = recorded_messages[1]
    b_contents = [m["content"] for m in session_b_msgs]
    assert not any("blue" in c for c in b_contents)


# TEST 7 — History endpoint
@patch("litellm.completion")
def test_history_endpoint(mock_completion, client):
    mock_completion.return_value = create_mock_completion("The capital of France is Paris.", in_tokens=42, out_tokens=9)

    session_id = "s-history-check"
    client.post("/chat", json={
        "session_id": session_id,
        "message": "What is the capital of France?"
    })

    r = client.get(f"/sessions/{session_id}/messages")
    assert r.status_code == 200
    data = r.json()
    assert data["session_id"] == session_id
    messages = data["messages"]
    assert len(messages) == 2

    # Oldest first
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "What is the capital of France?"
    assert "metadata" not in messages[0] or messages[0]["metadata"] is None

    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == "The capital of France is Paris."
    assert messages[1]["metadata"]["resolution_source"] == "LIVE_INFERENCE"
    assert messages[1]["metadata"]["input_tokens"] == 42
    assert messages[1]["metadata"]["output_tokens"] == 9

    # System prompt must NOT be in returned history
    assert not any(m["role"] == "system" for m in messages)

    # Unknown session returns 404
    r_unknown = client.get("/sessions/s-nonexistent-999/messages")
    assert r_unknown.status_code == 404
    assert r_unknown.json() == {"detail": "session not found"}


# TEST 8 — Failover 429
@patch("litellm.completion")
def test_failover_429(mock_completion, client):
    mock_completion.return_value = create_mock_completion("Real answer from key 2", in_tokens=30, out_tokens=6)

    response = client.post(
        "/chat",
        json={"session_id": "s-failover-429", "message": "Testing failover"},
        headers={"X-Simulate-Failure": "key_1=429"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["reply"] == "Real answer from key 2"
    meta = data["metadata"]
    assert meta["key_id"] == "key_2"
    assert meta["fallback_attempts"] == 1
    assert meta["resolution_source"] == "LIVE_INFERENCE"


# TEST 9 — Failover 500
@patch("litellm.completion")
def test_failover_500(mock_completion, client):
    mock_completion.return_value = create_mock_completion("Real answer after 500", in_tokens=25, out_tokens=5)

    response = client.post(
        "/chat",
        json={"session_id": "s-failover-500", "message": "Testing 500 failover"},
        headers={"X-Simulate-Failure": "key_1=500"}
    )
    assert response.status_code == 200
    data = response.json()
    meta = data["metadata"]
    assert meta["key_id"] == "key_2"
    assert meta["fallback_attempts"] == 1


# TEST 10 — Every key fails (503 and nothing in history)
@patch("litellm.completion")
def test_all_keys_fail_503(mock_completion, client):
    session_id = "s-all-fail"
    response = client.post(
        "/chat",
        json={"session_id": session_id, "message": "This will fail all keys"},
        headers={"X-Simulate-Failure": "*=429"}
    )
    assert response.status_code == 503
    assert response.json() == {"detail": "all providers unavailable"}

    # Verify nothing was added to history
    hist = client.get(f"/sessions/{session_id}/messages")
    assert hist.status_code == 404
    assert hist.json() == {"detail": "session not found"}


# TEST 12 — Secrets never leaked
@patch("litellm.completion")
def test_secrets_never_leaked(mock_completion, client):
    mock_completion.return_value = create_mock_completion("Safe answer", in_tokens=10, out_tokens=2)

    response = client.post("/chat", json={
        "session_id": "s-secret-test",
        "message": "Hello secret test"
    })
    raw_response = response.text
    # Ensure raw secret keys from environment are not present
    assert "test_key_alpha" not in raw_response
    assert "test_key_beta" not in raw_response


# Optional system prompt test
@patch("litellm.completion")
def test_custom_system_prompt(mock_completion, client):
    mock_completion.return_value = create_mock_completion("Yarr, ahoy matey!", in_tokens=25, out_tokens=5)

    session_id = "s-pirate-test"
    system_prompt = "You are a pirate. Speak only in pirate slang."

    response = client.post("/chat", json={
        "session_id": session_id,
        "message": "Hello there!",
        "system_prompt": system_prompt
    })
    assert response.status_code == 200

    # Verify system prompt was passed to LiteLLM
    call_messages = mock_completion.call_args.kwargs["messages"]
    assert call_messages[0]["role"] == "system"
    assert call_messages[0]["content"] == system_prompt

    # Verify history does NOT contain system prompt
    hist = client.get(f"/sessions/{session_id}/messages").json()
    assert not any(m["role"] == "system" for m in hist["messages"])


# TEST 13 — SQLite persistence across connection resets
@patch("litellm.completion")
def test_sqlite_persistence_across_reconnect(mock_completion, client):
    mock_completion.return_value = create_mock_completion("Persisted reply", in_tokens=15, out_tokens=3)

    session_id = "s-persistence-check"
    r = client.post("/chat", json={
        "session_id": session_id,
        "message": "Remember this persistence test"
    })
    assert r.status_code == 200

    # Read back from fresh client
    from app.database import get_session_messages
    stored_messages = get_session_messages(session_id)
    assert len(stored_messages) == 2
    assert stored_messages[0]["role"] == "user"
    assert stored_messages[0]["content"] == "Remember this persistence test"
    assert stored_messages[1]["role"] == "assistant"
    assert stored_messages[1]["content"] == "Persisted reply"

