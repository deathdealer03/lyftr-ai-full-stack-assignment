# Lyftr AI - LLM Chatbot with Key Failover & Response Caching

A production-quality full-stack AI chatbot application featuring automated key pool failover, LiteLLM built-in response caching, server-measured token tracking, multi-turn conversational state, and SQLite persistence.

---

## 1. Live URLs

| Tier | Deployment Platform | Status / URL |
| :--- | :--- | :--- |
| **Frontend UI** | Vercel / Netlify / Render | `https://lyftr-ai-frontend.vercel.app` *(Replace with your deployed URL)* |
| **Backend API** | Render / Railway / Fly.io | `https://lyftr-ai-backend.onrender.com` *(Replace with your deployed URL)* |

> **Note on Deployment Readiness:**
> The project is pre-configured and 100% deployment-ready:
> - The frontend reads `VITE_BACKEND_URL` to connect to the deployed backend without any hardcoded localhost URLs.
> - The backend configures CORS dynamically via the `ALLOWED_ORIGINS` environment variable.
> - See the [Deployment Steps](#deployment-guide) below for one-click setup on Render / Railway and Vercel.

---

## 2. Run locally: with Docker and without

### Prerequisites
- **Python:** 3.11+ (Python 3.12 recommended)
- **Node.js:** v18+ (v20+ recommended) and npm
- **Docker & Docker Compose:** Optional for containerized deployment
- **Make:** Optional for CLI automation

---

### Option A: Running with Docker (Recommended)

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd Lyftre
   ```

2. **Configure environment variables:**
   ```bash
   cp .env.example .env
   ```
   Add your API keys to `LLM_API_KEYS` in `.env` (comma-separated, at least 2 keys).

3. **Start the complete stack with Makefile:**
   ```bash
   make up
   ```
   *Or with Docker Compose directly:*
   ```bash
   docker compose up --build -d
   ```

4. **Verify the services:**
   - **Frontend UI:** [http://localhost:3000](http://localhost:3000)
   - **Backend Health:** [http://localhost:8000/health](http://localhost:8000/health)
   - **Interactive API Docs (Swagger):** [http://localhost:8000/docs](http://localhost:8000/docs)

5. **Stop the stack:**
   ```bash
   make down
   ```
   *Conversation history is persisted in the named Docker volume `lyftr_sqlite_data` mounted at `/data`.*

---

### Option B: Running locally without Docker

#### 1. Backend Setup
```bash
# Navigate to backend directory
cd backend

# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .env from example (in repository root or backend folder)
cp ../.env.example .env

# Start FastAPI server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Backend will be available at `http://localhost:8000`.

#### 2. Frontend Setup
```bash
# Open a new terminal and navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start Vite development server
npm run dev
```
Frontend will be available at `http://localhost:3000`.

---

## 3. Design decisions: key fallback, caching, SQLite session storage

### A. Key Fallback & Resilience Strategy
- **Key Pool Architecture:** The backend consumes `LLM_API_KEYS` (a comma-separated string) and generates safe internal identifiers (`key_1`, `key_2`, ...). Real API keys are **never** returned in API responses, logs, frontend UI, or error payloads.
- **Automatic Fallback:** When calling the LLM provider, if a key encounters:
  - HTTP 429 / Rate Limit error
  - Quota exhaustion
  - HTTP 500, 502, 503, or provider connection drops
  the service automatically captures the error, increments `fallback_attempts`, and transparently fails over to the next configured key in the pool.
- **Exhaustion Handling:** If every provider/key in the pool fails, the service aborts cleanly with an HTTP 503 JSON response: `{"detail": "all providers unavailable"}`.
- **History Integrity:** Requests that fail (422 validation or 503 provider failure) are **never** persisted to conversation history.

### B. Response Caching Strategy
- **LiteLLM Built-in Cache:** Response caching is implemented using LiteLLM's cache (`Cache(type="local")` / `disk`).
- **Deterministic Cache Identity:** The cache key is constructed from `(model, messages)`, where `messages` contains the full conversation context:
  `[system_prompt, ...earlier_session_messages, new_user_message]`.
- **Cache Hit Contract:**
  - `resolution_source` is strictly reported as `"LLM_CACHE"` (vs `"LIVE_INFERENCE"` on live provider execution).
  - `key_id` is returned as `null`.
  - `fallback_attempts` is returned as `0`.
  - Stored token usage (`input_tokens`, `output_tokens`) from the original completion is preserved.
  - Latency is server-measured and orders of magnitude faster (1–5 ms).
- **Session Reusability:** An identical query in a completely new session (with the same system prompt) hits the cache. Different queries never reuse cached responses.

### C. SQLite Session Storage & Multi-Turn State
- **Storage Engine:** SQLite is utilized as the persistent storage engine, requiring no external daemons (no Redis, no PostgreSQL).
- **Persistent Location:** Database path is governed by `DATABASE_URL` (defaulting to `sqlite:////data/app.db` under Docker named volume `lyftr_sqlite_data`).
- **Data Model:**
  ```sql
  CREATE TABLE messages (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      session_id TEXT NOT NULL,
      role TEXT NOT NULL, -- 'user' or 'assistant'
      content TEXT NOT NULL,
      metadata TEXT,     -- JSON string storing assistant metadata
      created_at TEXT NOT NULL -- ISO-8601 UTC server timestamp
  );
  CREATE INDEX idx_messages_session ON messages (session_id, id);

  CREATE TABLE sessions (
      session_id TEXT PRIMARY KEY,
      system_prompt TEXT,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL
  );
  ```
- **Context Handling:** The client sends only the new message. The backend fetches historical turns from SQLite ordered by `id ASC`, builds the prompt, calls the LLM, and persists the turn upon success.
- **System Prompt Isolation:** System prompts are persisted per session but are **strictly excluded** from `GET /sessions/{session_id}/messages`.

---

## 4. Verification: how to see failover, cache hits and per-message token usage

### 1. Health Endpoint Verification
```bash
curl -X GET http://localhost:8000/health
```
**Expected Response (HTTP 200):**
```json
{
  "status": "ok",
  "keys_configured": 2,
  "cache_enabled": true
}
```

---

### 2. Live Inference & Token Tracking
Send a message in a new session:
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "session-demo-1",
    "message": "What is the capital of France?"
  }'
```
**Expected Response (HTTP 200):**
```json
{
  "session_id": "session-demo-1",
  "reply": "The capital of France is Paris.",
  "metadata": {
    "resolution_source": "LIVE_INFERENCE",
    "model": "gemini/gemini-3.8-flash",
    "latency_ms": 780,
    "input_tokens": 42,
    "output_tokens": 9,
    "key_id": "key_1",
    "fallback_attempts": 0
  }
}
```

---

### 3. Response Cache Hit Demonstration
Send the exact same message in a **brand new session** (`session-demo-2`):
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "session-demo-2",
    "message": "What is the capital of France?"
  }'
```
**Expected Response (HTTP 200):**
```json
{
  "session_id": "session-demo-2",
  "reply": "The capital of France is Paris.",
  "metadata": {
    "resolution_source": "LLM_CACHE",
    "model": "gemini/gemini-3.8-flash",
    "latency_ms": 2,
    "input_tokens": 42,
    "output_tokens": 9,
    "key_id": null,
    "fallback_attempts": 0
  }
}
```
*Notice: `resolution_source` is `LLM_CACHE`, `key_id` is `null`, `fallback_attempts` is `0`, token usage matches, and latency is ~2 ms.*

---

### 4. Key Failover Demonstration (`X-Simulate-Failure: key_1=429`)
Trigger a simulated rate limit on `key_1`:
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-Simulate-Failure: key_1=429" \
  -d '{
    "session_id": "session-failover-test",
    "message": "Tell me a joke."
  }'
```
**Expected Response (HTTP 200):**
```json
{
  "session_id": "session-failover-test",
  "reply": "Why did the scarecrow win an award? Because he was outstanding in his field!",
  "metadata": {
    "resolution_source": "LIVE_INFERENCE",
    "model": "gemini/gemini-3.8-flash",
    "latency_ms": 810,
    "input_tokens": 38,
    "output_tokens": 16,
    "key_id": "key_2",
    "fallback_attempts": 1
  }
}
```
*Notice: The request succeeded with HTTP 200, answered using `key_2`, and recorded `fallback_attempts: 1`.*

---

### 5. All Keys Fail Demonstration (`X-Simulate-Failure: *=429`)
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-Simulate-Failure: *=429" \
  -d '{
    "session_id": "session-all-fail",
    "message": "This request will fail."
  }'
```
**Expected Response (HTTP 503):**
```json
{
  "detail": "all providers unavailable"
}
```
Verify nothing was inserted into session history:
```bash
curl -X GET http://localhost:8000/sessions/session-all-fail/messages
```
**Expected Response (HTTP 404):**
```json
{
  "detail": "session not found"
}
```

---

### 6. Conversation History Endpoint
```bash
curl -X GET http://localhost:8000/sessions/session-demo-1/messages
```
**Expected Response (HTTP 200):**
```json
{
  "session_id": "session-demo-1",
  "messages": [
    {
      "role": "user",
      "content": "What is the capital of France?"
    },
    {
      "role": "assistant",
      "content": "The capital of France is Paris.",
      "metadata": {
        "resolution_source": "LIVE_INFERENCE",
        "input_tokens": 42,
        "output_tokens": 9
      }
    }
  ]
}
```

---

### 7. Running Backend Tests & Audit Script
```bash
# Run pytest test suite (14 automated unit & contract tests)
python -m pytest -v --basetemp=.pytest_temp backend/tests

# Run standalone verification audit script
python scripts/verify.py
```

---

## 5. Setup Used: your editor and AI tools

- **Development Environment:** Windows 11 with PowerShell & Python 3.12.9
- **IDE:** Google Antigravity IDE / Visual Studio Code
- **AI Assistant:** Google Gemini 3.8 Flash (High) via Antigravity Agentic Framework
- **Backend Framework:** FastAPI 0.162.2, LiteLLM 1.110.3, Uvicorn 0.54.0, SQLite3
- **Frontend Stack:** React 18, Vite 6, TypeScript 5, Lucide Icons, Pure CSS Modules
- **Testing:** Pytest 9.2.0, HTTPX, unittest.mock
- **Containers:** Multi-stage Dockerfiles (Python 3.12-slim & Node 22-alpine + Nginx Alpine)

---

## 6. Bonus features (if any): what you added beyond the requirements and why

1. **Interactive UI Failover Simulator & Controls Panel:**
   - Added an on-screen simulation selector directly in the chat UI with one-click toggles for `key_1=429`, `key_1=500`, and `*=429`. This allows live demonstration of failover during video recordings without switching to a terminal.
2. **Real-time Live Gateway Health Monitor:**
   - A health badge in the UI header continuously monitors the `/health` endpoint, reporting key pool capacity (`keys_configured`) and cache status.
3. **Session Switcher & Fresh Generator:**
   - A "New Session" button generates a fresh client-side session ID, resets conversational state, and focuses the input in a single click.
4. **Offline Model Cost Map Protection:**
   - Hardcoded `LITELLM_LOCAL_MODEL_COST_MAP=True` to eliminate external network calls to GitHub on LiteLLM startup, preventing runtime freezes in isolated or sandboxed production environments.
5. **Standalone Verification Audit Script:**
   - Provided `scripts/verify.py` which executes an automated end-to-end verification pass for Tests 1 through 12.

---

## <a id="deployment-guide"></a>Deployment Guide

### 1. Backend Deployment (Render / Railway / Fly.io)
1. Fork or push this repository to GitHub.
2. On Render/Railway, create a new **Web Service** pointing to the repository root.
3. Set Environment to **Python 3** (or Docker).
   - Build Command: `pip install -r backend/requirements.txt`
   - Start Command: `cd backend && uvicorn app.main:app --host 0.0.0.0 --port 8000`
4. Set Environment Variables:
   - `LLM_API_KEYS`: `your_key_1,your_key_2`
   - `DATABASE_URL`: `sqlite:///./data/app.db`
   - `ALLOWED_ORIGINS`: `https://your-frontend-domain.vercel.app`
   - `ENABLE_FAILURE_SIMULATION`: `true`
   - `LLM_MODEL`: `gemini/gemini-3.8-flash`

### 2. Frontend Deployment (Vercel / Netlify / Cloudflare Pages)
1. Create a new project pointing to the `frontend/` directory.
2. Framework Preset: **Vite**
3. Build Command: `npm run build`
4. Output Directory: `dist`
5. Set Environment Variable:
   - `VITE_BACKEND_URL`: `https://your-backend-domain.onrender.com`
