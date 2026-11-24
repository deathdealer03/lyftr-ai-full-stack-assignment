import os
import sys
import shutil
import pytest
from fastapi.testclient import TestClient

# Ensure backend directory is in sys.path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
os.environ["LLM_API_KEYS"] = "test_key_alpha,test_key_beta"
os.environ["ENABLE_FAILURE_SIMULATION"] = "true"

from app.config import settings
from app.database import init_db
from app.main import app
import app.database as db_module
import app.llm_service as llm_service_module

TEST_TEMP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".test_db_dir"))


@pytest.fixture(autouse=True)
def setup_test_db():
    """Creates an isolated clean SQLite database in local workspace for each test."""
    os.makedirs(TEST_TEMP_DIR, exist_ok=True)
    test_db_file = os.path.join(TEST_TEMP_DIR, "test_app.db")
    if os.path.exists(test_db_file):
        try:
            os.remove(test_db_file)
        except Exception:
            pass

    db_module.DB_PATH = test_db_file
    init_db()

    # Clear LiteLLM cache between tests
    if hasattr(llm_service_module.litellm, "cache") and llm_service_module.litellm.cache:
        try:
            if hasattr(llm_service_module.litellm.cache, "cache"):
                if hasattr(llm_service_module.litellm.cache.cache, "cache_dict"):
                    llm_service_module.litellm.cache.cache.cache_dict.clear()
        except Exception:
            pass

    yield test_db_file

    if os.path.exists(test_db_file):
        try:
            os.remove(test_db_file)
        except Exception:
            pass


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c
