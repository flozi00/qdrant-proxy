import importlib.util
import sys
import types
from pathlib import Path

import pytest

SERVICE_ROOT = Path(__file__).resolve().parents[1]
TEST_PACKAGE = "_qdrant_embedding_test_pkg"


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="session", autouse=True)
def inserted_document():
    yield


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


previous_config_module = sys.modules.get("config")
previous_state_module = sys.modules.get("state")
previous_timings_module = sys.modules.get("utils.timings")

config_module = types.ModuleType("config")
config_module.settings = types.SimpleNamespace(
    dense_model_name="dense-test-model",
    colbert_model_name="colbert-test-model",
    colbert_endpoint_configured=False,
)
sys.modules["config"] = config_module

state_module = types.ModuleType("state")
state_module.get_app_state = lambda: types.SimpleNamespace(
    dense_model_id="dense-test-model",
    colbert_model_id="colbert-test-model",
    colbert_vector_size=128,
)
sys.modules["state"] = state_module

timings_module = types.ModuleType("utils.timings")


def _linetimer(*args, **kwargs):
    def decorator(fn):
        return fn

    return decorator


timings_module.linetimer = _linetimer
sys.modules["utils.timings"] = timings_module

package_module = types.ModuleType(TEST_PACKAGE)
package_module.__path__ = [str(SERVICE_ROOT / "services")]
sys.modules[TEST_PACKAGE] = package_module

try:
    embedding_module = _load_module(
        f"{TEST_PACKAGE}.embedding",
        SERVICE_ROOT / "services" / "embedding.py",
    )
finally:
    if previous_config_module is not None:
        sys.modules["config"] = previous_config_module
    else:
        sys.modules.pop("config", None)
    if previous_state_module is not None:
        sys.modules["state"] = previous_state_module
    else:
        sys.modules.pop("state", None)
    if previous_timings_module is not None:
        sys.modules["utils.timings"] = previous_timings_module
    else:
        sys.modules.pop("utils.timings", None)


class _FakeEmbeddingsAPI:
    def __init__(self, handler):
        self._handler = handler

    def create(self, input, model):
        return self._handler(input=input, model=model)


class _FakeDenseClient:
    def __init__(self, handler):
        self.embeddings = _FakeEmbeddingsAPI(handler)


def _embedding_response(values):
    return types.SimpleNamespace(
        data=[types.SimpleNamespace(embedding=value) for value in values]
    )


@pytest.mark.anyio
async def test_encode_dense_retries_with_shorter_text_after_context_error(monkeypatch):
    calls = []
    long_text = "x" * 20000

    def handler(input, model):
        calls.append(input)
        if len(calls) == 1:
            raise RuntimeError(
                "Error code: 400 - {'error': {'message': \"You passed 8193 input tokens and requested 0 output tokens. However, the model's context length is only 8192 tokens.\"}}"
            )
        return _embedding_response([[0.1, 0.2, 0.3]])

    monkeypatch.setattr(embedding_module, "_dense_client", _FakeDenseClient(handler))

    vector = await embedding_module.encode_dense(long_text)

    assert vector == [0.1, 0.2, 0.3]
    assert len(calls) == 2
    assert len(calls[1]) < len(calls[0])


@pytest.mark.anyio
async def test_encode_dense_batch_falls_back_to_individual_retries(monkeypatch):
    batch_calls = []
    single_calls = []

    def handler(input, model):
        if isinstance(input, list):
            batch_calls.append(input)
            raise RuntimeError(
                "Error code: 400 - {'error': {'message': \"You passed 9000 input tokens and requested 0 output tokens. However, the model's context length is only 8192 tokens.\"}}"
            )

        single_calls.append(input)
        return _embedding_response([[0.4, 0.5]])

    monkeypatch.setattr(embedding_module, "_dense_client", _FakeDenseClient(handler))

    vectors = await embedding_module.encode_dense_batch(["a" * 10000, "b" * 10000])

    assert vectors == [[0.4, 0.5], [0.4, 0.5]]
    assert len(batch_calls) == 1
    assert len(single_calls) == 2
