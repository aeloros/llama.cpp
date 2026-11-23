import os
import threading
import time

import pytest
from utils import *

# A second copy of the model (--model-copy-device): the server loads it when a task is sent to it and unloads it when
# it has been without a task for --model-copy-release seconds. By default the copy is on the CPU, beside the first
# model ("none" names no device), so the tests run on any build. Set LLAMA_TEST_MODEL_COPY_DEVICE to a device that
# the server lists, e.g. CUDA0, to have the copy there; the tests of the device's memory need that.
DEVICE = os.environ.get("LLAMA_TEST_MODEL_COPY_DEVICE")
needs_device = pytest.mark.skipif(not DEVICE, reason="LLAMA_TEST_MODEL_COPY_DEVICE names no device for the copy")

server = ServerPreset.tinyllama2()

PROMPT = "Once upon a time there was a little cat who lived in a big house."


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.kv_unified = True
    server.n_ctx = 4096
    server.n_slots = 2
    server.server_slots = True
    server.temperature = 0.0
    server.model_copy_device = DEVICE or "none"
    server.model_copy_ctx_size = 1024
    server.model_copy_parallel = 1
    server.model_copy_after = 0
    server.model_copy_release = 2


def the_copy():
    res = server.make_request("GET", "/props")
    assert res.status_code == 200
    return res.body["model_copy"]


def wait_until(what, cond, seconds=10.0):
    t_end = time.time() + seconds
    while time.time() < t_end:
        if cond():
            return
        time.sleep(0.05)
    raise TimeoutError(what)


def complete(n_predict=16, prompt=PROMPT, **more):
    data = {"prompt": prompt, "n_predict": n_predict, "cache_prompt": False}
    data.update(more)
    res = server.make_request("POST", "/completion", data=data)
    assert res.status_code == 200, res.body
    return res.body


def while_a_task_runs(data, then):
    """Starts a long request, waits until a slot of the first model works on it, calls then(), and returns what the
    long request and then() gave."""
    out = {}

    def run():
        out["res"] = server.make_request("POST", "/completion", data=data)

    t = threading.Thread(target=run)
    t.start()
    try:
        t_end = time.time() + 10
        while time.time() < t_end:
            res = server.make_request("GET", "/slots")
            if res.status_code == 200 and any(slot["is_processing"] for slot in res.body):
                return out, then()
            time.sleep(0.002)
        raise TimeoutError("no slot started processing")
    finally:
        t.join()


def test_without_the_option_there_is_no_copy():
    global server
    server.model_copy_device = None
    server.model_copy_ctx_size = None
    server.model_copy_parallel = None
    server.model_copy_after = None
    server.model_copy_release = None
    server.start()

    res = server.make_request("GET", "/props")
    assert res.status_code == 200
    assert "model_copy" not in res.body
    assert complete()["tokens_predicted"] == 16


def test_the_copy_is_loaded_for_a_task_and_unloaded_when_it_is_idle():
    global server

    # what the first model writes for this prompt, from a server without the copy
    server.model_copy_device = None
    server.start()
    expected = complete(n_predict=32)["content"]
    server.stop()

    # --model-copy-after 0: a task goes to the copy whenever the copy has room for it
    server.model_copy_device = DEVICE or "none"
    server.start()

    copy = the_copy()
    assert copy["state"] == "unloaded"
    assert copy["n_loads"] == 0
    assert copy["n_ctx"] == 1024
    assert copy["total_slots"] == 1

    body = complete(n_predict=32)
    assert body["content"] == expected
    copy = the_copy()
    assert copy["state"] == "loaded"
    assert copy["n_loads"] == 1
    assert copy["n_tasks_total"] == 1
    assert copy["tasks"] == 0

    # a second task finds it loaded
    assert complete(n_predict=32)["content"] == expected
    copy = the_copy()
    assert copy["n_loads"] == 1
    assert copy["n_tasks_total"] == 2

    # two seconds without a task, and it is unloaded
    wait_until("the copy was not unloaded", lambda: the_copy()["state"] == "unloaded", seconds=8.0)

    # the next task loads it again
    assert complete(n_predict=32)["content"] == expected
    copy = the_copy()
    assert copy["state"] == "loaded"
    assert copy["n_loads"] == 2
    assert copy["n_tasks_total"] == 3


def test_a_task_goes_to_the_copy_when_the_first_model_has_its_tasks():
    global server
    server.model_copy_after = 1
    server.start()

    # one task at a time stays with the first model, and the copy is not loaded for it
    for _ in range(3):
        complete()
    copy = the_copy()
    assert copy["state"] == "unloaded"
    assert copy["n_tasks_total"] == 0

    # while the first model works on one task, the next goes to the copy
    long_task = {"prompt": PROMPT, "n_predict": 1900, "ignore_eos": True, "cache_prompt": False}
    out, body = while_a_task_runs(long_task, lambda: complete(n_predict=16))
    assert body["tokens_predicted"] == 16
    assert out["res"].status_code == 200
    assert out["res"].body["tokens_predicted"] == 1900

    copy = the_copy()
    assert copy["n_loads"] == 1
    assert copy["n_tasks_total"] == 1
    assert copy["tasks"] == 0
    assert copy["tasks_first"] == 0


def test_the_copy_takes_only_what_fits_there():
    global server
    server.start()

    res = server.make_request("POST", "/tokenize", data={"content": PROMPT, "add_special": True})
    n_prompt = len(res.body["tokens"])

    # 1024 cells: a task whose prompt and output need more stays with the first model
    body = complete(n_predict=1024 - n_prompt)
    assert body["tokens_predicted"] == 1024 - n_prompt
    copy = the_copy()
    assert copy["state"] == "unloaded"
    assert copy["n_tasks_total"] == 0

    # one that fits goes there
    body = complete(n_predict=1024 - n_prompt - 1)
    assert body["tokens_predicted"] == 1024 - n_prompt - 1
    assert the_copy()["n_tasks_total"] == 1


def test_a_task_stays_with_the_first_model_while_the_copy_has_no_free_slot():
    global server
    server.model_copy_ctx_size = 2048
    server.model_copy_release = 60
    server.start()

    # the copy has one slot: while it works on a task, the next task goes to the first model
    long_task = {"prompt": PROMPT, "n_predict": 1900, "ignore_eos": True, "cache_prompt": False}

    def then():
        wait_until("the long task did not reach the copy", lambda: the_copy()["tasks"] == 1)
        body = complete(n_predict=16)
        return body, the_copy()

    out = {}

    def run():
        out["res"] = server.make_request("POST", "/completion", data=long_task)

    t = threading.Thread(target=run)
    t.start()
    try:
        body, copy = then()
    finally:
        t.join()

    assert body["tokens_predicted"] == 16
    assert out["res"].status_code == 200
    assert out["res"].body["tokens_predicted"] == 1900
    # the short task was counted with the first model: the copy had the long one only
    assert the_copy()["n_tasks_total"] == 1


@needs_device
def test_the_copy_is_not_loaded_when_its_device_has_not_the_memory():
    global server
    # more than any device has, in MiB: what has to stay free on the device beside the copy
    server.model_copy_headroom = 1000000
    server.start()

    assert complete()["tokens_predicted"] == 16
    copy = the_copy()
    assert copy["state"] == "unloaded"
    assert copy["n_loads"] == 0
    assert copy["n_tasks_total"] == 0
    assert copy["n_refused"] == 1
    # what the copy takes of the device was measured without loading it
    assert copy["need_mib"][DEVICE] > 0

    assert complete()["tokens_predicted"] == 16
    assert the_copy()["n_refused"] == 2


@needs_device
def test_the_copy_is_loaded_when_its_device_has_the_memory():
    global server
    server.model_copy_headroom = 256
    server.start()

    assert complete()["tokens_predicted"] == 16
    copy = the_copy()
    assert copy["state"] == "loaded"
    assert copy["n_loads"] == 1
    assert copy["n_tasks_total"] == 1
    assert copy["n_refused"] == 0
    assert copy["need_mib"][DEVICE] > 0


MODEL_DRAFT_FILE_URL = "https://huggingface.co/ggml-org-dev/tiny-llamas/resolve/main/stories15M-q4_0.gguf"


def start_with_a_draft_model(copy_drafts):
    """A server whose first model drafts tokens with a draft model, and a copy that every task goes to."""
    global server
    server = ServerPreset.stories15m_moe()
    server.model_draft = download_file(MODEL_DRAFT_FILE_URL)
    server.spec_type = "draft-simple"
    server.spec_draft_n_min = 4
    server.spec_draft_n_max = 8
    server.fa = "off"
    server.kv_unified = True
    server.n_slots = 2
    server.model_copy_device = DEVICE or "none"
    server.model_copy_ctx_size = 1024
    server.model_copy_parallel = 1
    server.model_copy_after = 0
    server.model_copy_release = 60
    if not copy_drafts:
        server.model_copy_draft = False
    server.start()


DRAFTED = {
    "prompt": "I believe the meaning of life is",
    "temperature": 0.2,
    "top_k": 5,
    "seed": 4242,
    "n_predict": 16,
}


def test_the_copy_drafts_tokens_as_the_first_model_does():
    start_with_a_draft_model(copy_drafts=True)
    res = server.make_request("POST", "/completion", data=DRAFTED)
    assert res.status_code == 200
    assert the_copy()["n_tasks_total"] == 1
    assert res.body["timings"]["draft_n"] > 0


def test_the_copy_does_not_draft_with_no_model_copy_draft():
    start_with_a_draft_model(copy_drafts=False)
    res = server.make_request("POST", "/completion", data=DRAFTED)
    assert res.status_code == 200
    assert the_copy()["n_tasks_total"] == 1
    assert res.body["timings"].get("draft_n", 0) == 0


# A server with a multimodal projector: its prompts are lists of text tokens and media, and the copy has no projector.

IMG_URL = "https://huggingface.co/ggml-org-dev/tinygemma3-GGUF/resolve/main/test/11_truck.png"


def start_with_a_projector():
    global server
    server = ServerPreset.tinygemma3()
    server.kv_unified = True
    server.model_copy_device = DEVICE or "none"
    server.model_copy_ctx_size = 1024
    server.model_copy_parallel = 1
    server.model_copy_after = 0
    server.model_copy_release = 60
    server.start()


def test_a_text_task_of_a_server_with_a_projector_goes_to_the_copy():
    start_with_a_projector()
    for i in range(2):
        res = server.make_request("POST", "/chat/completions", data={
            "temperature": 0.0,
            "max_tokens": 8,
            "messages": [{"role": "user", "content": "What is the color of the sky?"}],
        })
        assert res.status_code == 200, res.body
        assert res.body["choices"][0]["message"]["content"] is not None
        assert the_copy()["n_tasks_total"] == i + 1


def test_a_task_with_an_image_stays_with_the_first_model():
    import base64
    import requests

    try:
        response = requests.get(IMG_URL, timeout=20)
        response.raise_for_status()
    except Exception as e:
        pytest.skip("the test image could not be fetched: %s" % e)
    image = "data:image/png;base64," + base64.b64encode(response.content).decode("utf-8")

    start_with_a_projector()
    res = server.make_request("POST", "/chat/completions", data={
        "temperature": 0.0,
        "max_tokens": 8,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "What is this:"},
            {"type": "image_url", "image_url": {"url": image}},
        ]}],
    })
    assert res.status_code == 200, res.body
    copy = the_copy()
    assert copy["n_tasks_total"] == 0
    assert copy["state"] == "unloaded"


# The options of a server that keeps its model on a device, with a slot that comes first, quantized KV cells and a copy
# of the model, and the requests that such a server gets, at once: streamed and not, with a slot id, text and images.

def start_like_a_house_server(preset, more_args, cells="q4_0"):
    global server
    server = preset
    server.n_slots = 4
    server.n_ctx = 2048
    server.n_batch = 512
    server.n_gpu_layer = 999
    server.kv_unified = True
    server.no_cache_idle_slots = True
    server.server_metrics = True
    server.cache_ram = 512
    server.ctk = cells
    server.ctv = cells
    server.fa = "on"
    server.model_copy_device = DEVICE
    server.model_copy_ctx_size = 2048
    server.model_copy_parallel = 2
    server.model_copy_after = 1
    server.model_copy_release = 60
    server.model_copy_headroom = 256
    server.model_copy_draft = False
    server.extra_args = [
        "--device", DEVICE,
        "--kv-unified-per-slot", "512,2048,2048,2048",
        "--batch-priority-slots", "0",
        "--batch-size-busy", "64",
    ] + more_args
    server.start()


def requests_at_once(n_rounds, house_request):
    """Six requests of coding clients and two of the house, n_rounds times; returns the ones that failed."""
    text = "Tell me a long story about a cat and a dog who look for a red ball in the garden. " * 6
    results = []
    lock = threading.Lock()

    def messages(i):
        # the Anthropic endpoint, as a coding client uses it; every second one streamed
        data = {
            "model": "x",
            "max_tokens": 96,
            "messages": [{"role": "user", "content": text + " Part %d." % i}],
            "id_slot": 1 + i % 3,
        }
        if i % 2 == 0:
            res = server.make_request("POST", "/v1/messages", data=data)
            ok = res.status_code == 200
        else:
            data["stream"] = True
            ok = len(list(server.make_stream_request("POST", "/v1/messages", data=data))) > 0
        with lock:
            results.append(("messages %d" % i, ok))

    def house(i):
        res = server.make_request("POST", "/chat/completions", data=house_request)
        with lock:
            results.append(("house %d" % i, res.status_code == 200))

    for _ in range(n_rounds):
        threads = [threading.Thread(target=messages, args=(i,)) for i in range(6)]
        threads += [threading.Thread(target=house, args=(i,)) for i in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    assert len(results) == 8 * n_rounds
    return [r for r in results if not r[1]]


@needs_device
def test_a_server_with_a_projector_a_slot_that_comes_first_and_a_copy():
    import base64
    import requests

    start_like_a_house_server(ServerPreset.tinygemma3(), ["--no-mmproj-offload"])

    content = [{"type": "text", "text": "What is this:"}]
    try:
        response = requests.get(IMG_URL, timeout=20)
        response.raise_for_status()
        content.append({"type": "image_url", "image_url": {
            "url": "data:image/png;base64," + base64.b64encode(response.content).decode("utf-8")}})
    except Exception:
        pass  # without the image the house's requests are text

    failed = requests_at_once(3, {
        "temperature": 0.0,
        "max_tokens": 8,
        "messages": [{"role": "user", "content": content}],
        "id_slot": 0,
    })
    assert failed == []
    assert server.make_request("GET", "/health").status_code == 200
    copy = the_copy()
    assert copy["n_tasks_total"] >= 1
    assert copy["tasks"] == 0
    assert copy["tasks_first"] == 0


@needs_device
def test_a_server_with_a_second_pool_a_slot_that_comes_first_and_a_copy():
    start_like_a_house_server(ServerPreset.tinyllama2(), [
        "--kv-overflow-device", DEVICE,
        "--kv-overflow-size", "4096",
        "--kv-overflow-release", "60",
        "--kv-overflow-headroom", "256",
    ], cells=None)  # the heads of this model are too small for quantized cells

    failed = requests_at_once(3, {
        "temperature": 0.0,
        "max_tokens": 8,
        "messages": [{"role": "user", "content": "What is the color of the sky?"}],
        "id_slot": 0,
    })
    assert failed == []
    assert server.make_request("GET", "/health").status_code == 200
    copy = the_copy()
    assert copy["n_tasks_total"] >= 1
    assert copy["tasks"] == 0
    assert copy["tasks_first"] == 0
