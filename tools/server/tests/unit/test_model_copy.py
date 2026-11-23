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
