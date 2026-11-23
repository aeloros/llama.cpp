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
    assert res.body["total_slots"] == 2
    res = server.make_request("GET", "/slots")
    assert [slot["id"] for slot in res.body] == [0, 1]
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


# Where a conversation goes on. The copy has cells and a prompt cache of its own, so a conversation that it read is
# read again by the first model. By default the first model comes first all the same; with
# --model-copy-keep-conversations the conversation goes on in the copy while the copy is loaded.

STORY = (
    "Once upon a time there was a little cat who lived in a big house. The cat had a friend, a small brown dog, and "
    "every morning they went into the garden to look for the red ball. One day the ball was gone. The cat looked "
    "under the bench and the dog looked behind the tree, and then they asked the old bird who sat on the fence. "
    "The bird said that the wind had taken the ball over the wall, into the garden of the house next door. "
)
LONG_TASK = {"prompt": PROMPT, "n_predict": 1900, "ignore_eos": True, "cache_prompt": False}


def n_tokens(text):
    res = server.make_request("POST", "/tokenize", data={"content": text, "add_special": True})
    return len(res.body["tokens"])


def in_background(data):
    """Starts a request; returns its thread and the dict that gets its response."""
    out = {}

    def run():
        out["res"] = server.make_request("POST", "/completion", data=data)

    t = threading.Thread(target=run)
    t.start()
    return t, out


def wait_fast(what, cond, seconds=10.0):
    t_end = time.time() + seconds
    while time.time() < t_end:
        if cond():
            return
        time.sleep(0.002)
    raise TimeoutError(what)


def test_a_conversation_goes_on_in_the_copy_that_read_it():
    global server
    server.model_copy_keep_conversations = True
    server.model_copy_after = 1
    server.model_copy_ctx_size = 2048
    server.model_copy_release = 60
    server.start()

    n_story = n_tokens(STORY)
    assert n_story > 80

    # the first request of the conversation arrives while the first model works on a task, so it goes to the copy
    out, first = while_a_task_runs(LONG_TASK, lambda: complete(n_predict=16, prompt=STORY, cache_prompt=True))
    assert out["res"].status_code == 200
    copy = the_copy()
    assert copy["n_tasks_total"] == 1
    assert copy["n_tasks_back"] == 0
    assert copy["tasks_first"] == 0

    # the first model is free now. The conversation goes on in the copy all the same, which has its prompt
    second = complete(n_predict=16, prompt=STORY + first["content"] + " And then the cat", cache_prompt=True)
    copy = the_copy()
    assert copy["n_tasks_total"] == 2
    assert copy["n_tasks_back"] == 1
    assert second["timings"]["cache_n"] >= n_story - 1

    # another conversation stays with the first model, which is free
    complete(n_predict=16, prompt="The sun was hot and the sea was blue, and the two children ran down to the water. " * 4)
    copy = the_copy()
    assert copy["n_tasks_total"] == 2
    assert copy["n_tasks_back"] == 1


def test_a_conversation_that_the_first_model_read_last_stays_there():
    global server
    server.model_copy_keep_conversations = True
    server.model_copy_after = 1
    server.model_copy_ctx_size = 2048
    server.model_copy_release = 60
    server.start()

    n_story = n_tokens(STORY)

    # the conversation starts in the copy
    out, first = while_a_task_runs(LONG_TASK, lambda: complete(n_predict=16, prompt=STORY, cache_prompt=True))
    assert the_copy()["n_tasks_total"] == 1

    # its next request finds the copy's one slot at work on another task, so the first model reads it
    t_a, out_a = in_background(LONG_TASK)
    try:
        wait_fast("the first model got no task", lambda: the_copy()["tasks_first"] == 1)
        t_b, out_b = in_background({"prompt": "The sun was hot and the sea was blue. " * 3, "n_predict": 1900,
                                    "ignore_eos": True, "cache_prompt": False})
        try:
            wait_fast("the other task did not reach the copy", lambda: the_copy()["tasks"] == 1)
            second = complete(n_predict=16, prompt=STORY + first["content"] + " And then the cat", cache_prompt=True)
            seen = the_copy()
        finally:
            t_b.join()
    finally:
        t_a.join()
    assert out_a["res"].status_code == 200
    assert out_b["res"].status_code == 200
    # the copy still had the other task when the request was answered, so the request was not there
    assert seen["tasks"] == 1
    copy = the_copy()
    assert copy["n_tasks_total"] == 2
    assert copy["n_tasks_back"] == 0

    # the first model has the conversation now, and both models are free: the next request stays with the first one
    third = complete(n_predict=16, cache_prompt=True,
                     prompt=STORY + first["content"] + " And then the cat" + second["content"] + " said")
    copy = the_copy()
    assert copy["n_tasks_total"] == 2
    assert copy["n_tasks_back"] == 0
    assert third["timings"]["cache_n"] >= n_story - 1


# The copy takes the tasks that the first pool of KV cells has no room for, and it gives way to a task that is too
# large for that pool and for the copy: such a task needs the second pool of the first model, on the copy's device.

def test_a_task_that_would_spill_out_of_the_first_pool_goes_to_the_copy():
    global server
    server.n_ctx = 1024            # the first pool
    server.model_copy_after = 5    # far away: here only the room in the first pool sends a task to the copy
    server.model_copy_ctx_size = 1024
    server.model_copy_release = 60
    server.start()

    # alone, a task stays with the first model
    assert complete(n_predict=200)["tokens_predicted"] == 200
    assert the_copy()["n_tasks_total"] == 0

    # the first model works on a task that takes most of its pool: the next one does not fit beside it
    t, out = in_background({"prompt": PROMPT, "n_predict": 900, "ignore_eos": True, "cache_prompt": False})
    try:
        wait_fast("the first model got no task", lambda: the_copy()["tasks_first"] == 1)
        body = complete(n_predict=200)
    finally:
        t.join()
    assert body["tokens_predicted"] == 200
    assert out["res"].status_code == 200
    assert out["res"].body["tokens_predicted"] == 900
    copy = the_copy()
    assert copy["n_tasks_total"] == 1
    assert copy["n_tasks_spill"] == 1
    assert copy["n_home"] == 1024


@needs_device
def test_the_copy_gives_way_to_a_task_that_only_the_second_pool_can_hold():
    global server
    server.n_ctx = 1024
    server.n_slots = 3
    server.kv_overflow_device = DEVICE
    server.kv_overflow_size = 4096
    server.kv_overflow_release = 60
    server.kv_overflow_headroom = 64
    server.model_copy_ctx_size = 1024
    server.model_copy_after = 0
    server.model_copy_release = 60
    server.model_copy_headroom = 64
    server.extra_args = ["--kv-unified-per-slot", "256,2048,2048", "--batch-priority-slots", "0"]
    server.start()

    # a task loads the copy, which may stay for 60 s
    assert complete(n_predict=16)["tokens_predicted"] == 16
    assert the_copy()["state"] == "loaded"

    # the copy works on a long task. Behind it comes a task that neither the first pool can hold (768 cells for it)
    # nor the copy (1024): it waits until the copy has ended its task and is unloaded, and runs in the second pool
    t_long, out_long = in_background({"prompt": PROMPT, "n_predict": 990, "ignore_eos": True, "cache_prompt": False})
    try:
        wait_fast("the long task did not reach the copy", lambda: the_copy()["tasks"] == 1)
        # (it names a coding slot: slot 0 comes first, and its limit is 256)
        large = complete(n_predict=1500, ignore_eos=True, id_slot=1)
        seen = the_copy()
    finally:
        t_long.join()
    assert large["tokens_predicted"] == 1500
    assert out_long["res"].status_code == 200
    assert out_long["res"].body["tokens_predicted"] == 990
    assert seen["state"] == "unloaded"
    assert seen["tasks"] == 0
    copy = the_copy()
    assert copy["n_home"] == 768
    assert copy["n_loads"] == 1
    assert copy["n_tasks_total"] == 2
    assert copy["tasks_pool"] == 0


# The copy has a prompt cache of its own, with its own limit: the first model's --cache-ram is not there twice.

def test_the_copys_prompt_cache_has_a_limit_of_its_own(tmp_path):
    global server
    server.cache_ram = 2048
    server.log_path = str(tmp_path / "server.log")
    server.debug = True
    server.start()
    assert the_copy()["cache_ram"] == 8192   # the default, whatever --cache-ram says
    assert complete()["tokens_predicted"] == 16
    server.stop()
    with open(server.log_path) as f:
        log = f.read()
    assert "copy srv    load_model: prompt cache is enabled, size limit: 8192 MiB" in log

    server.model_copy_cache_ram = 64
    server.log_path = str(tmp_path / "server2.log")
    server.start()
    assert the_copy()["cache_ram"] == 64
    assert complete()["tokens_predicted"] == 16
    server.stop()
    with open(server.log_path) as f:
        log = f.read()
    assert "copy srv    load_model: prompt cache is enabled, size limit: 64 MiB" in log
    assert "srv    load_model: prompt cache is enabled, size limit: 2048 MiB" in log


# What the copy's device has free is asked before the copy is loaded, and every two seconds while it is loaded.
# LLAMA_SERVER_DEBUG_MODEL_COPY_FREE names a file that holds the reading in MiB, in place of the device's own: a test
# writes there what another program would leave of the device.

@pytest.fixture
def device_free(tmp_path, monkeypatch):
    """Sets what the device of the copy says it has free, in MiB."""
    path = tmp_path / "device-free.txt"

    def set_free(mib: int):
        # written whole: the server reads it at any moment
        tmp = tmp_path / "device-free.tmp"
        tmp.write_text(str(mib))
        os.replace(tmp, path)

    set_free(1024 * 1024)
    monkeypatch.setenv("LLAMA_SERVER_DEBUG_MODEL_COPY_FREE", str(path))
    return set_free


@needs_device
def test_the_copy_gives_way_when_another_program_takes_its_devices_memory(device_free):
    global server
    server.model_copy_headroom = 1024
    server.model_copy_release = 60
    server.start()

    assert complete()["tokens_predicted"] == 16
    copy = the_copy()
    assert copy["state"] == "loaded"
    assert copy["gives_way"] is False

    # another program takes the device: 300 MiB are left, less than half the headroom. The copy is idle, so it goes
    # at once and does not wait out its 60 seconds
    device_free(300)
    wait_until("the copy did not give way", lambda: the_copy()["state"] == "unloaded", seconds=8.0)

    # and it is not loaded again while the device is short
    assert complete()["tokens_predicted"] == 16
    copy = the_copy()
    assert copy["state"] == "unloaded"
    assert copy["n_loads"] == 1
    assert copy["n_tasks_total"] == 1
    assert copy["n_refused"] == 1

    # the other program is gone
    device_free(1024 * 1024)
    assert complete()["tokens_predicted"] == 16
    copy = the_copy()
    assert copy["state"] == "loaded"
    assert copy["n_loads"] == 2
    assert copy["n_tasks_total"] == 2


def test_a_task_too_large_for_the_first_pool_waits_for_the_copys_room():
    global server
    server.n_ctx = 1024            # the first pool
    server.model_copy_after = 5
    server.model_copy_ctx_size = 2048
    server.model_copy_release = 60
    server.start()

    # a task of 1500 tokens does not fit in the first pool; the copy has 2048 cells and one slot
    t, out = in_background({"prompt": PROMPT, "n_predict": 1450, "ignore_eos": True, "cache_prompt": False})
    try:
        wait_fast("the task did not reach the copy", lambda: the_copy()["tasks"] == 1)
        # a second one of that size arrives while the copy works on the first: it waits for the copy's slot, and is
        # not cut short in the first pool
        body = complete(n_predict=1400, ignore_eos=True)
    finally:
        t.join()
    assert out["res"].status_code == 200
    assert out["res"].body["tokens_predicted"] == 1450
    assert body["tokens_predicted"] == 1400
    copy = the_copy()
    assert copy["n_tasks_total"] == 2
    assert copy["n_tasks_waited"] == 1
    assert copy["tasks"] == 0
    assert copy["tasks_first"] == 0


# The second pool of KV cells of the first model and the copy are on the same device here, and both take their memory
# when they find it free. A copy that is being loaded has claimed its memory: the second pool does not take it.

@needs_device
def test_the_second_pool_leaves_the_memory_that_a_loading_copy_has_claimed(tmp_path, monkeypatch):
    import re

    global server
    pool_free = tmp_path / "pool-free.txt"
    copy_free = tmp_path / "copy-free.txt"

    def write(path, mib):
        tmp = tmp_path / (path.name + ".tmp")
        tmp.write_text(str(mib))
        os.replace(tmp, path)

    write(pool_free, 1024 * 1024)
    write(copy_free, 1024 * 1024)
    monkeypatch.setenv("LLAMA_SERVER_DEBUG_KV_OVERFLOW_FREE", str(pool_free))
    monkeypatch.setenv("LLAMA_SERVER_DEBUG_MODEL_COPY_FREE", str(copy_free))
    # the copy's load takes a second and a half longer, as a large model's does
    monkeypatch.setenv("LLAMA_SERVER_DEBUG_MODEL_COPY_LOAD_MS", "1500")

    server.n_ctx = 2048
    server.n_slots = 3
    server.kv_overflow_device = DEVICE
    server.kv_overflow_size = 1024
    server.kv_overflow_release = 60
    server.kv_overflow_headroom = 500
    server.model_copy_ctx_size = 1024
    server.model_copy_after = 1
    server.model_copy_release = 60
    server.model_copy_headroom = 64
    server.log_path = str(tmp_path / "server.log")
    server.start()

    def log():
        with open(server.log_path) as f:
            return f.read()

    # the device is made to have just what a task in the second pool needs: the pool's cells, the headroom of 500 MiB,
    # and half a MiB. The copy takes more than that half
    pool_mib = float(re.search(r"on top of the ([0-9.]+) MiB of the pool's cells", log()).group(1))
    write(pool_free, 500 + pool_mib + 0.5)

    # the first model works on a long task, so the next task goes to the copy, whose load begins
    t_b, out_b = in_background({"prompt": PROMPT, "n_predict": 1500, "ignore_eos": True, "cache_prompt": False})
    try:
        wait_fast("the first model got no task", lambda: the_copy()["tasks_first"] == 1)
        t_a, out_a = in_background({"prompt": PROMPT, "n_predict": 500, "ignore_eos": True, "cache_prompt": False})
        try:
            wait_fast("the copy did not start to load", lambda: the_copy()["state"] == "loading")
            assert the_copy()["need_mib"][DEVICE] > 0.5
            # a third task fits neither beside the long one in the first pool nor beside the copy's task. The first
            # model would give it the second pool, whose device has the memory by its own figure. The copy has
            # claimed it, so the task waits for the first pool
            third = complete(n_predict=600, ignore_eos=True)
        finally:
            t_a.join()
    finally:
        t_b.join()

    assert third["tokens_predicted"] == 600
    assert out_a["res"].status_code == 200
    assert out_a["res"].body["tokens_predicted"] == 500
    assert out_b["res"].status_code == 200
    assert out_b["res"].body["tokens_predicted"] == 1500
    copy = the_copy()
    assert copy["n_loads"] == 1
    assert copy["n_tasks_total"] == 1
    text = log()
    assert "the second pool of KV cells gets no task for now" in text
    assert "the second pool of KV cells took its memory" not in text


def test_the_first_model_comes_first_for_a_conversation_of_the_copy():
    global server
    server.model_copy_after = 1
    server.model_copy_ctx_size = 2048
    server.model_copy_release = 60
    server.start()
    assert the_copy()["keep_conversations"] is False

    n_story = n_tokens(STORY)

    # the first request of the conversation arrives while the first model works on a task, so it goes to the copy
    out, first = while_a_task_runs(LONG_TASK, lambda: complete(n_predict=16, prompt=STORY, cache_prompt=True))
    assert out["res"].status_code == 200
    assert the_copy()["n_tasks_total"] == 1

    # the first model is free now, and it comes first: the conversation goes on there, where it is read again
    second = complete(n_predict=16, prompt=STORY + first["content"] + " And then the cat", cache_prompt=True)
    copy = the_copy()
    assert copy["n_tasks_total"] == 1
    assert copy["n_tasks_back"] == 0
    assert copy["state"] == "loaded"
    # read again there, but for its first sentence: that is the long task's prompt, which the slot still had
    assert second["timings"]["prompt_n"] >= n_story - n_tokens(PROMPT) - 1


# The slots of the copy count as slots of the server: /props and /slots have them after those of the first model. A
# proxy in front that lets in one request for each slot so lets in as many as both models have slots for. The server
# places the tasks itself then: the id of a slot is binding only for a slot that comes first.

def test_the_copys_slots_count_as_slots_of_the_server():
    global server
    server.model_copy_parallel = 3
    server.model_copy_release = 60
    server.start()

    res = server.make_request("GET", "/props")
    assert res.body["total_slots"] == 2 + 3

    def slots():
        res = server.make_request("GET", "/slots")
        assert res.status_code == 200
        return res.body

    # not loaded yet: its slots are there, idle
    body = slots()
    assert [slot["id"] for slot in body] == [0, 1, 2, 3, 4]
    assert [bool(slot.get("model_copy")) for slot in body] == [False, False, True, True, True]
    assert not any(slot["is_processing"] for slot in body)

    # a task in the copy shows in one of its slots (every task goes to the copy here, --model-copy-after 0)
    t, out = in_background({"prompt": PROMPT, "n_predict": 900, "ignore_eos": True, "cache_prompt": False})
    try:
        seen = {}

        def a_copy_slot_works():
            body = slots()
            seen["ids"] = [slot["id"] for slot in body]
            seen["busy"] = [slot["id"] for slot in body if slot["is_processing"]]
            return len(seen["busy"]) > 0

        wait_fast("no slot started processing", a_copy_slot_works)
    finally:
        t.join()
    assert out["res"].status_code == 200
    assert seen["ids"] == [0, 1, 2, 3, 4]
    assert len(seen["busy"]) == 1 and seen["busy"][0] >= 2
    assert the_copy()["n_tasks_total"] == 1


def test_with_a_copy_only_the_id_of_a_slot_that_comes_first_is_binding():
    global server
    server.n_slots = 3
    server.n_ctx = 8192
    server.model_copy_after = 5       # far away: the tasks stay with the first model here
    server.model_copy_release = 60
    server.extra_args = ["--kv-unified-per-slot", "256,2048,2048", "--batch-priority-slots", "0"]
    server.start()

    # a task without a slot id never takes the slot that comes first, though that one is empty and unused
    seen = set()
    for i in range(6):
        seen.add(complete(n_predict=8, prompt="Story %d. " % i + PROMPT)["id_slot"])
    assert seen <= {1, 2}

    # the slot that comes first is given to the task that asks for it
    assert complete(n_predict=8, id_slot=0)["id_slot"] == 0

    # the id of another slot is not binding: a task with the id of a slot that is at work does not wait for that slot
    t, out = in_background({"prompt": PROMPT, "n_predict": 1900, "ignore_eos": True, "cache_prompt": False, "id_slot": 1})
    try:
        wait_fast("the first model got no task", lambda: the_copy()["tasks_first"] == 1)
        body = complete(n_predict=8, id_slot=1)
        long_one_at_work = the_copy()["tasks_first"] == 1
    finally:
        t.join()
    assert out["res"].status_code == 200
    assert out["res"].body["tokens_predicted"] == 1900
    assert long_one_at_work
    assert body["id_slot"] in (1, 2)
    assert body["id_slot"] != out["res"].body["id_slot"]
    assert the_copy()["n_tasks_total"] == 0
