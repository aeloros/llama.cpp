import threading
import time

import pytest
from utils import *

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


def slot_while_running(data):
    """The busy slot, as /slots reports it while the request runs, and the request's response."""
    out = {}

    def run():
        out["res"] = server.make_request("POST", "/completion", data=data)

    t = threading.Thread(target=run)
    t.start()
    try:
        t_end = time.time() + 10
        while time.time() < t_end:
            res = server.make_request("GET", "/slots")
            busy = [slot for slot in res.body if slot["is_processing"]] if res.status_code == 200 else []
            if busy:
                return busy[0], out
            time.sleep(0.002)
        raise TimeoutError("no slot started processing")
    finally:
        t.join()


def test_a_task_with_an_output_limit_has_a_size_of_its_own():
    # the slot may hold 2048 tokens, the model's training context. The task can only grow to its prompt and its
    # output, and one more: the token it stops at is not counted against it
    global server
    server.start()

    res = server.make_request("POST", "/tokenize", data={"content": PROMPT, "add_special": True})
    n_prompt = len(res.body["tokens"])

    slot, out = slot_while_running({"prompt": PROMPT, "n_predict": 1500, "ignore_eos": True, "cache_prompt": False})
    assert slot["n_ctx"] == 2048
    assert slot["n_ctx_task"] == n_prompt + 1500 + 1

    res = out["res"]
    assert res.status_code == 200
    assert res.body["tokens_predicted"] == 1500
    assert not res.body["truncated"]


def test_a_task_without_an_output_limit_has_the_size_of_its_slot():
    global server
    server.n_predict = -1
    server.start()

    slot, out = slot_while_running({"prompt": PROMPT, "n_predict": -1, "ignore_eos": True, "cache_prompt": False})
    assert slot["n_ctx"] == 2048
    assert slot["n_ctx_task"] == 2048

    # it runs until the slot is full
    res = out["res"]
    assert res.status_code == 200
    assert res.body["truncated"]
