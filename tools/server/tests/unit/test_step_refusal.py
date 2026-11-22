import threading
import time

import pytest
from utils import *

server = ServerPreset.tinyllama2()

# Tasks grow in one pool of cells (--kv-unified) until it is full. Each of them fits in the pool by itself, and in
# its slot, which the model's training context limits to 2048.
N_CTX = 2560
N_PREDICT = 1800

PROMPT_OLDER = "Once upon a time there was a little cat who lived in a big house."
PROMPT_NEWER = "One day a boy and his dog went to the park to play with a red ball."


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.kv_unified = True
    server.n_ctx = N_CTX
    server.n_slots = 2
    server.server_slots = True
    server.temperature = 0.0


def complete(prompt, n_predict: int = N_PREDICT, id_slot: int = -1, **more):
    return server.make_request("POST", "/completion", data={
        "prompt": prompt,
        "n_predict": n_predict,
        "id_slot": id_slot,
        "ignore_eos": True,
        "temperature": 0.0,
        "seed": 42,
        "cache_prompt": False,
        **more,
    })


def tokenize(text: str):
    res = server.make_request("POST", "/tokenize", data={"content": text, "add_special": True})
    assert res.status_code == 200
    return res.body["tokens"]


def wait_for_slots(ready, timeout: float = 10.0):
    t_end = time.time() + timeout
    while time.time() < t_end:
        res = server.make_request("GET", "/slots")
        if res.status_code == 200 and ready(res.body):
            return
        time.sleep(0.002)
    raise TimeoutError("the slots did not get to the expected state")


def n_decoded(slot) -> int:
    return slot["next_token"][0]["n_decoded"] if slot["is_processing"] else 0


def run_older_and_newer(newer_prompt, newer_n_predict: int, start_newer):
    # the older task in slot 0, the newer one in slot 1, started when start_newer(slots) holds
    results = {}

    def run(key: str, prompt, n_predict: int, id_slot: int):
        results[key] = complete(prompt, n_predict, id_slot, return_tokens=True)

    t_older = threading.Thread(target=run, args=("older", PROMPT_OLDER, N_PREDICT, 0))
    t_older.start()
    wait_for_slots(start_newer)
    t_newer = threading.Thread(target=run, args=("newer", newer_prompt, newer_n_predict, 1))
    t_newer.start()
    t_older.join()
    t_newer.join()

    return results["older"], results["newer"]


def assert_older_went_on_alone(older, newer):
    assert newer.status_code == 400, f"the newer task: {newer.status_code} {str(newer.body)[:200]}"
    assert newer.body["error"]["type"] == "exceed_context_size_error"

    assert older.status_code == 200, f"the older task: {older.status_code} {str(older.body)[:200]}"
    assert older.body["tokens_predicted"] == N_PREDICT
    assert not older.body["truncated"]

    # A refused step leaves the other task as it was: what its slot holds is still its prompt and its text, token for
    # token, so a request for the token after them reads one token and no more.
    # (The text and the probabilities are no measure of it. Both change with what else shares a batch, by rounding:
    # with no step refused at all, a log-probability from a slot and from a fresh read of the same tokens differ by
    # up to 0.009 on the CPU and 0.13 with CUDA, on this model.)
    tokens = tokenize(PROMPT_OLDER) + older.body["tokens"]
    held = complete(tokens, 1, 0, cache_prompt=True)
    assert held.status_code == 200
    assert held.body["timings"]["prompt_n"] == 1, "the slot holds every token but the last, which was never decoded"


def test_refusal_fails_the_newest_task_alone():
    # the pool cannot hold both tasks to their end: 2 * (prompt + 1800) > 2560
    # the step that does not fit fails the task that came last, with the context-size error,
    # and the older task goes on as if nothing had happened
    global server
    server.start()

    older, newer = run_older_and_newer(PROMPT_NEWER, N_PREDICT,
            start_newer=lambda slots: any(slot["is_processing"] for slot in slots))

    assert_older_went_on_alone(older, newer)


def test_refusal_while_a_prompt_is_read():
    # the newer task comes with a prompt of 1900 tokens when the older one holds over 1000 cells of the 2560:
    # the prompt is read in smaller and smaller steps until no token of it fits, then the newer task fails alone
    global server
    server.start()

    tokens = tokenize(PROMPT_NEWER)
    prompt = (tokens * (1900 // len(tokens) + 1))[:1900]

    older, newer = run_older_and_newer(prompt, 8,
            start_newer=lambda slots: any(n_decoded(slot) >= 1000 for slot in slots))

    assert_older_went_on_alone(older, newer)


def test_refusal_gives_up_an_idle_cache_before_a_task():
    # the first task is done and its slot keeps its 1500 tokens: with --no-cache-idle-slots an idle slot is not
    # cleared when another task starts. The second task, in the other slot, grows until the pool is full: the idle
    # slot's cached prompt goes, and no task fails
    global server
    server.no_cache_idle_slots = True
    server.start()

    first = complete(PROMPT_OLDER, 1480, id_slot=0)
    assert first.status_code == 200
    assert first.body["tokens_predicted"] == 1480

    second = complete(PROMPT_NEWER, 1480, id_slot=1)
    assert second.status_code == 200, f"the second task: {second.status_code} {str(second.body)[:200]}"
    assert second.body["tokens_predicted"] == 1480
    assert not second.body["truncated"]
