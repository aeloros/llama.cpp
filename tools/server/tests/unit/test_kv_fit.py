import os
import threading
import time

import pytest
from utils import *

# A second pool of KV cells on another device (--kv-overflow-device, --kv-overflow-size) without a list of slots for
# it: a task is fitted when it starts. It goes to the first pool, the home pool, when it fits there, and only then
# whole to the second pool, and it waits when neither has the room.
#
# The second pool needs a device: set LLAMA_TEST_KV_OVERFLOW_DEVICE to one that the server lists, e.g. CUDA0.
DEVICE = os.environ.get("LLAMA_TEST_KV_OVERFLOW_DEVICE")

pytestmark = pytest.mark.skipif(not DEVICE, reason="LLAMA_TEST_KV_OVERFLOW_DEVICE names no device for the second pool")

server = ServerPreset.tinyllama2()

N_HOME = 2560       # cells of the home pool; a slot holds 2048, the model's training context
N_OVERFLOW = 2560   # cells of the second pool

PROMPT_OLDER = "Once upon a time there was a little cat who lived in a big house."
PROMPT_NEWER = "One day a boy and his dog went to the park to play with a red ball."


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.kv_unified = True
    server.n_ctx = N_HOME
    server.kv_overflow_device = DEVICE
    server.kv_overflow_size = N_OVERFLOW
    server.n_slots = 3
    server.server_slots = True
    server.temperature = 0.0


def complete(prompt, n_predict: int, id_slot: int):
    return server.make_request("POST", "/completion", data={
        "prompt": prompt,
        "n_predict": n_predict,
        "id_slot": id_slot,
        "ignore_eos": True,
        "temperature": 0.0,
        "seed": 42,
        "cache_prompt": False,
    })


def n_tokens(text: str) -> int:
    res = server.make_request("POST", "/tokenize", data={"content": text, "add_special": True})
    assert res.status_code == 200
    return len(res.body["tokens"])


def slots_at_work():
    res = server.make_request("GET", "/slots")
    return [slot for slot in res.body if slot["is_processing"]] if res.status_code == 200 else []


def run_two(n_predict_older: int, n_predict_newer: int):
    """The older task in slot 0, the newer one in slot 1 as soon as the older is at work. Returns both responses,
    the largest number of slots seen at work at once, and what /slots said of each slot while it was at work."""
    results = {}
    seen = {}

    def run(key: str, prompt, n_predict: int, id_slot: int):
        results[key] = complete(prompt, n_predict, id_slot)

    t_older = threading.Thread(target=run, args=("older", PROMPT_OLDER, n_predict_older, 0))
    t_older.start()
    t_end = time.time() + 10
    while not slots_at_work() and time.time() < t_end:
        time.sleep(0.002)
    t_newer = threading.Thread(target=run, args=("newer", PROMPT_NEWER, n_predict_newer, 1))
    t_newer.start()

    n_busy_max = 0
    while t_older.is_alive() or t_newer.is_alive():
        busy = slots_at_work()
        n_busy_max = max(n_busy_max, len(busy))
        for slot in busy:
            seen[slot["id"]] = slot
        time.sleep(0.002)
    t_older.join()
    t_newer.join()

    return results["older"], results["newer"], n_busy_max, seen


def assert_served(res, n_predict: int):
    assert res.status_code == 200, f"{res.status_code} {str(res.body)[:200]}"
    assert res.body["tokens_predicted"] == n_predict
    assert not res.body["truncated"]


def test_two_tasks_that_fit_at_home_are_both_at_home():
    global server
    server.start()

    older, newer, n_busy_max, seen = run_two(1000, 1000)

    assert n_busy_max == 2
    assert_served(older, 1000)
    assert_served(newer, 1000)
    assert not seen[0]["kv_overflow"]
    assert not seen[1]["kv_overflow"], "the second task fits at home beside the first: 2 * (prompt + 1000 + 1) <= 2560"


def test_a_task_that_does_not_fit_at_home_goes_whole_to_the_second_pool():
    # 2 * (prompt + 1800 + 1) > 2560: where a full pool stops a task, and --kv-reserve makes it wait, the second
    # pool takes it, and both run at once
    global server
    server.start()

    older, newer, n_busy_max, seen = run_two(1800, 1800)

    assert n_busy_max == 2
    assert_served(older, 1800)
    assert_served(newer, 1800)
    assert not seen[0]["kv_overflow"]
    assert seen[1]["kv_overflow"]
    assert seen[1]["n_ctx_task"] == n_tokens(PROMPT_NEWER) + 1800 + 1


def test_a_task_at_home_takes_its_room_as_it_goes():
    # with a headroom of 256 a task starts with its prompt and 256 tokens of room, and takes 256 more each time it
    # runs into that: alone in the pool it gets to its end
    global server
    server.kv_fit_headroom = 256
    server.start()

    seen = {}
    out = {}

    def run():
        out["res"] = complete(PROMPT_OLDER, 1800, 0)

    t = threading.Thread(target=run)
    t.start()
    while t.is_alive():
        for slot in slots_at_work():
            seen.setdefault("first", slot)
            seen["last"] = slot
        time.sleep(0.002)
    t.join()

    assert_served(out["res"], 1800)
    assert not seen["first"]["kv_overflow"]
    assert seen["first"]["n_ctx_task"] < n_tokens(PROMPT_OLDER) + 1800 + 1, "it started with less than its whole size"
    assert seen["last"]["n_ctx_task"] > seen["first"]["n_ctx_task"]


def test_home_goes_first_for_tasks_whose_prompt_and_a_headroom_fit_there():
    # with a headroom of 256 both tasks go home though their sizes together are more than the pool: each starts
    # with its prompt and 256 tokens of room. When the pool has no more, the newer task gives up the room it has
    # not used and ends as truncated, and the older one goes on to its end
    global server
    server.kv_fit_headroom = 256
    server.start()

    older, newer, n_busy_max, seen = run_two(1800, 1800)

    assert n_busy_max == 2
    assert not seen[0]["kv_overflow"]
    assert not seen[1]["kv_overflow"]
    assert_served(older, 1800)
    assert newer.status_code == 200, f"{newer.status_code} {str(newer.body)[:200]}"
    assert newer.body["truncated"]
    assert 256 < newer.body["tokens_predicted"] < 1800


def test_a_task_that_fits_in_neither_pool_waits():
    # a third task of the same size: the home pool and the second pool hold one each
    global server
    server.start()

    results = {}

    def run(key: str, prompt, id_slot: int):
        results[key] = complete(prompt, 1800, id_slot)

    threads = []
    for i, prompt in enumerate((PROMPT_OLDER, PROMPT_NEWER, PROMPT_OLDER + " The end.")):
        t = threading.Thread(target=run, args=(i, prompt, i))
        t.start()
        threads.append(t)
        t_end = time.time() + 10
        while len(slots_at_work()) < min(i + 1, 2) and time.time() < t_end:
            time.sleep(0.002)

    n_busy_max = 0
    while any(t.is_alive() for t in threads):
        n_busy_max = max(n_busy_max, len(slots_at_work()))
        time.sleep(0.002)

    assert n_busy_max == 2, "three tasks of 1800 were at work at once in two pools of 2560"
    for i in range(3):
        assert_served(results[i], 1800)
