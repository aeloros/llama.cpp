import threading
import time

import pytest
from utils import *

server = ServerPreset.tinyllama2()

# With --kv-reserve a task starts only when its size, its prompt and its output limit, fits in the pool beside the
# sizes of the tasks at work. The slots hold 2048 each, the model's training context, and the pool 2560.
N_CTX = 2560

PROMPT_OLDER = "Once upon a time there was a little cat who lived in a big house."
PROMPT_NEWER = "One day a boy and his dog went to the park to play with a red ball."


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.kv_unified = True
    server.kv_reserve = True
    server.n_ctx = N_CTX
    server.n_slots = 2
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


def run_two(n_predict_older: int, n_predict_newer: int):
    """The older task in slot 0, the newer one in slot 1 as soon as the older is at work.
    Returns both responses and the largest number of slots seen at work at once."""
    results = {}

    def run(key: str, prompt, n_predict: int, id_slot: int):
        results[key] = complete(prompt, n_predict, id_slot)

    def n_busy() -> int:
        res = server.make_request("GET", "/slots")
        return sum(1 for slot in res.body if slot["is_processing"]) if res.status_code == 200 else 0

    t_older = threading.Thread(target=run, args=("older", PROMPT_OLDER, n_predict_older, 0))
    t_older.start()
    t_end = time.time() + 10
    while n_busy() == 0 and time.time() < t_end:
        time.sleep(0.002)
    t_newer = threading.Thread(target=run, args=("newer", PROMPT_NEWER, n_predict_newer, 1))
    t_newer.start()

    n_busy_max = 0
    while t_older.is_alive() or t_newer.is_alive():
        n_busy_max = max(n_busy_max, n_busy())
        time.sleep(0.002)
    t_older.join()
    t_newer.join()

    return results["older"], results["newer"], n_busy_max


def test_a_task_that_does_not_fit_waits_and_no_task_fails():
    # 2 * (prompt + 1800 + 1) > 2560: the newer task starts when the older one is done
    global server
    server.start()

    older, newer, n_busy_max = run_two(1800, 1800)

    assert n_busy_max == 1, "the two tasks were at work at once"
    for res in (older, newer):
        assert res.status_code == 200, f"{res.status_code} {str(res.body)[:200]}"
        assert res.body["tokens_predicted"] == 1800
        assert not res.body["truncated"]


def test_tasks_that_fit_together_run_together():
    # (prompt + 1800 + 1) + (prompt + 600 + 1) <= 2560
    global server
    server.start()

    older, newer, n_busy_max = run_two(1800, 600)

    assert n_busy_max == 2, "the newer task waited though it fits"
    assert older.status_code == 200 and older.body["tokens_predicted"] == 1800
    assert newer.status_code == 200 and newer.body["tokens_predicted"] == 600
