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


# An idle slot keeps the cells of its cached prompt (--no-cache-idle-slots) until a task needs them. Then it gives
# them up, saved to the prompt cache, and its conversation comes back from there when it goes on.


def turn(prompt, n_predict: int, id_slot: int):
    """One turn of a conversation: the prompt is cached, and the tokens of the answer come back with it."""
    return server.make_request("POST", "/completion", data={
        "prompt": prompt,
        "n_predict": n_predict,
        "id_slot": id_slot,
        "ignore_eos": True,
        "temperature": 0.0,
        "seed": 42,
        "cache_prompt": True,
        "return_tokens": True,
    })


def tokens_of(text: str):
    res = server.make_request("POST", "/tokenize", data={"content": text, "add_special": True})
    assert res.status_code == 200
    return res.body["tokens"]


def assert_not_read_again(res):
    # the slot holds every token of the conversation but the last, which was never decoded: one token is read
    assert res.status_code == 200, f"{res.status_code} {str(res.body)[:200]}"
    assert res.body["timings"]["prompt_n"] == 1, "the conversation is read again"


def test_an_idle_slot_gives_up_its_cells_to_a_task_and_its_conversation_comes_back():
    global server
    server.no_cache_idle_slots = True
    server.start()

    # a conversation in slot 0: 1500 cells of the home pool, then idle
    first = turn(PROMPT_OLDER, 1480, 0)
    assert_served(first, 1480)
    held = tokens_of(PROMPT_OLDER) + first.body["tokens"]

    # a task in slot 1 that fits at home by the limits of the tasks at work, and not beside those 1500 cells
    second = turn(PROMPT_NEWER, 1480, 1)
    assert_served(second, 1480)

    assert_not_read_again(turn(held, 8, 0))


def test_a_limit_that_grows_takes_the_cells_of_an_idle_slot_and_saves_them():
    global server
    server.no_cache_idle_slots = True
    server.kv_fit_headroom = 256
    server.start()

    first = turn(PROMPT_OLDER, 1480, 0)
    assert_served(first, 1480)
    held = tokens_of(PROMPT_OLDER) + first.body["tokens"]

    # the task starts with its prompt and 256 tokens of room, which the pool has beside the idle slot's 1500 cells;
    # it grows by 256 at a time, and at 1300 the pool no longer has both
    second = turn(PROMPT_NEWER, 1480, 1)
    assert_served(second, 1480)

    assert_not_read_again(turn(held, 8, 0))


def test_a_conversation_whose_slot_changes_pools_is_not_read_again():
    global server
    server.no_cache_idle_slots = True
    server.start()

    # a conversation in slot 0: 500 cells of the home pool, then idle
    first = turn(PROMPT_OLDER, 480, 0)
    assert_served(first, 480)
    held = tokens_of(PROMPT_OLDER) + first.body["tokens"]

    # a task of 1900 tokens at work at home leaves the idle cells where they are: 1900 + 500 <= 2560. The next turn
    # of the conversation, 500 + 300 tokens, does not fit beside its limit (2560 - 1900 = 660), so it goes to the
    # second pool, and the slot with it
    results = {}
    seen = {}

    def run_long():
        results["long"] = complete(PROMPT_NEWER, 1880, 1)

    def run_again():
        results["again"] = turn(held, 300, 0)

    t_long = threading.Thread(target=run_long)
    t_long.start()
    t_end = time.time() + 10
    while not slots_at_work() and time.time() < t_end:
        time.sleep(0.002)
    t_again = threading.Thread(target=run_again)
    t_again.start()
    while t_again.is_alive():
        for slot in slots_at_work():
            seen[slot["id"]] = slot
        time.sleep(0.002)
    t_again.join()
    t_long.join()

    assert_served(results["long"], 1880)
    assert seen[0]["kv_overflow"], "the next turn went to the second pool"
    assert not seen[1]["kv_overflow"]
    assert_not_read_again(results["again"])
    assert results["again"].body["tokens_predicted"] == 300


# The second pool has its memory only while it is in use (--kv-overflow-release): it is taken from the device when a
# task goes there, and it goes back when the pool has been without a task for that many seconds. The server's log
# says so, and the tests read it there: a request to the server would run a step of its loop, and the memory has to
# go back without one.

LOG_ON_DEMAND = "the second pool takes its memory from"
LOG_FROM_THE_START = "the second pool has its memory on"
LOG_TOOK = "the second pool of KV cells took its memory on"
LOG_GAVE_BACK = "the second pool of KV cells gave its memory back to"
LOG_NO_MEMORY = "the second pool of KV cells did not get its memory on"


def log_text() -> str:
    with open(server.log_path) as f:
        return f.read()


def wait_for_log(text: str, n: int, timeout: float = 10.0) -> bool:
    """Waits until the server's log has the text n times, without a request to the server."""
    t_end = time.time() + timeout
    while time.time() < t_end:
        if log_text().count(text) >= n:
            return True
        time.sleep(0.05)
    return False


def slots():
    res = server.make_request("GET", "/slots")
    assert res.status_code == 200
    return res.body


def test_the_second_pool_takes_no_memory_before_a_task_goes_there(tmp_path):
    global server
    server.log_path = str(tmp_path / "server.log")
    server.start()

    log = log_text()
    assert LOG_ON_DEMAND in log, "the second pool is added without its memory"
    assert LOG_FROM_THE_START not in log

    # two tasks that both fit at home leave it so
    older, newer, n_busy_max, seen = run_two(1000, 1000)
    assert_served(older, 1000)
    assert_served(newer, 1000)
    assert not seen[1]["kv_overflow"]

    assert LOG_TOOK not in log_text()


def test_the_second_pool_takes_its_memory_when_a_task_goes_there_and_gives_it_back(tmp_path):
    global server
    server.kv_overflow_release = 1
    server.log_path = str(tmp_path / "server.log")
    server.start()

    for n in (1, 2):
        # the newer task does not fit at home beside the older one: 2 * (prompt + 1800 + 1) > 2560
        older, newer, n_busy_max, seen = run_two(1800, 1800)
        assert n_busy_max == 2
        assert_served(older, 1800)
        assert_served(newer, 1800)
        assert seen[1]["kv_overflow"]
        assert log_text().count(LOG_TOOK) == n

        # nothing is asked of the server from here on: the memory goes back by itself
        assert wait_for_log(LOG_GAVE_BACK, n), "the second pool kept its memory without a task"

        assert not any(slot["kv_overflow"] for slot in slots()), "a slot is in a pool that is without its memory"


def test_the_second_pool_keeps_its_memory_while_a_task_is_at_work_there(tmp_path):
    # the memory goes back as soon as the last task there ends, and not before
    global server
    server.kv_overflow_release = 0
    server.log_path = str(tmp_path / "server.log")
    server.start()

    for n in (1, 2):
        older, newer, n_busy_max, seen = run_two(1800, 1800)
        assert n_busy_max == 2
        assert_served(older, 1800)
        assert_served(newer, 1800)
        assert seen[1]["kv_overflow"]

        assert wait_for_log(LOG_GAVE_BACK, n)
        assert log_text().count(LOG_TOOK) == n
        assert log_text().count(LOG_GAVE_BACK) == n, "the memory went back more than once for one use of the pool"


def test_a_conversation_in_the_second_pool_comes_back_after_its_memory_went_back(tmp_path):
    global server
    server.no_cache_idle_slots = True
    server.kv_overflow_release = 1
    server.log_path = str(tmp_path / "server.log")
    server.start()

    # a conversation in slot 0: 500 cells of the home pool, then idle
    first = turn(PROMPT_OLDER, 480, 0)
    assert_served(first, 480)
    held = tokens_of(PROMPT_OLDER) + first.body["tokens"]

    # its next turn, 500 + 300 tokens, does not fit at home beside a task of 1900 that is at work there: it goes to
    # the second pool, and the slot with it (as in the test above)
    results = {}
    seen = {}

    def run_long():
        results["long"] = complete(PROMPT_NEWER, 1880, 1)

    def run_again():
        results["again"] = turn(held, 300, 0)

    t_long = threading.Thread(target=run_long)
    t_long.start()
    t_end = time.time() + 10
    while not slots_at_work() and time.time() < t_end:
        time.sleep(0.002)
    t_again = threading.Thread(target=run_again)
    t_again.start()
    while t_again.is_alive():
        for slot in slots_at_work():
            seen[slot["id"]] = slot
        time.sleep(0.002)
    t_again.join()
    t_long.join()

    assert_served(results["long"], 1880)
    assert seen[0]["kv_overflow"], "the next turn went to the second pool"
    assert_not_read_again(results["again"])
    held = held + results["again"].body["tokens"]

    # the conversation is idle in the second pool with its 800 cells. Nothing is asked of the server: its cells are
    # saved to the prompt cache and the memory of the pool goes back
    assert wait_for_log(LOG_GAVE_BACK, 1), "the second pool kept its memory for an idle slot"
    assert not any(slot["kv_overflow"] for slot in slots())

    # the conversation goes on at home, from the prompt cache
    again = turn(held, 8, 0)
    assert_not_read_again(again)
    assert again.body["tokens_predicted"] == 8
    assert not slots()[0]["kv_overflow"]
    assert log_text().count(LOG_TOOK) == 1, "the conversation fits at home now"


def test_a_task_and_its_child_go_to_the_second_pool_together(tmp_path):
    # a request for two completions is a task with a child, and the two share the cells of their prompt: they are
    # fitted as one, 2 * (prompt + 500 + 1) cells, which the first pool does not have beside a task of 1800
    global server
    server.kv_overflow_release = 1
    server.log_path = str(tmp_path / "server.log")
    server.start()

    results = {}
    seen = {}

    def run_older():
        results["older"] = complete(PROMPT_OLDER, 1800, 0)

    def run_pair():
        results["pair"] = server.make_request("POST", "/chat/completions", data={
            "max_tokens": 500,
            "n": 2,
            "messages": [{"role": "user", "content": "What is the best book"}],
            "id_slot": 1,
            "ignore_eos": True,
            "temperature": 0.0,
            "seed": 42,
        })

    t_older = threading.Thread(target=run_older)
    t_older.start()
    t_end = time.time() + 10
    while not slots_at_work() and time.time() < t_end:
        time.sleep(0.002)
    t_pair = threading.Thread(target=run_pair)
    t_pair.start()
    while t_pair.is_alive():
        for slot in slots_at_work():
            seen[slot["id"]] = slot
        time.sleep(0.002)
    t_pair.join()
    t_older.join()

    assert_served(results["older"], 1800)
    pair = results["pair"]
    assert pair.status_code == 200, f"{pair.status_code} {str(pair.body)[:200]}"
    assert len(pair.body["choices"]) == 2
    assert [choice["finish_reason"] for choice in pair.body["choices"]] == ["length", "length"]
    assert pair.body["usage"]["completion_tokens"] == 1000

    assert not seen[0]["kv_overflow"]
    assert seen[1]["kv_overflow"] and seen[2]["kv_overflow"], "the task and its child are in the second pool"
    assert log_text().count(LOG_TOOK) == 1

    assert wait_for_log(LOG_GAVE_BACK, 1), "the second pool kept its memory without a task"
    assert not any(slot["kv_overflow"] for slot in slots())


def test_the_second_pool_keeps_its_memory_from_the_start_with_release_off(tmp_path):
    # --kv-overflow-release -1: the memory is taken at startup and kept
    global server
    server.kv_overflow_release = -1
    server.log_path = str(tmp_path / "server.log")
    server.start()

    log = log_text()
    assert LOG_FROM_THE_START in log
    assert LOG_ON_DEMAND not in log

    older, newer, n_busy_max, seen = run_two(1800, 1800)
    assert n_busy_max == 2
    assert_served(older, 1800)
    assert_served(newer, 1800)
    assert seen[1]["kv_overflow"]

    time.sleep(2.5)
    log = log_text()
    assert LOG_TOOK not in log
    assert LOG_GAVE_BACK not in log


def test_a_task_waits_when_the_device_has_no_memory_for_the_second_pool(tmp_path, monkeypatch):
    # LLAMA_SERVER_DEBUG_KV_OVERFLOW_NO_MEMORY: the device refuses the memory of the pool, as it does when another
    # program holds it. The task that would have gone there is fitted without the pool: it waits for the first one
    global server
    monkeypatch.setenv("LLAMA_SERVER_DEBUG_KV_OVERFLOW_NO_MEMORY", "1")
    server.log_path = str(tmp_path / "server.log")
    server.start()

    older, newer, n_busy_max, seen = run_two(1800, 1800)

    assert n_busy_max == 1, "two tasks of 1800 were at work at once in one pool of 2560"
    assert_served(older, 1800)
    assert_served(newer, 1800)
    assert not seen[0]["kv_overflow"]
    assert not seen[1]["kv_overflow"]

    log = log_text()
    assert LOG_NO_MEMORY in log
    assert LOG_TOOK not in log


# The device of the second pool gives only what it has (--kv-overflow-headroom). When a task would go to the second
# pool, the server reads what the device has free: the task goes there when that covers the pool's cells, if the pool
# is still without its memory, and the headroom on top. Otherwise the task is fitted without the pool.
#
# LLAMA_SERVER_DEBUG_KV_OVERFLOW_FREE names a file that holds the reading in MiB, in place of the device's own: a
# test writes there what another program would leave of the device.

LOG_NO_ROOM = "the second pool of KV cells gets no task for now"


@pytest.fixture
def device_free(tmp_path, monkeypatch):
    """Sets what the device of the second pool says it has free, in MiB."""
    path = tmp_path / "device-free.txt"

    def set_free(mib: int):
        # written whole: the server reads it at any moment
        tmp = tmp_path / "device-free.tmp"
        tmp.write_text(str(mib))
        os.replace(tmp, path)

    set_free(1024 * 1024)
    monkeypatch.setenv("LLAMA_SERVER_DEBUG_KV_OVERFLOW_FREE", str(path))
    return set_free


def test_a_task_does_not_go_to_the_second_pool_when_its_device_lacks_the_memory(tmp_path, device_free):
    # 100 MiB free is less than the headroom (1536 MiB by default): nothing is taken from the device, and the task
    # that would have gone to the second pool waits for the first one
    global server
    device_free(100)
    server.log_path = str(tmp_path / "server.log")
    server.start()

    older, newer, n_busy_max, seen = run_two(1800, 1800)

    assert n_busy_max == 1, "two tasks of 1800 were at work at once in one pool of 2560"
    assert_served(older, 1800)
    assert_served(newer, 1800)
    assert not seen[0]["kv_overflow"]
    assert not seen[1]["kv_overflow"]

    log = log_text()
    assert LOG_NO_ROOM in log
    assert LOG_TOOK not in log, "the pool took its memory from a device that did not have it free"
    assert LOG_NO_MEMORY not in log, "the device was asked for the memory"


def test_a_task_goes_to_the_second_pool_again_when_its_device_has_the_memory(tmp_path, device_free):
    global server
    device_free(100)
    server.log_path = str(tmp_path / "server.log")
    server.start()

    older, newer, n_busy_max, seen = run_two(1800, 1800)
    assert n_busy_max == 1
    assert not seen[1]["kv_overflow"]
    assert LOG_TOOK not in log_text()

    # the other program has left
    device_free(20000)

    older, newer, n_busy_max, seen = run_two(1800, 1800)
    assert n_busy_max == 2
    assert_served(older, 1800)
    assert_served(newer, 1800)
    assert seen[1]["kv_overflow"]
    assert log_text().count(LOG_TOOK) == 1


@pytest.mark.parametrize("headroom,free,goes", [
    (None, 2000, True),   # the default, 1536 MiB
    (4096, 2000, False),
    (4096, 5000, True),
    (0,    64,   True),   # only the pool's own cells have to be free: a few MiB here
    (0,    0,    False),
    (-1,   0,    True),   # the device is not asked
])
def test_the_headroom_is_what_the_device_has_free_beyond_the_pool(tmp_path, device_free, headroom, free, goes):
    global server
    device_free(free)
    server.kv_overflow_headroom = headroom
    server.log_path = str(tmp_path / "server.log")
    server.start()

    older, newer, n_busy_max, seen = run_two(1800, 1800)

    assert_served(older, 1800)
    assert_served(newer, 1800)
    assert n_busy_max == (2 if goes else 1)
    assert seen[1]["kv_overflow"] == goes
    assert (LOG_TOOK in log_text()) == goes
    assert (LOG_NO_ROOM in log_text()) == (not goes)


def test_a_task_does_not_join_the_second_pool_when_its_device_has_run_short(tmp_path, device_free):
    # The pool has its memory and a task at work. Then another program takes the device's memory. A third task has
    # the cells in the second pool, and not at home: it waits all the same, for the room at home
    global server
    server.kv_overflow_release = 0
    server.log_path = str(tmp_path / "server.log")
    server.start()

    results = {}
    seen = {}

    def run(key: str, n_predict: int, id_slot: int):
        results[key] = complete(PROMPT_OLDER, n_predict, id_slot)

    def wait_at_work(n: int):
        t_end = time.time() + 10
        while len(slots_at_work()) < n and time.time() < t_end:
            time.sleep(0.002)
        assert len(slots_at_work()) >= n

    # at home: prompt + 1900 + 1 cells of 2560. The next task, of prompt + 1000 + 1, does not fit beside it
    t_home = threading.Thread(target=run, args=("home", 1900, 0))
    t_home.start()
    wait_at_work(1)
    t_pool = threading.Thread(target=run, args=("pool", 1000, 1))
    t_pool.start()
    wait_at_work(2)
    assert log_text().count(LOG_TOOK) == 1

    device_free(100)

    # prompt + 700 + 1 cells: free in the second pool, and not at home
    t_third = threading.Thread(target=run, args=("third", 700, 2))
    t_third.start()

    n_busy_max = 0
    while t_home.is_alive() or t_pool.is_alive() or t_third.is_alive():
        busy = slots_at_work()
        n_busy_max = max(n_busy_max, len(busy))
        for slot in busy:
            seen[slot["id"]] = slot
        time.sleep(0.002)
    for t in (t_home, t_pool, t_third):
        t.join()

    assert_served(results["home"], 1900)
    assert_served(results["pool"], 1000)
    assert_served(results["third"], 700)
    assert seen[1]["kv_overflow"]
    assert not seen[2]["kv_overflow"], "the third task went to a pool whose device had run short"
    assert n_busy_max == 2
    assert LOG_NO_ROOM in log_text()


def conversation_in_the_second_pool(n_long: int):
    """A task of n_long tokens at work at home in slot 0, and one turn of a conversation in slot 1, which does not
    fit beside it and so is in the second pool: prompt + 700 cells. Returns the thread of the task at home, where its
    response goes, and the tokens that the conversation holds."""
    results = {}

    def run_long():
        results["long"] = complete(PROMPT_NEWER, n_long, 0)

    t_long = threading.Thread(target=run_long)
    t_long.start()
    t_end = time.time() + 10
    while not slots_at_work() and time.time() < t_end:
        time.sleep(0.002)

    first = turn(PROMPT_OLDER, 700, 1)
    assert_served(first, 700)
    assert slots()[1]["kv_overflow"]
    assert log_text().count(LOG_TOOK) == 1

    return t_long, results, tokens_of(PROMPT_OLDER) + first.body["tokens"]


def test_a_conversation_in_the_second_pool_goes_on_there_when_its_device_has_run_short(tmp_path, device_free):
    # The next turn of a conversation whose cells are in the second pool does not fit at home. It is not turned
    # away for the device's sake: it runs where its cells are, and nothing waits
    global server
    server.no_cache_idle_slots = True
    server.log_path = str(tmp_path / "server.log")
    server.start()

    # at home: prompt + 1900 + 1 cells of 2560, so prompt + 700 + 1 do not fit beside it
    t_long, results, held = conversation_in_the_second_pool(1900)

    device_free(100)

    seen = {}

    def run_again():
        results["again"] = turn(held, 300, 1)

    t_again = threading.Thread(target=run_again)
    t_again.start()
    while t_again.is_alive():
        for slot in slots_at_work():
            seen[slot["id"]] = slot
        time.sleep(0.002)
    t_again.join()
    assert t_long.is_alive(), "the task at home ended first: the turn had the room at home, and the test tells nothing"
    t_long.join()

    assert_served(results["long"], 1900)
    assert seen[1]["kv_overflow"], "the turn left the pool that holds its cells"
    assert_not_read_again(results["again"])
    assert results["again"].body["tokens_predicted"] == 300
    assert LOG_NO_ROOM not in log_text()


def test_a_new_conversation_in_a_slot_of_the_second_pool_is_turned_away_like_any_task(tmp_path, device_free):
    # the slot holds the cells of another conversation in the second pool: a task that does not go on from them
    # is a new task for the device, which has run short
    global server
    server.no_cache_idle_slots = True
    server.log_path = str(tmp_path / "server.log")
    server.start()

    t_long, results, held = conversation_in_the_second_pool(1900)

    device_free(100)

    seen = {}

    def run_new():
        # prompt + 700 + 1 cells: free in the second pool, and not at home
        results["new"] = complete("The sun was warm and the birds sang in the old tree by the river.", 700, 1)

    t_new = threading.Thread(target=run_new)
    t_new.start()
    n_busy_max = 0
    while t_new.is_alive():
        busy = slots_at_work()
        n_busy_max = max(n_busy_max, len(busy))
        for slot in busy:
            seen[slot["id"]] = slot
        time.sleep(0.002)
    t_new.join()
    t_long.join()

    assert_served(results["long"], 1900)
    assert_served(results["new"], 700)
    assert not seen[1]["kv_overflow"], "a new task went to a pool whose device had run short"
    assert LOG_NO_ROOM in log_text()
