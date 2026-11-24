import os
import threading
import time

import pytest
from utils import *

# --slots-max: the slots beyond --parallel take the memory of their state when a task uses them, out of what the KV
# cells have not taken, and the tasks are fitted into what the memory may take.
#
# The state of a slot costs something only in a model with recurrent layers. The tests of that use a dummy model as
# test-llama-archs writes them: LLAMA_TEST_DUMMY_MODELS names their folder (they are not downloaded). Its vocabulary
# is not for text, so its prompts are token ids. In it a row of the state takes what 128 cells take.
DUMMY = os.environ.get("LLAMA_TEST_DUMMY_MODELS")
HYBRID = os.path.join(DUMMY, "qwen35-dense.gguf") if DUMMY else None
needs_hybrid = pytest.mark.skipif(not HYBRID or not os.path.exists(HYBRID),
                                  reason="LLAMA_TEST_DUMMY_MODELS names no folder with qwen35-dense.gguf")

ROW_CELLS = 128

# A second pool of KV cells beside the first needs a device: LLAMA_TEST_KV_OVERFLOW_DEVICE names one that the server
# lists, e.g. CUDA0 (see test_kv_fit.py).
DEVICE = os.environ.get("LLAMA_TEST_KV_OVERFLOW_DEVICE")
needs_second_pool = pytest.mark.skipif(not DEVICE, reason="LLAMA_TEST_KV_OVERFLOW_DEVICE names no device for a second pool")

server = ServerPreset.tinyllama2()

PROMPT = "Once upon a time there was a little cat who lived in a big house."


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.kv_unified = True
    server.server_slots = True
    server.temperature = 0.0


def with_second_pool(n_cells):
    global server
    if n_cells:
        server.kv_overflow_device = DEVICE
        server.kv_overflow_size = n_cells
        server.kv_overflow_headroom = -1  # the device is not asked: what else is on it is not the tests' matter


def start_tiny(n_slots, slots_max, n_ctx=4096, more_args=None, second_pool=None):
    global server
    server = ServerPreset.tinyllama2()
    server.kv_unified = True
    server.server_slots = True
    server.temperature = 0.0
    server.n_ctx = n_ctx
    server.n_slots = n_slots
    server.slots_max = slots_max
    server.extra_args = more_args
    with_second_pool(second_pool)
    server.start()


def start_hybrid(n_slots, slots_max, n_ctx=4096, more_args=None, second_pool=None, on_cpu=True):
    global server
    server = ServerProcess()
    server.model_hf_repo = None
    server.model_hf_file = None
    server.model_file = HYBRID
    server.model_alias = "qwen35-dummy"
    server.kv_unified = True
    server.server_slots = True
    server.temperature = 0.0
    server.seed = 42
    server.n_ctx = n_ctx
    server.n_batch = 64
    server.n_slots = n_slots
    server.slots_max = slots_max
    # an idle slot keeps what it holds until a task needs it
    server.no_cache_idle_slots = True
    server.cache_ram = 64
    # the dummy model was "trained" with 256 tokens, which would be the limit of every slot
    server.extra_args = ["--override-kv", "qwen35.context_length=int:65536"] + (more_args or [])
    # The sizes in these tests are in cells, and a row is 128 of them where the memory is taken in pages. A GPU hands
    # it out in larger pieces (2 MiB with CUDA), more than this model's whole memory: there the model stays on the CPU.
    if on_cpu:
        server.extra_args += ["--n-gpu-layers", "0"]
    with_second_pool(second_pool)
    server.start()


def ids(n, first=1):
    """n token ids of the dummy model, a different run of them for each `first`"""
    return [1 + (first * 7 + i * 13) % 97 for i in range(n)]


def slots():
    res = server.make_request("GET", "/slots")
    assert res.status_code == 200
    return res.body


def n_at_work():
    return sum(1 for s in slots() if s["is_processing"])


def in_background(data):
    """Starts a request; returns its thread and the dict that gets its response and the time it ended."""
    out = {}

    def run():
        out["res"] = server.make_request("POST", "/completion", data=data)
        out["t_end"] = time.time()

    t = threading.Thread(target=run)
    t.start()
    return t, out


def wait_fast(what, cond, seconds=20.0):
    t_end = time.time() + seconds
    while time.time() < t_end:
        if cond():
            return
        time.sleep(0.002)
    raise TimeoutError(what)


def test_without_the_option_the_slots_are_those_of_parallel():
    start_tiny(2, None)
    props = server.make_request("GET", "/props").body
    assert props["total_slots"] == 2
    assert props["slots_set_aside"] == 0
    assert len(slots()) == 2


def test_the_slots_beyond_parallel_are_slots_of_the_server():
    start_tiny(1, 4)
    props = server.make_request("GET", "/props").body
    assert props["total_slots"] == 4
    assert props["slots_set_aside"] == 1
    assert len(slots()) == 4


def test_slots_max_that_is_not_more_than_parallel_changes_nothing():
    start_tiny(2, 2)
    props = server.make_request("GET", "/props").body
    assert props["total_slots"] == 2
    assert props["slots_set_aside"] == 0


def test_more_tasks_than_parallel_run_at_once():
    start_tiny(1, 4)

    tasks = [in_background({"prompt": "Story %d. " % i + PROMPT, "n_predict": 700, "ignore_eos": True, "cache_prompt": False})
             for i in range(4)]

    seen = 0
    while any(t.is_alive() for t, _ in tasks):
        seen = max(seen, n_at_work())
    for t, out in tasks:
        t.join()
        assert out["res"].status_code == 200, out["res"].body
        assert out["res"].body["tokens_predicted"] == 700

    assert seen == 4


def test_the_answers_are_those_of_a_server_with_all_its_memory():
    answers = []
    for slots_max in (None, 4):
        start_tiny(2, slots_max)
        res = server.make_request("POST", "/completion", data={"prompt": PROMPT, "n_predict": 48, "seed": 42, "cache_prompt": False})
        assert res.status_code == 200
        answers.append(res.body["content"])
        server.stop()

    assert answers[0] == answers[1]


@needs_hybrid
def test_the_answers_of_a_model_with_state_are_those_of_a_server_with_all_its_memory():
    answers = []
    for slots_max in (None, 4):
        start_hybrid(2, slots_max)
        both = []
        for first in (1, 2, 1):
            res = server.make_request("POST", "/completion", data={
                "prompt": ids(200, first), "n_predict": 32, "seed": 42, "ignore_eos": True, "cache_prompt": True})
            assert res.status_code == 200, res.body
            assert res.body["tokens_predicted"] == 32
            both.append(res.body["content"])
        answers.append(both)
        server.stop()

    assert answers[0] == answers[1]
    # the same conversation again gives the same answer
    assert answers[1][0] == answers[1][2]


# What the memory of the dummy model may take, in cells (a cell is 1024 bytes): the cells of -c and a row of 128 for
# each slot that is set aside, less what the small states of the slots beyond those hold from the start, 10 a slot.
# The cells that can have memory beside n_rows rows: what is left, in the steps of 256 that the cell mark moves in.
def cells_beside(n_ctx, n_set_aside, n_slots, n_rows):
    budget = n_ctx + ROW_CELLS * n_set_aside - 10 * (n_slots - n_set_aside)
    return min(n_ctx, (budget - ROW_CELLS * n_rows) // 256 * 256)


def run_two_long_and_third(n_predict_a, n_predict_b, n_predict_c, n_prompt_c=292):
    """Two long tasks, and a third as soon as both are at work. Returns the three results, the most slots seen at
    work at once, and whether a slot was seen at work in the second pool."""
    t_a, out_a = in_background({"prompt": ids(100, 1), "n_predict": n_predict_a, "ignore_eos": True, "cache_prompt": False})
    t_b, out_b = in_background({"prompt": ids(100, 2), "n_predict": n_predict_b, "ignore_eos": True, "cache_prompt": False})
    wait_fast("the two long tasks are at work", lambda: n_at_work() == 2)

    t_c, out_c = in_background({"prompt": ids(n_prompt_c, 3), "n_predict": n_predict_c, "ignore_eos": True, "cache_prompt": False})

    seen = 0
    seen_second_pool = False
    while t_a.is_alive() or t_b.is_alive() or t_c.is_alive():
        at_work = [s for s in slots() if s["is_processing"]]
        seen = max(seen, len(at_work))
        seen_second_pool = seen_second_pool or any(s["kv_overflow"] for s in at_work)
    for t in (t_a, t_b, t_c):
        t.join()

    for out, n_predict in ((out_a, n_predict_a), (out_b, n_predict_b), (out_c, n_predict_c)):
        assert out["res"].status_code == 200, out["res"].body
        assert out["res"].body["tokens_predicted"] == n_predict
        assert not out["res"].body["truncated"]

    return out_a, out_b, out_c, seen, seen_second_pool


@needs_hybrid
def test_the_row_of_a_slot_beyond_parallel_takes_its_memory_from_the_cells():
    # 4096 cells and the memory of one row. Two tasks of 2001 and 1501 cells are at work. A third of 301 cells fits
    # beside them by its cells, 3803 in all, where two rows leave 3840. Its row is the third: beside three rows 3584
    # cells can have their memory, so it waits for one of the two to end.
    assert cells_beside(4096, 1, 3, 2) == 3840 and cells_beside(4096, 1, 3, 3) == 3584
    start_hybrid(1, 3)

    out_a, out_b, out_c, seen, _ = run_two_long_and_third(1900, 1400, 8)

    assert seen == 2
    assert out_c["t_end"] >= min(out_a["t_end"], out_b["t_end"])


@needs_hybrid
def test_two_rows_that_are_set_aside_leave_the_cells_to_the_tasks():
    # the same three tasks with the memory of two rows set aside: beside three rows 3840 cells can have their
    # memory, and the third task is at work with the two others
    assert cells_beside(4096, 2, 3, 3) == 3840
    start_hybrid(2, 3)

    out_a, out_b, out_c, _, _ = run_two_long_and_third(1900, 1400, 8)

    assert out_c["t_end"] < min(out_a["t_end"], out_b["t_end"]), "the third task waited for one of the long ones to end"


@needs_hybrid
def test_an_idle_slot_gives_up_its_row_and_its_cells_for_a_task():
    # 2048 cells and the memory of one row. Two conversations stay in their slots, 609 cells and a row each. A third
    # of 1209 cells does not fit beside either of them: both idle slots give up what they hold, and the first
    # conversation comes back from the prompt cache when it goes on.
    start_hybrid(1, 3, n_ctx=2048)

    first = []
    for i in (1, 2):
        res = server.make_request("POST", "/completion", data={
            "prompt": ids(600, i), "n_predict": 8, "seed": 42, "ignore_eos": True, "cache_prompt": True})
        assert res.status_code == 200, res.body
        first.append(res.body["content"])

    held = [s for s in slots() if s.get("n_prompt_tokens", 0) > 0]
    assert len(held) == 2, "an idle slot gave up its conversation before a task needed its memory"

    res = server.make_request("POST", "/completion", data={
        "prompt": ids(1200, 3), "n_predict": 8, "seed": 42, "ignore_eos": True, "cache_prompt": True})
    assert res.status_code == 200, res.body
    assert res.body["tokens_predicted"] == 8

    held = [s for s in slots() if s.get("n_prompt_tokens", 0) > 0]
    assert len(held) == 1, "the idle slots still hold their conversations beside a task that needed their memory"

    # the first conversation goes on from the prompt cache: the same answer, and its prompt is not read again
    res = server.make_request("POST", "/completion", data={
        "prompt": ids(600, 1), "n_predict": 8, "seed": 42, "ignore_eos": True, "cache_prompt": True})
    assert res.status_code == 200, res.body
    assert res.body["content"] == first[0]
    assert res.body["timings"]["prompt_n"] < 50


@needs_hybrid
def test_the_rows_of_idle_slots_are_room_for_a_task():
    # 2048 cells and the memory of one row. Two small conversations stay in their slots, with a row each, the second
    # of them out of the cells. A task of 1659 cells fits when both give up what they hold: one row is left, its
    # own, and 1792 cells beside it. With the rows of the two it would be the third, and 1536 cells would be all.
    assert cells_beside(2048, 1, 3, 1) == 1792 and cells_beside(2048, 1, 3, 3) == 1536
    start_hybrid(1, 3, n_ctx=2048)

    for i in (1, 2):
        res = server.make_request("POST", "/completion", data={
            "prompt": ids(100, i), "n_predict": 8, "seed": 42, "ignore_eos": True, "cache_prompt": True})
        assert res.status_code == 200, res.body
    assert len([s for s in slots() if s.get("n_prompt_tokens", 0) > 0]) == 2

    res = server.make_request("POST", "/completion", data={
        "prompt": ids(1650, 3), "n_predict": 8, "seed": 42, "ignore_eos": True, "cache_prompt": True})
    assert res.status_code == 200, res.body
    assert res.body["tokens_predicted"] == 8
    assert not res.body["truncated"]
    assert res.body["timings"]["prompt_n"] == 1650


@needs_hybrid
def test_a_task_starts_beside_another_when_the_rows_of_idle_slots_are_its_room():
    # 4096 cells and the memory of one row. Three small conversations stay in their slots, and a long task of 3301
    # cells is at work: four rows, three of them out of the cells, which leaves 3584 cells (a mark of the cells
    # moves in steps of 256). A task of 409 cells does not fit beside that. It does when idle slots give up their
    # rows: with the two rows of the tasks 3840 cells can have their memory, 539 beside the long task. So it starts
    # at once; it does not wait for the long task to end.
    assert cells_beside(4096, 1, 5, 4) == 3584 and cells_beside(4096, 1, 5, 2) == 3840
    start_hybrid(1, 5, n_ctx=4096)

    for i in (1, 2, 3):
        res = server.make_request("POST", "/completion", data={
            "prompt": ids(60, i), "n_predict": 8, "seed": 42, "ignore_eos": True, "cache_prompt": True})
        assert res.status_code == 200, res.body
    assert len([s for s in slots() if s.get("n_prompt_tokens", 0) > 0]) == 3

    t_long, out_long = in_background({"prompt": ids(100, 4), "n_predict": 3200, "ignore_eos": True, "cache_prompt": False})
    wait_fast("the long task is at work", lambda: n_at_work() == 1)
    assert len([s for s in slots() if s.get("n_prompt_tokens", 0) > 0 and not s["is_processing"]]) == 3, \
        "an idle slot gave up its conversation for the long task: nothing is shown"

    res = server.make_request("POST", "/completion", data={
        "prompt": ids(400, 5), "n_predict": 8, "ignore_eos": True, "cache_prompt": False})
    assert res.status_code == 200, res.body
    assert res.body["tokens_predicted"] == 8
    assert not res.body["truncated"]
    assert t_long.is_alive(), "the task waited for the long one to end"

    t_long.join()
    assert out_long["res"].status_code == 200, out_long["res"].body
    assert out_long["res"].body["tokens_predicted"] == 3200
    assert not out_long["res"].body["truncated"]


@needs_hybrid
def test_a_conversation_in_a_high_row_does_not_keep_the_memory_of_the_rows_below_it():
    # 2048 cells and the memory of two rows. Slot 0 comes first, has 256 cells kept, and is never asked to give up
    # what it holds. Its conversation of 248 cells is the fourth to start, so its state is in the fourth row. Then
    # the first two give up their rows for a task of 1409 cells; the third keeps its own. With the states moved down,
    # three rows hold memory, and 1792 cells can have theirs: 256 for slot 0, 68 for the third conversation, and
    # 1468 for the task. If the rows below slot 0's kept their memory, four rows would hold it and 1536 cells
    # could: 1212 for the task, which would not fit.
    assert cells_beside(2048, 2, 5, 3) == 1792 and cells_beside(2048, 2, 5, 4) == 1536
    start_hybrid(2, 5, n_ctx=2048, more_args=["--kv-unified-per-slot", "256,2048", "--batch-priority-slots", "0"])

    for i in (1, 2, 3):
        res = server.make_request("POST", "/completion", data={
            "prompt": ids(60, i), "n_predict": 8, "seed": 42, "ignore_eos": True, "cache_prompt": True, "id_slot": i})
        assert res.status_code == 200, res.body

    house = {"prompt": ids(240, 7), "n_predict": 8, "seed": 42, "ignore_eos": True, "cache_prompt": True, "id_slot": 0}
    res = server.make_request("POST", "/completion", data=house)
    assert res.status_code == 200, res.body
    assert res.body["id_slot"] == 0
    house_answer = res.body["content"]
    assert len([s for s in slots() if s.get("n_prompt_tokens", 0) > 0]) == 4

    res = server.make_request("POST", "/completion", data={
        "prompt": ids(1400, 5), "n_predict": 8, "seed": 42, "ignore_eos": True, "cache_prompt": True, "id_slot": 4})
    assert res.status_code == 200, res.body
    assert res.body["tokens_predicted"] == 8
    assert not res.body["truncated"]

    # slot 0 still holds its conversation: the same request reads no prompt again, and has the same answer
    assert [s for s in slots() if s["id"] == 0][0]["n_prompt_tokens"] > 0
    res = server.make_request("POST", "/completion", data=house)
    assert res.status_code == 200, res.body
    assert res.body["timings"]["prompt_n"] < 20
    assert res.body["content"] == house_answer


def test_a_prompt_that_does_not_fit_in_the_room_of_its_task_is_refused_at_once():
    # 2560 cells. Slot 0 comes first and has 1024 of them kept, so 1536 are left for a task in another slot, whose
    # own limit is 2048. A prompt of 1700 tokens fits in the slot and not in that room. It is refused with the
    # context-size error, which names the room; it is not read into the cells that are kept for slot 0.
    start_tiny(2, 3, n_ctx=2560, more_args=["--kv-unified-per-slot", "1024,2048", "--batch-priority-slots", "0"])

    res = server.make_request("POST", "/tokenize", data={"content": (PROMPT + " ") * 200, "add_special": True})
    assert res.status_code == 200
    tokens = res.body["tokens"]
    assert len(tokens) > 1700

    res = server.make_request("POST", "/completion", data={
        "prompt": tokens[:1700], "n_predict": 8, "ignore_eos": True, "cache_prompt": False, "id_slot": 1})
    assert res.status_code == 400, res.body
    assert res.body["error"]["type"] == "exceed_context_size_error"
    assert res.body["error"]["n_prompt_tokens"] == 1700
    assert res.body["error"]["n_ctx"] == 1536

    # a prompt that fits in the room is read, and slot 0 has all of its cells for a task of its own
    res = server.make_request("POST", "/completion", data={
        "prompt": tokens[:1500], "n_predict": 8, "ignore_eos": True, "cache_prompt": False, "id_slot": 1})
    assert res.status_code == 200, res.body
    assert res.body["tokens_predicted"] == 8
    res = server.make_request("POST", "/completion", data={
        "prompt": tokens[:1000], "n_predict": 8, "ignore_eos": True, "cache_prompt": False, "id_slot": 0})
    assert res.status_code == 200, res.body
    assert res.body["id_slot"] == 0
    assert res.body["tokens_predicted"] == 8
    assert not res.body["truncated"]


@needs_hybrid
def test_a_slot_that_comes_first_has_its_row_and_its_cells_kept():
    # slot 0 comes first and has 512 cells; the memory of its row and of one more is set aside. Beside those 512
    # cells and the row of a second task, 3328 cells are left for the others: two long tasks of 1700 cells would
    # fit in the pool together, but not beside what is kept for slot 0, so the second waits for the first. A task
    # for slot 0 starts all the same, and ends while the first is still at work.
    start_hybrid(2, 4, more_args=["--kv-unified-per-slot", "512,3584", "--batch-priority-slots", "0"])

    # the long tasks ask for the slots behind slot 0, as a proxy in front of the server has them do
    longs = [in_background({"prompt": ids(60, i), "n_predict": 1639, "ignore_eos": True, "cache_prompt": False, "id_slot": i})
             for i in (1, 2)]
    wait_fast("a long task is at work", lambda: n_at_work() >= 1)

    res = server.make_request("POST", "/completion", data={
        "prompt": ids(400, 9), "n_predict": 8, "ignore_eos": True, "cache_prompt": False, "id_slot": 0})
    assert res.status_code == 200, res.body
    assert res.body["id_slot"] == 0
    assert all(t.is_alive() for t, _ in longs), "a long task ended before the one of slot 0: %s" % [
        (out["res"].status_code, str(out["res"].body)[:300]) for _, out in longs if "res" in out]

    seen = 0
    while any(t.is_alive() for t, _ in longs):
        seen = max(seen, n_at_work())
    for t, out in longs:
        t.join()
        assert out["res"].status_code == 200, out["res"].body
        assert out["res"].body["tokens_predicted"] == 1639

    assert seen == 1, "two long tasks were at work at once in the cells that are kept for the slot that comes first"


# With a second pool of KV cells (--kv-overflow-device) a task that does not fit in the first pool goes whole to the
# second. Its row stays in the first pool, where the state of every slot is.

@needs_second_pool
def test_a_task_that_does_not_fit_in_the_first_pool_goes_to_the_second():
    # 2560 cells in each pool, the first on a GPU where the server has one. Two tasks of more than 1800 cells do not
    # fit in the first together: the second of them goes to the second pool, and both are at work at once.
    start_tiny(1, 3, n_ctx=2560, second_pool=2560)

    tasks = [in_background({"prompt": "Story %d. " % i + PROMPT, "n_predict": 1800, "ignore_eos": True, "cache_prompt": False})
             for i in range(2)]

    seen = 0
    seen_second_pool = False
    while any(t.is_alive() for t, _ in tasks):
        at_work = [s for s in slots() if s["is_processing"]]
        seen = max(seen, len(at_work))
        seen_second_pool = seen_second_pool or any(s["kv_overflow"] for s in at_work)
    for t, out in tasks:
        t.join()
        assert out["res"].status_code == 200, out["res"].body
        assert out["res"].body["tokens_predicted"] == 1800
        assert not out["res"].body["truncated"]

    assert seen == 2
    assert seen_second_pool


@needs_hybrid
@needs_second_pool
def test_a_task_in_the_second_pool_has_its_row_in_the_first():
    # 4096 cells in each pool and the memory of one row. Two tasks of 2001 and 1501 cells are at work in the first
    # pool. A third of 1381 cells goes to the second pool. Its row is in the first, the third there: beside three
    # rows 3584 cells can have their memory, and the two tasks have 3502. All three are at work at once.
    assert cells_beside(4096, 1, 3, 3) == 3584
    start_hybrid(1, 3, second_pool=4096)

    _, _, _, seen, seen_second_pool = run_two_long_and_third(1900, 1400, 1000, n_prompt_c=380)

    assert seen == 3
    assert seen_second_pool


@needs_hybrid
@needs_second_pool
def test_a_task_waits_for_its_row_although_the_second_pool_has_the_cells():
    # the same with tasks of 2201 and 1501 cells in the first pool, 3702 of the 3840 that two rows leave. Beside a
    # third row 3584 can have their memory: there is none for the row of a third task. It waits for one of the two
    # to end, with all of the second pool free.
    assert cells_beside(4096, 1, 3, 2) == 3840 and cells_beside(4096, 1, 3, 3) == 3584
    start_hybrid(1, 3, second_pool=4096)

    out_a, out_b, out_c, seen, _ = run_two_long_and_third(2100, 1400, 1000, n_prompt_c=380)

    assert seen == 2
    assert out_c["t_end"] >= min(out_a["t_end"], out_b["t_end"])


@needs_hybrid
@needs_second_pool
def test_a_task_goes_on_in_the_second_pool_with_lazy_memory(tmp_path):
    # 4096 cells in each pool and the memory of one row. Two tasks of 2401 cells each start in the first pool with
    # their prompts and 256 cells of room, and take more as they write. Beside two rows 3840 cells can have their
    # memory, and the two need 4802: the one that finds no more there goes on in the second pool and keeps its cells
    # in the first. Both get to their ends, and no step is refused.
    assert cells_beside(4096, 1, 3, 2) == 3840
    global server
    start_hybrid(1, 3, second_pool=4096, more_args=["--kv-fit-headroom", "256", "--log-file", str(tmp_path / "server.log")])

    tasks = [in_background({"prompt": ids(100, i + 1), "n_predict": 2300, "ignore_eos": True, "cache_prompt": False})
             for i in range(2)]

    seen_both = {}
    while any(t.is_alive() for t, _ in tasks):
        for s in slots():
            if s["is_processing"] and s["kv_overflow"] and s["kv_cells_home"] > 0:
                seen_both[s["id"]] = s["kv_cells_home"]
    for t, out in tasks:
        t.join()
        assert out["res"].status_code == 200, out["res"].body
        assert out["res"].body["tokens_predicted"] == 2300
        assert not out["res"].body["truncated"]

    assert seen_both, "no task was seen with cells in both pools"
    assert all(n > 256 for n in seen_both.values())

    log = open(tmp_path / "server.log", encoding="utf-8", errors="replace").read()
    assert "goes on in the second pool" in log
    assert "does not fit in the KV" not in log, "a step was refused"


@needs_hybrid
def test_the_answers_of_a_model_with_state_on_the_device_are_those_of_a_server_with_all_its_memory():
    # the model where the server puts it by default: on a GPU, where there is one, the rows and the cells take their
    # memory in the pieces that the device hands out
    answers = []
    for slots_max in (None, 4):
        start_hybrid(2, slots_max, on_cpu=False)
        both = []
        for first in (1, 2, 3, 1):
            res = server.make_request("POST", "/completion", data={
                "prompt": ids(200, first), "n_predict": 32, "seed": 42, "ignore_eos": True, "cache_prompt": True})
            assert res.status_code == 200, res.body
            assert res.body["tokens_predicted"] == 32
            both.append(res.body["content"])
        answers.append(both)
        server.stop()

    assert answers[0] == answers[1]
    assert answers[1][0] == answers[1][3]


# The options of a server that has a slot that comes first, a list of limits, small batches for busy slots, a second
# pool of cells and a copy of the model beside it, with slots beyond --parallel, and the requests that such a server
# gets, at once.

def requests_at_once(n_rounds, n_coding, house_request):
    text = "Tell me a long story about a cat and a dog who look for a red ball in the garden. " * 4
    results = []
    lock = threading.Lock()

    def messages(i):
        data = {
            "model": "x",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": text + " Part %d." % i}],
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
        threads = [threading.Thread(target=messages, args=(i,)) for i in range(n_coding)]
        threads += [threading.Thread(target=house, args=(i,)) for i in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    assert len(results) == (n_coding + 2) * n_rounds
    return [r for r in results if not r[1]]


def start_like_a_house_server(preset, more_args):
    global server
    server = preset
    server.n_slots = 3
    server.slots_max = 6
    server.n_ctx = 2048
    server.n_batch = 512
    server.kv_unified = True
    server.no_cache_idle_slots = True
    server.server_slots = True
    server.server_metrics = True
    server.cache_ram = 256
    server.fa = "on"
    server.extra_args = [
        "--kv-unified-per-slot", "512,2048,2048",
        "--batch-priority-slots", "0",
        "--batch-size-busy", "64",
    ] + more_args
    if DEVICE:
        # a second pool that has its memory only while a task is in it
        with_second_pool(4096)
        server.kv_overflow_release = 1
    server.start()


def test_a_server_with_a_slot_that_comes_first_and_slots_beyond_parallel():
    start_like_a_house_server(ServerPreset.tinyllama2(), [])

    failed = requests_at_once(3, 7, {
        "temperature": 0.0,
        "max_tokens": 8,
        "messages": [{"role": "user", "content": "What is the color of the sky?"}],
        "id_slot": 0,
    })
    assert failed == []
    assert server.make_request("GET", "/health").status_code == 200
    assert len(slots()) == 6
    assert n_at_work() == 0


MODEL_DRAFT_FILE_URL = "https://huggingface.co/ggml-org-dev/tiny-llamas/resolve/main/stories15M-q4_0.gguf"


def test_a_server_that_drafts_tokens_has_slots_beyond_parallel():
    # the draft's context holds its memory whole, the target's takes it as it is used: the same tokens come out,
    # drafted, as from a server with all its memory, in a slot within --parallel and in one beyond it
    global server
    request = {
        "prompt": "I believe the meaning of life is",
        "temperature": 0.2,
        "top_k": 5,
        "seed": 4242,
        "n_predict": 32,
        "return_tokens": True,
        "cache_prompt": False,
    }

    answers = []
    for slots_max in (None, 4):
        server = ServerPreset.stories15m_moe()
        server.model_draft = download_file(MODEL_DRAFT_FILE_URL)
        server.spec_type = "draft-simple"
        server.spec_draft_n_min = 4
        server.spec_draft_n_max = 8
        server.fa = "on"
        server.kv_unified = True
        server.n_slots = 2
        server.slots_max = slots_max
        server.start()

        tokens = []
        for id_slot in ((0, 1) if slots_max is None else (0, 3)):
            res = server.make_request("POST", "/completion", data={**request, "id_slot": id_slot})
            assert res.status_code == 200, res.body
            assert res.body["timings"]["draft_n"] > 0
            tokens.append(res.body["tokens"])
        assert tokens[0] == tokens[1]
        answers.append(tokens[0])
        server.stop()

    assert answers[0] == answers[1]


def test_a_model_whose_memory_cannot_be_lazy_is_refused_at_start():
    # a model with a sliding window keeps its cells in a cache of another kind: the server says so and stops
    global server
    server = ServerPreset.tinygemma3()
    server.kv_unified = True
    server.n_slots = 2
    server.slots_max = 4
    with pytest.raises(RuntimeError):
        server.start()


def test_a_server_with_a_copy_of_the_model_and_slots_beyond_parallel():
    global server
    preset = ServerPreset.tinyllama2()
    preset.model_copy_device = os.environ.get("LLAMA_TEST_MODEL_COPY_DEVICE") or "none"
    preset.model_copy_ctx_size = 2048
    preset.model_copy_parallel = 2
    preset.model_copy_after = 2
    preset.model_copy_release = 60
    start_like_a_house_server(preset, [])

    props = server.make_request("GET", "/props").body
    assert props["total_slots"] == 8
    assert props["slots_set_aside"] == 3

    failed = requests_at_once(3, 9, {
        "temperature": 0.0,
        "max_tokens": 8,
        "messages": [{"role": "user", "content": "What is the color of the sky?"}],
        "id_slot": 0,
    })
    assert failed == []
    assert server.make_request("GET", "/health").status_code == 200
    assert props["model_copy"]["total_slots"] == 2
    assert server.make_request("GET", "/props").body["model_copy"]["n_tasks_total"] >= 1
