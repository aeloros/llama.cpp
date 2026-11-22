import pytest
from utils import *

server = ServerPreset.tinyllama2()

# Two conversations that start with the same tokens, as the conversations of one client do with its system prompt.
# One of them is in the prompt cache when the other starts in an empty slot: the new one takes the shared start from
# the cached state. The cached conversation keeps its copy, and comes back from the cache when it goes on.

N_SHARED = 300
N_OWN = 100


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.n_ctx = 2048
    server.n_slots = 2
    server.temperature = 0.0


def tokens_of(text: str, n: int, add_special: bool = False):
    res = server.make_request("POST", "/tokenize", data={"content": text, "add_special": add_special})
    assert res.status_code == 200
    tokens = res.body["tokens"]
    return (tokens * (n // len(tokens) + 1))[:n]


def turn(prompt, id_slot: int):
    return server.make_request("POST", "/completion", data={
        "prompt": prompt,
        "n_predict": 8,
        "id_slot": id_slot,
        "ignore_eos": True,
        "temperature": 0.0,
        "seed": 42,
        "cache_prompt": True,
        "return_tokens": True,
    })


def test_a_cached_conversation_keeps_its_copy_when_another_starts_like_it():
    global server
    server.start()

    shared = tokens_of("Once upon a time there was a little cat who lived in a big house.", N_SHARED, add_special=True)
    x = shared + tokens_of(" The cat liked to sleep in the sun and to play with a red ball.", N_OWN)
    y = shared + tokens_of(" One day a boy and his dog went to the park to look for a kite.", N_OWN)
    z = tokens_of("Tom and Lily are friends. They like to draw big pictures of the sea.", 150, add_special=True)
    n_same = next(i for i in range(len(x)) if x[i] != y[i])
    assert N_SHARED <= n_same < N_SHARED + 8, "the two go apart right after the shared start"
    assert z[1] != shared[1], "the third shares nothing with them but the first token"

    # X in slot 0; then another conversation takes that slot, and X goes to the prompt cache
    first = turn(x, 0)
    assert first.status_code == 200
    assert turn(z, 0).status_code == 200

    # Y starts like X, in a slot that holds nothing: it takes the shared start from X's cached state
    second = turn(y, 1)
    assert second.status_code == 200
    assert second.body["timings"]["prompt_n"] == len(y) - n_same, "Y read its shared start although X's state has it"

    # X goes on: its copy is still in the prompt cache
    again = turn(x + first.body["tokens"], 0)
    assert again.status_code == 200
    assert again.body["timings"]["prompt_n"] == 1, "X is read again: its copy left the prompt cache with Y"
