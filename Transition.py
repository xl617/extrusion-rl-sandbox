from collections import namedtuple

Transition = namedtuple(
    "Transition", ("state", "action", "action1", "next_state", "reward", "done")
)