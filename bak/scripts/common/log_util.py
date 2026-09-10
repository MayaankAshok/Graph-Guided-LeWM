"""Trivial shared logger, split into its own module so common/graph_lib.py and
common/training.py don't need to pick an arbitrary import order between each other."""


def log(*a):
    print(*a, flush=True)
