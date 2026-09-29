"""Small stdout logger shared by command-line scripts."""


def log(*a):
    print(*a, flush=True)
