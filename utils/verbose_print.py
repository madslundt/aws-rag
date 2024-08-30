from env import VERBOSE


def verbose_print(*values: str) -> None:
    if VERBOSE:
        print(*values)
