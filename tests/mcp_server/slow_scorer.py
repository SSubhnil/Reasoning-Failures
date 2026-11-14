"""A top-level score function that sleeps, for the hard-timeout test (spawn workers import it by name)."""

import time


def slow_score(trace, gt, steps=None):
    time.sleep(5)
    return True, {"reason": "ok", "extracted_answer": "slow"}
