from __future__ import annotations

import atexit
import multiprocessing as mp
import threading
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Callable, Optional

from pebble import ProcessPool

_WORKER_CHECKER = None

_POOLS: dict[tuple, ProcessPool] = {}
_POOLS_LOCK = threading.Lock()

def _get_pool(start_method: str, max_workers: int, max_tasks: int) -> ProcessPool:
    key = (start_method, max_workers, max_tasks)
    with _POOLS_LOCK:
        pool = _POOLS.get(key)
        if pool is None or not pool.active:
            pool = ProcessPool(
                max_workers=max_workers,
                max_tasks=max_tasks,
                context=mp.get_context(start_method),
            )
            _POOLS[key] = pool
        return pool

@atexit.register
def _shutdown_pools() -> None:
    for pool in _POOLS.values():
        if pool.active:
            pool.stop()
            pool.join()

def _default_score(trace: str, gt: str, steps) -> tuple[bool, Optional[int]]:
    global _WORKER_CHECKER
    if _WORKER_CHECKER is None:
        from sst.shared.rule_checker import RuleBasedChecker

        _WORKER_CHECKER = RuleBasedChecker()
    if steps:
        return _WORKER_CHECKER.check(trace, gt, steps)
    return _WORKER_CHECKER.check(trace, gt)

def _coerce(out) -> tuple[bool, Optional[int]]:
    if isinstance(out, tuple) and len(out) == 2:
        return bool(out[0]), out[1]
    return bool(out), None

def batch_check_hard_timeout(
    traces: list[str],
    ground_truths: list[str],
    extracted_steps: Optional[list] = None,
    *,
    timeout_s: float,
    score_fn: Optional[Callable] = None,
    start_method: str = "spawn",
    max_workers: int = 4,
    max_tasks: int = 0,
) -> tuple[list[bool], list[Optional[int]]]:

    n = len(traces)
    if len(ground_truths) != n:
        raise ValueError(
            f"Length mismatch: len(traces)={n} vs len(ground_truths)={len(ground_truths)}"
        )
    if extracted_steps is None:
        extracted_steps = [None] * n
    elif len(extracted_steps) != n:
        raise ValueError(
            f"Length mismatch: len(extracted_steps)={len(extracted_steps)} vs len(traces)={n}"
        )

    fn = score_fn or _default_score
    pool = _get_pool(start_method, max_workers, max_tasks)
    results: list[tuple[bool, Optional[int]]] = [(False, None)] * n

    futures = [
        pool.schedule(
            fn,
            args=(traces[i], ground_truths[i], extracted_steps[i]),
            timeout=timeout_s,
        )
        for i in range(n)
    ]
    for i, fut in enumerate(futures):
        try:
            results[i] = _coerce(fut.result())
        except FuturesTimeoutError:

            results[i] = (False, None)

    flags = [r[0] for r in results]
    step_idxs = [r[1] for r in results]
    return flags, step_idxs
