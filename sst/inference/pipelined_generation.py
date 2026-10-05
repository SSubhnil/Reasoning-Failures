from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from itertools import islice

from sst.shared.schemas import GenerationOutput


def generate_in_batches(student, questions: list[str], batch_size: int, max_new_tokens: int, pipeline_depth: int,
                        temperature: float, seed: int | None) -> Iterator[tuple[int, GenerationOutput]]:
    """Yield (first index, output) per batch while pipeline_depth batches stay submitted to the server."""

    def submit(start: int) -> GenerationOutput:
        return student.generate(questions[start:start + batch_size], num_traces=1, do_sample=False,
                                temperature=temperature, seed=seed, max_new_tokens=max_new_tokens,
                                force_final_on_truncation=True)

    upcoming = iter(range(0, len(questions), batch_size))
    with ThreadPoolExecutor(max_workers=pipeline_depth) as pool:
        pending = deque((start, pool.submit(submit, start)) for start in islice(upcoming, pipeline_depth))
        while pending:
            start, future = pending.popleft()
            output = future.result()
            pending.extend((nxt, pool.submit(submit, nxt)) for nxt in islice(upcoming, 1))
            yield start, output
