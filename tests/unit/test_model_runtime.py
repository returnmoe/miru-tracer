"""The process-wide model lock prevents hooks/forwards from overlapping."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

from miru_tracer.core.model_runtime import serialized_model_operation


def test_serialized_model_operation_excludes_parallel_calls():
    active = 0
    maximum = 0
    guard = threading.Lock()

    @serialized_model_operation
    def operation():
        nonlocal active, maximum
        with guard:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.01)
        with guard:
            active -= 1

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: operation(), range(8)))

    assert maximum == 1
