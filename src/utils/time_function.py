import logging
import time
import os
import shutil
from contextlib import contextmanager
import datetime


@contextmanager
def time_function():
    """
    Context manager for creating temporary folder and removing it after.
    For example if a file has to be created before uploading to neptune.
    args:
      path: path to temporary folder
    """
    logging.info(f"\n------------------\nStarting new timer at {time.strftime('%H:%M:%S')}")

    start_timer = lambda: datetime.datetime.now().astimezone().replace(microsecond=0)
    t_diff = lambda t: str(start_timer() - t)
    t_stamp = lambda t=None: str(t) if t else str(start_time())

    start_time = start_timer()

    t = time.process_time()
    # Execute code
    yield
    # do some stuff
    elapsed_cpu_time = time.process_time() - t

    used_time = t_diff(start_time)

    logging.info(
        f"Ending timer at: {time.strftime('%H:%M:%S')} \nElapsed CPU time: {elapsed_cpu_time * 1000 / 60}s\n"
        f"Used real time: {used_time}\n"
        f"------------------"
    )
