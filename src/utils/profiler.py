import cProfile
import io
import logging
import pstats
from cgitb import enable
from typing import Callable, TypeVar

T = TypeVar("T")


def profiler(fnc: Callable[..., T], enable: bool = True) -> T:
    """
    A decorator that uses cProfile to profile an execution of a function
    """

    def inner(*args, **kwargs):
        if enable:
            profiler = cProfile.Profile()
            profiler.enable()
            # Call the function
            return_value = fnc(*args, **kwargs)
            profiler.disable()

            stream = io.StringIO()
            sortby = "cumulative"
            ps = pstats.Stats(profiler, stream=stream).sort_stats(sortby)
            # logging.INFO(ps.print_stats())
            ps.print_stats()
            return return_value
        else:
            # Call the function
            return_value = fnc(*args, **kwargs)
            return return_value

    return inner
