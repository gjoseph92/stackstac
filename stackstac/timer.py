from contextlib import contextmanager
from timeit import default_timer
import logging

logger = logging.getLogger(__name__)


@contextmanager
def time(statement: str, level=logging.INFO):
    start = default_timer()
    error = None
    try:
        yield
    except Exception as e:
        error = e
    finally:
        t = default_timer() - start
        msg = statement.format(t=f"{t:.2f}s")
        if error:
            msg = f"ERROR: {error} - {msg}"
        logger.log(level, msg)
    if error:
        raise error
