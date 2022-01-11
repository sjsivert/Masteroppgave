import logging
import os
import shutil
from contextlib import contextmanager


@contextmanager
def temp_files(path: str):
    """
    Context manager for creating temporary folder and removing it after.
    For example if a file has to be created before uploading to neptune.
    args:
      path: path to temporary folder
    """
    logging.debug(f"Creating temporary folder {path}")

    try:
        os.mkdir(path)
        yield

    except FileExistsError:  # pragma: no cover
        # It should not delete folders it did not create self
        logging.debug(f"{path} already exists")
        yield
    else:  # noqa
        logging.debug(f"Removing temporary folder {path}")
        shutil.rmtree(path)
