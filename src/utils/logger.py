import logging


def init_logging():
    logging.basicConfig(
        # filename='main.log',
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s {%(module)s} [%(funcName)s] %(message)s",
        datefmt="%Y-%m-%d,%H:%M:%S:%f",
    )
