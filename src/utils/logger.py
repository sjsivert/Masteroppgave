import logging


def init_logging(log_level: str = "INFO", log_file: str = "") -> None:
    logging.basicConfig(
        level=logging.getLevelName(log_level),
        format="[%(asctime)s] %(levelname)s {%(module)s} [%(funcName)s] %(message)s",
        datefmt="%Y-%m-%d,%H:%M:%S",
    )

    logging.getLogger("pytorch_lightning").setLevel(log_level)
    if log_file != None and log_file != "":
        logger = logging.getLogger()
        # TODO: Make log file configurable
        handler = logging.FileHandler("./log_file.log", mode="w")
        logger.addHandler(handler)
    logging.info(f"Logger initialized. Log level: {log_level}, log file: {log_file}")
