from typing import OrderedDict, List

from src.utils.config_parser import config


def extract_tags_from_config() -> List[str]:
    """
    Extracts relevant information from the config which can be used as tags.
    :return: a list of tags from the config
    """
    tags = [config["model"]["model_type"].get()]
    return tags
