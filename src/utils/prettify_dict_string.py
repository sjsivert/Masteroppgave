def prettify_dict_string(dict_string: str) -> str:
    return "\n".join([f"{key}: {value}" for key, value in dict_string.items()])
