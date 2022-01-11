#!/usr/bin/env python

from pathlib import Path

import click
import confuse

current_dir = Path(__file__)
root_dir = Path(current_dir.parent.parent.parent)

config = confuse.Configuration("Masteroppgave", __name__)
config.set_file(f"{root_dir}/config.yaml")
config.set_env()
config.exists()


def get_absolute_path(path):
    return f"{root_dir}/{path}"
