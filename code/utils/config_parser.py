#!/usr/bin/env python

import confuse
from pathlib import Path
current_dir = Path(__file__)
root_dir = Path(current_dir.parent.parent.parent)

config = confuse.Configuration('Masteroppgave', __name__)
config.set_file(f'{root_dir}/config.yaml')
config.set_env()
figure_save_location = config['raport']['figure_save_location'].get()

def get_absolute_path(path):
    return f"{root_dir}/{path}"

