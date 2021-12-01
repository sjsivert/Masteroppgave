#!/usr/bin/env python

import confuse
from pathlib import Path
current_dir = Path(__file__)

config = confuse.Configuration('Masteroppgave', __name__)
config.set_file(f'{current_dir}/../../config.yaml')
figure_save_location = config['raport']['figure_save_location'].get()

