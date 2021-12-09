from confuse.exceptions import NotFoundError
import neptune.new as neptune
import logging
from utils.config_parser import config

def init_neptune():
    '''
    Initialize neptune experiment
    '''
    try:
      project_name = config['neptune']['project'].get()

      api_token = config['neptune']['api_token'].get()
      logging.info(f'Initializing neptune with \n \
          project_name: {project_name} \n \
          api_token: {api_token} \n \
      ')

      return neptune.init(
        project = project_name,
        api_token = api_token
      )  
    except NotFoundError as e:
      logging.exception(f'Neptune initialization failed. Failed to load configuration: {e}')



