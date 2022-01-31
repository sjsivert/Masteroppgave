import logging
import os
import shutil
from pathlib import Path

from src.utils.config_parser import config


class LocalCheckpointSaveSource:
    @staticmethod
    def get_checkpoint_save_location() -> Path:
        path = config["experiment"]["checkpoint_save_location"].get()
        return Path(path)

    @staticmethod
    def get_log_frequency() -> int:
        return config["experiment"]["log_model_every_n_epoch"].get()

    @staticmethod
    def wipe_and_init_checkpoint_save_location() -> None:
        logging.info(
            f"Wiping and initializing checkpoint save location {LocalCheckpointSaveSource.get_checkpoint_save_location()}"
        )
        try:
            shutil.rmtree(LocalCheckpointSaveSource.get_checkpoint_save_location())
        except FileNotFoundError:
            pass

        os.mkdir(LocalCheckpointSaveSource.get_checkpoint_save_location())

    @staticmethod
    def write_file(file_name: str, file_content: str) -> None:
        if LocalCheckpointSaveSource.get_log_frequency() > 0:
            try:
                with open(
                        f"{LocalCheckpointSaveSource.get_checkpoint_save_location()}/{file_name}", "w"
                ) as f:
                    f.write(file_content)
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Could not write to {LocalCheckpointSaveSource.get_checkpoint_save_location()}/{file_name}"
                )


def init_local_checkpoint_save_location(title: str, description: str) -> None:
    if LocalCheckpointSaveSource().get_log_frequency() > 0:
        LocalCheckpointSaveSource().wipe_and_init_checkpoint_save_location()
        LocalCheckpointSaveSource().write_file(
            file_name="title-description.txt", file_content=f"{title}\n{description}"
        )
        LocalCheckpointSaveSource().write_file(
            file_name="options.yaml", file_content=config.dump()
        )
