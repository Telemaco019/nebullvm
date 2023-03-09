from typing import Optional

from surfer.common import constants
from surfer.common.exceptions import InternalError
from surfer.common.schemas import SurferConfig


class SurferConfigManager:
    def __init__(
        self,
        base_path=constants.SURFER_CONFIG_BASE_DIR_PATH,
        config_file_name=constants.SURFER_CONFIG_FILE_NAME,
    ):
        """
        Parameters
        ----------
        base_path: Path
            The base path to the directory containing Cloud Surfer files
        config_file_name: str
            The name of the Cloud Surfer configuration file
        """
        self.config_file_path = base_path / config_file_name

    def config_exists(self) -> bool:
        """Check if the Cloud Surfer configuration file exists

        Returns
        -------
        bool
            True if the Cloud Surfer configuration file exists,
            False otherwise
        """
        return self.config_file_path.exists()

    def save_config(self, config: SurferConfig):
        """Save to file the Cloud Surfer configuration

        If the Cloud Surfer configuration file already exists,
        it will be overwritten.

        Parameters
        ----------
        config: SurferConfig
            The Cloud Surfer configuration to save
        """
        # Create config file
        self.config_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_file_path, "w") as f:
            f.write(config.json(indent=2))

    def load_config(self) -> Optional[SurferConfig]:
        """Load the Cloud Surfer configuration

        Raises
        ------
        InternalError
            If the Cloud Surfer file is found but failed to parse

        Returns
        -------
        Optional[SurferConfig]
            The Cloud Surfer configuration if Cloud Surfer has been
            already initialized, None otherwise
        """
        if not self.config_exists():
            return None
        try:
            return SurferConfig.parse_file(self.config_file_path)
        except Exception as e:
            raise InternalError(
                "error parsing CloudSurfer config at {}: {}".format(
                    self.config_file_path,
                    e,
                )
            )
