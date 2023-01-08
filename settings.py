import os

# server settings
PACKET_SIZE = 1024
DEFAULT_PORT = 50006
DEFAULT_HOST = "0.0.0.0"
MAX_CLIENTS = 1
DEBUG = True

# Settings for auto updates
KSD_VERSION_CHECK_URL = "https://raw.githubusercontent.com/w4ffl35/krita_stable_diffusion/master/VERSION"
RUNAI_VERSION_CHECK_URL = "https://raw.githubusercontent.com/w4ffl35/run-ai-socket-server/master/VERSION"
CACHE_TIME = 60 * 10  # 10 minutes
KSD_VERSION_FILE = "ksd_version.txt"
RUNAI_VERSION_FILE = "runai_version.txt"
BACKUP_DIR = os.path.join("tmp", "runai")
UPGRADE_PATH = os.path.join("tmp", "upgrade")
KSD_DOWNLOAD_FILE = os.path.join(UPGRADE_PATH, "ksd.latest.tar.gz")
RUNAI_DOWNLOAD_FILE = os.path.join(UPGRADE_PATH, "runai.latest.tar.gz")
