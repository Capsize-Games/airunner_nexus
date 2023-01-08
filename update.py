import os
import time
from logger import logger
import shutil
import urllib.request
import tarfile
from settings import (
    KSD_VERSION_CHECK_URL,
    RUNAI_VERSION_CHECK_URL,
    CACHE_TIME,
    KSD_VERSION_FILE,
    RUNAI_VERSION_FILE,
    KSD_DOWNLOAD_FILE,
    RUNAI_DOWNLOAD_FILE,
    UPGRADE_PATH,
    VERSION,
)


class Update:
    def do_update(self):
        # download the latest versions
        self.download_updates()

        # make a backup of this directory
        self.backup_files()

        # replace old files with new downloaded files
        # self.update_all_files()

    def cached_or_request(self, file, url):
        contents = self.read_cache_file(file)
        if not contents:
            logger.info("Making request to " + url)
            response = urllib.request.urlopen(url)
            contents = response.read().decode("utf-8")
            self.save_to_cache(file, contents)
        return contents

    def save_to_cache(self, file, contents):
        # open with create if not exists
        with open(file, 'w+') as f:
            logger.info("Saving cache to " + file)
            f.write(contents)

    def read_cache_file(self, file_name):
        try:
            # check if file exists
            if os.path.isfile(file_name):
                file_age = time.time() - os.path.getmtime(file_name)
                if file_age < CACHE_TIME:
                    with open(file_name) as f:
                        contents = f.read()
                        return contents
        except Exception as e:
            logger.error(e)
        logger.info("No cache file found")
        return None

    def get_current_versions(self):
        logger.info("CHECK FOR LATEST VERSION")
        try:
            # check the files for json
            ksd_version = self.cached_or_request(KSD_VERSION_FILE, KSD_VERSION_CHECK_URL)
            runai_version = self.cached_or_request(RUNAI_VERSION_FILE, RUNAI_VERSION_CHECK_URL)

            # send versions to client
            return {
                "versions": {
                    "latest_ksd_version": ksd_version,
                    "latest_runai_version": runai_version,
                    "current_runai_version": VERSION,
                }
            }
        except Exception as e:
            logger.error("Could not check for latest version" + str(e))
        return {}

    def download_and_extract(self, url, file_name):
        HERE = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(HERE, file_name)
        logger.info(f"Downloading {url} to {path}")
        urllib.request.urlretrieve(url, path)
        with tarfile.open(path) as tar:
            tar.extractall(os.path.join(HERE, "tmp"))

    def download_extract_krita_stable_diffusion_plugin(self, version_data):
        latest_ksd_version = version_data["versions"]["latest_ksd_version"]
        ksd_file_name = f"{latest_ksd_version}_ksd.tar.gz"
        ksd_download_url = f"https://github.com/w4ffl35/krita_stable_diffusion/releases/tag/{latest_ksd_version}/{ksd_file_name}"
        self.download_and_extract(ksd_download_url, ksd_file_name)

    def download_extract_runai_server(self, version_data):
        latest_runai_version = version_data["versions"]["latest_runai_version"]
        runai_file_name = f"{latest_runai_version}_runai.tar.gz"
        runai_download_url = f"https://sddist.s3.amazonaws.com/{runai_file_name}"
        self.download_and_extract(runai_download_url, runai_file_name)

    def download_updates(self):
        version_data = self.get_current_versions()
        #self.download_extract_krita_stable_diffusion_plugin(version_data)
        self.download_extract_runai_server(version_data)

    def backup_files(self):
        """
        Make a backup of this directory
        :return:
        """
        logger.info("Backing up files")
        # make a backup of this directory
        HERE = os.path.dirname(os.path.realpath(__file__))
        backup_dir = os.path.join(HERE, "tmp")
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)
        backup_file = os.path.join(
            backup_dir,
            "runai_backup_" + str(time.time()) + ".tar.gz"
        )
        # move this to ~/stablediffusion
        with tarfile.open(backup_file, "w:gz") as tar:
            tar.add(
                os.path.dirname(
                    os.path.realpath(__file__)
                ),
                arcname=os.path.basename(
                    os.path.dirname(
                        os.path.realpath(__file__)
                    )
                ),
                filter=lambda x: None if x.name in [
                    "build",
                    "dist",
                    "lib",
                    "upgrade",
                    "backup",
                    ".git",
                    "tmp",
                    "database.db",
                    "krita-stable-diffusion.log"
                ] else x
            )

    def update_all_files(self):
        """
        copy files from the update directory to the current directory
        :return:
        """
        logger.info("Updating all files")
        # copy files from the update directory to the current directory
        for file in os.listdir(os.path.join(UPGRADE_PATH, "runai")):
            shutil.copy(
                os.path.join(UPGRADE_PATH, "runai", file),
                os.path.join(os.path.dirname(os.path.realpath(__file__)), file)
            )
