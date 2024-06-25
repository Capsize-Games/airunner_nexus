import os
import time
import zipfile

from logger import logger
import shutil
import urllib.request
import tarfile
from settings import (
    CACHE_TIME,
)

HERE = os.path.dirname(os.path.realpath(__file__))
KRITA_DIR = os.path.join(HERE, "..", "..")
TMP = "/tmp"
# read VERSION file
with open(os.path.join(HERE, "VERSION"), "r") as f:
    VERSION = f.read().strip()


class Update:
    def do_update(self):
        # download the latest versions
        self.download_updates()

        # make a backup of this directory
        # self.backup_files()

        # replace old files with new downloaded files
        self.update_all_files()

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

    def download_and_extract(self, url, file_name):
        path = os.path.join(TMP, file_name)
        logger.info(f"Downloading {url} to {path}")
        urllib.request.urlretrieve(url, path)
        with tarfile.open(path) as tar:
            tar.extractall(os.path.join(TMP))

    # def download_extract_krita_stable_diffusion_plugin(self, version_data):
    #     latest_ksd_version = version_data["versions"]["latest_ksd_version"]
    #     ksd_file_name = f"krita_stable_diffusion-{latest_ksd_version}.zip"
    #     ksd_download_url = f"https://github.com/w4ffl35/krita_stable_diffusion/releases/tag/{latest_ksd_version}/{ksd_file_name}"
    #     self.download_and_extract(ksd_download_url, ksd_file_name)
    #
    # def download_extract_runai_server(self, version_data):
    #     latest_runai_version = version_data["versions"]["latest_runai_version"]
    #     runai_file_name = f"runai-{latest_runai_version}.tar.gz"
    #     runai_download_url = f"https://sddist.s3.amazonaws.com/{runai_file_name}"
    #     self.download_and_extract(runai_download_url, runai_file_name)

    def download_updates(self):
        # version_data = self.get_current_versions()
        #self.download_extract_krita_stable_diffusion_plugin(version_data)
        #self.download_extract_runai_server(version_data)
        pass

    def backup_files(self):
        """
        Make a backup of this directory
        :return:
        """
        logger.info("Backing up files")
        # make a backup of this directory
        backup_dir = os.path.join(TMP)
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)
        backup_file = os.path.join(
            backup_dir,
            "runai_backup_" + str(time.time()) + ".tar.gz"
        )
        # move this to ~/stablediffusion
        with tarfile.open(backup_file, "w:gz") as tar:
            tar.add(
                os.path.dirname(os.path.realpath(__file__))
            )

    def update_all_files(self):
        """
        copy files from the update directory to the current directory
        :return:
        """
        logger.info("Removing old files")
        os.chdir(KRITA_DIR)
        shutil.rmtree(os.path.join(KRITA_DIR, "krita_stable_diffusion"))
        os.remove(os.path.join(KRITA_DIR, "krita_stable_diffusion.desktop"))

        url = "https://github.com/w4ffl35/krita_stable_diffusion/releases/download/latest-linux/krita_stable_diffusion.zip"
        file_name = "krita_stable_diffusion.zip"
        krita_zip = os.path.join(TMP, file_name)

        logger.info(f"Downloading {url} to {krita_zip}")
        urllib.request.urlretrieve(url, krita_zip)

        logger.info(f"Extracting {krita_zip} to {KRITA_DIR}")
        with tarfile.open(krita_zip) as tar:
            tar.extractall(KRITA_DIR)

        logger.info("Extracting server")
        ksd_path = os.path.join(KRITA_DIR, "krita_stable_diffusion")
        os.chdir(ksd_path)
        krita_file = os.path.join(KRITA_DIR, "krita_stable_diffusion", "runai.zip")
        with zipfile.ZipFile(krita_file, 'r') as zip_ref:
            zip_ref.extractall(ksd_path)

        logger.info("Cleaning up")
        os.remove(krita_zip)
        os.remove(krita_file)
