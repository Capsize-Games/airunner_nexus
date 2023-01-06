#!/usr/bin/bash
DISABLE_TELEMETRY=true
PYTHONOPTIMIZE=0 pyinstaller --log-level=DEBUG --noconfirm ./build.spec --clean

## get version from setup.py
VERSION=$(grep -oP '(?<=version=")[0-9.]+(?=")' setup.py)
#
## tar.gz the dist/runai directory
tar -czvf runai-${VERSION}.tar.gz dist/runai
