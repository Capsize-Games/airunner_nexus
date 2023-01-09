#!/usr/bin/bash
DISABLE_TELEMETRY=true
PYTHONOPTIMIZE=0 python3 -m PyInstaller --log-level=WARN --noconfirm --clean ./build.spec 2>&1 | tee build.log

## get version from VERSION file
VERSION=$(cat VERSION)

## tar.gz the dist/runai directory
cd dist
tar -czvf runai-${VERSION}.tar.gz runai
cd ..
