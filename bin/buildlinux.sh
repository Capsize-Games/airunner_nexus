#!/usr/bin/bash
DISABLE_TELEMETRY=true
PYTHONOPTIMIZE=0 pyinstaller --log-level=DEBUG --noconfirm ./build.spec --clean

## get version from VERSION file
VERSION=$(cat VERSION)

## tar.gz the dist/runai directory
cd dist
tar -czvf runai-${VERSION}.tar.gz runai
cd ..
