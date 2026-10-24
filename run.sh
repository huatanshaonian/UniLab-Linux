#!/usr/bin/env bash
# 启动 UniLab for Linux
cd "$(dirname "$0")"
exec python3 unilab_linux.py "$@"
