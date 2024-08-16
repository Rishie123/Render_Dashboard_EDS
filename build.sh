#!/usr/bin/env bash
# Upgrade pip
/opt/render/project/src/.venv/bin/python -m pip install --upgrade pip

# Install requirements
/opt/render/project/src/.venv/bin/pip install -r requirements.txt
