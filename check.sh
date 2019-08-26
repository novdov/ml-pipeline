#!/usr/bin/env bash

echo "[ run black ]"
black -v mnist/

echo "[ run flake8 ]"
flake8 mnist/

echo "[ run isort ]"
isort -rc mnist/