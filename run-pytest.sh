#!/usr/bin/env bash

if  [[ $1 = "--cbs" ]]; then
    pytest -s tests/
else
    pytest -rP tests/ -k "not cbs"
fi

