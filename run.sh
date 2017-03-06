#!/usr/bin/env bash
LOG_FOLDER="results/logs/"
LG_EXT=".log"
PYTHONPATH='/home/bms/projects/stylometory' python -W ignore -u $1 -n $2  ${@:3} > $LOG_FOLDER$2$LG_EXT 2>&1 &
