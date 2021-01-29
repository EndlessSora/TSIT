#!/usr/bin/env bash

set -x

DATAROOT=$1

cp -r ./datasets/bdd100k_lists $DATAROOT
ln -s $DATAROOT ./datasets
