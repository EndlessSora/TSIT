#!/usr/bin/env bash

set -x

cd ./datasets
wget --no-check-certificate https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/summer2winter_yosemite.zip
unzip summer2winter_yosemite.zip
rm -f summer2winter_yosemite.zip
cd ..
