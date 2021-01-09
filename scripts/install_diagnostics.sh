#!/usr/bin/env bash

sudo apt update
sudo apt install zip pandoc kile okular

git clone --recursive https://github.com/Sensirion/embedded-sfm.git
cd embedded-sfm
make release
cd ..

