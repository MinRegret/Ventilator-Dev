#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

sudo apt-get install -y python3 python3-pip python3-setuptools libatlas-base-dev gfortran libffi-dev i2c-tools python-smbus libi2c-dev pigpio
sudo adduser pi i2c

sudo $DIR/i2c.sh

sudo systemctl enable pigpiod
sudo systemctl start pigpiod

sudo raspi-config nonint do_i2c 0
sudo raspi-config nonint do_serial 0
sudo raspi-config nonint do_spi 0
sudo raspi-config nonint do_rgpio 0
sudo raspi-config nonint do_ssh 0
sudo raspi-config nonint do_vnc 0
