#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

sudo apt-get install -y libatlas-base-dev gfortran libffi-dev python3-h5py i2c-tools python-smbus libi2c-dev
sudo adduser pi i2c

sudo $DIR/i2c.sh

#pip3 install virtualenv
#/home/pi/.local/bin/virtualenv venv
#source venv/bin/activate

pip3 install -e $DIR/..
pip3 install jupyter ipython tqdm matplotlib $DIR/../external/*.whl
sudo systemctl enable pigpiod
sudo systemctl start pigpiod
