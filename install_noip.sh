#!/usr/bin/env bash

mkdir -p /home/pi/noip
cd /home/pi/noip
wget https://www.noip.com/client/linux/noip-duc-linux.tar.gz
tar vzxf noip-duc-linux.tar.gz
cd /home/pi/noip/noip-2.1.9-1
sudo make
sudo make install
