#!/usr/bin/env python3
import classe_ahrs
import time
import numpy as np
import gps
import subprocess

out = subprocess.Popen(['bash', 'getGPS'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

stdout, _ = out.communicate()
portGPS = stdout.split()[0]
portGPS = portGPS.decode("utf-8")

out = subprocess.Popen(['bash', 'getAHRS1'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

stdout, _ = out.communicate()
portAHRS1 = stdout.split()[0]
portAHRS1 = portAHRS1.decode("utf-8")

out = subprocess.Popen(['bash', 'getAHRS2'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

stdout, _ = out.communicate()
portAHRS2 = stdout.split()[0]
portAHRS2 = portAHRS2.decode("utf-8")

print(portGPS, portAHRS1, portAHRS2)


# Check usb serial port adresses !!!
# ls -l /dev/ttyUSB* /dev/ttyACM*
# dmesg | grep tty


# Linux cmds to launch data log on screen:
# screen -S sillage_log
# python3 sillage_python/Sillage/log_data.py

# To detach screen : 
# screen -d 
# or 
# Ctrl + a, Ctrl + d

# To reatach :
# screen -r sillage_log

# To check screens :
# screen -ls

# To kill a screen :
# go in the screen with screen -r
# exit

capt1 = classe_ahrs.AHRS('/dev/' + portAHRS1)
capt2 = classe_ahrs.AHRS('/dev/' + portAHRS2)
# capt3 = classe_ahrs.AHRS('/dev/ttyUSB3')
# capt4 = classe_ahrs.AHRS('/dev/ttyUSB4')

gps1 = gps.gps('/dev/' + portGPS)




print("Start logging")
# cnt = 0
# start = time.time()
while True:
	# st = time.time()
	d1 = capt1.last_data
	d2 = capt2.last_data
	# d3 = capt3.last_data
	# d4 = capt4.last_data
	l = gps1.last_data
	# lp_t = time.time() - st
	# if lp_t > 1/50.:
	# 	cnt += 1
	# 	print("loop time : ", lp_t, ' and count : ', cnt)
	# if cnt%1000 == 0 : 
	# 	print(cnt%1000)
	# cnt +=1

print("Done!")
