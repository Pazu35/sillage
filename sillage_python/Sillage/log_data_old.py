import classe_ahrs
import time
import numpy as np
import gps


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

capt1 = classe_ahrs.AHRS('/dev/ttyUSB0')
capt2 = classe_ahrs.AHRS('/dev/ttyUSB1')
# capt3 = classe_ahrs.AHRS('/dev/ttyUSB3')
# capt4 = classe_ahrs.AHRS('/dev/ttyUSB4')

gps1 = gps.gps('/dev/ttyUSB2')




print("Start logging")
cnt = 0
start = time.time()
while True:
	st = time.time()
	# d1 = capt1.last_data
	# d2 = capt2.last_data
	d3 = capt3.last_data
	d4 = capt4.last_data
	l = gps1.last_data
	lp_t = time.time() - st
	if lp_t > 1/50.:
		cnt += 1
		print("loop time : ", lp_t, ' and count : ', cnt)
	# if cnt%1000 == 0 : 
	# 	print(cnt%1000)
	# cnt +=1

print("Done!")