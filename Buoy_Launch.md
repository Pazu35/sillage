# Check usb serial port adresses cat sillage_python/Sillage/log_data.py !
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

