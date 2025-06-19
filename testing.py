import serial
import subprocess
import pyautogui
import time
import threading
import sys
from datetime import datetime



ser = serial.Serial('COM4', 9600, timeout=1)

ser.write(b'LIGHT_ON\n')
ser.write(b'FAN_ON\n')