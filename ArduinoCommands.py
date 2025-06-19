import serial
import subprocess
import pyautogui
import time
import threading
import sys
from datetime import datetime

from utility import is_similar_command

""" 
import difflib

def is_similar_command(text, target, threshold=0.75):
    ratio = difflib.SequenceMatcher(None, text, target).ratio()
    return ratio >= threshold

"""


#presentation_command ='"C:\Program Files\LibreOffice\program\soffice.exe" --show Resoluciondepset6.pptx'

presentation_process = None

command_lock = threading.Lock()
ser = serial.Serial('COM3', 9600, timeout=1)


log_file = open("command_log.txt", "a")

POWEROFF_COMMAND = "artemis apagate"
LEDON_COMMAND = "artemis enciende led"
LEDOFF_COMMAND = "artemis apaga led"
FANON_COMMAND = "artemis enciende led"
FANOFF_COMMAND = "artemis apaga led"

OPEN_PRESENTATION = "Artemis pon la presentacion"
CLOSE_PRESENTATION = "Artemis cierra la presentacion"
OPEN_CODE = "code ."

NEXT_PRESENTATION_SLIDE1 = "Artemis pasa a la Siguiente "
NEXT_PRESENTATION_SLIDE2 = "Artemis continua a la Siguiente "

log_file = open("voice_log.txt", "a")

def threaded_command(func):
    def wrapper(*args, **kwargs):
        if command_lock.acquire(blocking=False):
            def run():
                try:
                    func(*args, **kwargs)
                finally:
                    command_lock.release()
            threading.Thread(target=run).start()
        else:
            print("⚠️ Command ignored: another command is running.")
    return wrapper

presentation_command = [
                r"C:\Program Files\LibreOffice\program\soffice.exe", 
                "--show", 
                "Resoluciondepset6.pptx"
            ]


def open_presentation():
    global presentation_process
    print("📂 Abriendo presentación...")
    presentation_process = subprocess.Popen(presentation_command)
    time.sleep(5) 

# Navigation commands
@threaded_command
def next_slide():
    print("➡️ Siguiente diapositiva")
    pyautogui.press("right")

@threaded_command
def previous_slide():
    print("⬅️ Diapositiva anterior")
    pyautogui.press("left")


def handle_command(text):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {text}"
    
    print("Recieved:", text)
    log_file.write(log_line + "\n")
    log_file.flush()
    """ 
    if not command_lock.acquire(blocking=False):
        print("⚠️ Command ignored: another command is running.")
        return """

    if text =="LIGHT_OFF":
        print("👋 LED OFF")
        ser.write(b'LIGHT_OFF\n') 
        return   

    if text=="LIGHT_ON":
        print("LED ON")
        ser.write(b'LIGHT_ON\n')
        return
    
    if is_similar_command(text, LEDON_COMMAND):
        print("👋 LED ON")
        #log_file.write(f"[{timestamp}] SYSTEM: LED TURNED ON.\n")
        ser.write(b'LIGHT_ON\n')
    elif is_similar_command(text, LEDOFF_COMMAND):
        print("👋 LED OFF")
        #log_file.write(f"[{timestamp}] SYSTEM: LED TURNED OFF.\n")
        ser.write(b'LIGHT_OFF\n')
    elif is_similar_command(text,FANON_COMMAND):
        print()
    elif is_similar_command(text,FANOFF_COMMAND):
        print()
    elif is_similar_command(text,OPEN_PRESENTATION):
        print("👋 Presentacion puesta")
        open_presentation()  

    elif is_similar_command(text,NEXT_PRESENTATION_SLIDE1):
        print("👋 Presentacion si")
        next_slide()
    elif is_similar_command(text,NEXT_PRESENTATION_SLIDE2):
        next_slide()
    elif is_similar_command(text,FANON_COMMAND):
        print()
    elif is_similar_command(text, POWEROFF_COMMAND):
        print("👋 Artemis shutting down. Bye bye")
        #log_file.write(f"[{timestamp}] SYSTEM: Artemis shutdown triggered.\n")
        log_file.close()
        sys.exit(0)

