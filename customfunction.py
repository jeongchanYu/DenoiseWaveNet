import os
import datetime
import time

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Issue: Creating directory. ' +  directory)

def load_path():
    path = os.path.join(os.path.dirname(__file__))
    if path == "":
        path = "."
    return path

def process_time(func):
    def wrapper():
        start = time.time()
        func()
        print("Processing time :", datetime.timedelta(seconds=time.time()-start))
    return wrapper
