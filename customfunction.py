import datetime
import time
import os


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


# for WGP Server
def write_plot_file(filename, index, value):
    with open(filename, 'a') as f:
        f.write("{{x:{}, y:{}}},".format(index, value))


def clear_plot_file(filename):
    with open(filename, 'w') as f:
        pass


def read_path_list(dirname, extention=""):
    try:
        return_list = []
        filenames = os.listdir(dirname)
        for filename in filenames:
            full_filename = os.path.join(dirname, filename)
            if os.path.isdir(full_filename):
                return_list.extend(read_path_list(full_filename, extention))
            else:
                ext = os.path.splitext(full_filename)[-1][1:]
                if extention == "" or ext == extention:
                    return_list.append(full_filename)
        return_list.sort()
        return return_list
    except PermissionError:
        pass


def compare_path_list(dirname1, dirname2, extention=""):
    list1 = read_path_list(dirname1, extention)
    list2 = read_path_list(dirname2, extention)
    for i in range(len(list1)):
        list1[i] = list1[i].replace(dirname1, "")
    for i in range(len(list2)):
        list2[i] = list2[i].replace(dirname2, "")
    list1.sort()
    list2.sort()
    if list1 == list2:
        return True
    else:
        return False
