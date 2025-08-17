from __future__ import print_function
import sys
import subprocess
import logging
import socket
import flags 
FLAGS = flags.FLAGS

def make_logger(name):
    LOG_FORMAT = '{name}:{levelname} [{asctime}] {message}'

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(LOG_FORMAT, style='{'))
    logger.addHandler(ch)

    return logger

def get_host_ip():
    """get the host ip elegantly
    https://www.chenyudong.com/archives/python-get-local-ip-graceful.html
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip

def print_fn(log):
    if FLAGS.print:
        print(log)
        if FLAGS.flush_stdout:
            sys.stdout.flush()


def mkdir(folder_path):
    cmd = 'mkdir -p ' + folder_path
    ret = subprocess.check_call(cmd, shell=True)
    print_fn(ret)


def search_dict_list(dict_list, key, value):
    '''
    Search the targeted <key, value> in the dict_list
    Return:
        list entry, or just None 
    '''
    for e in dict_list:
        # if e.has_key(key) == True:
        if key in e:
            if e[key] == value:
                return e

    return None
