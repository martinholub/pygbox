from os import path as pth
from time import strftime
from glob import glob

def make_fpath( head, tail = None, ext = None, append_date = True):

    head = pth.abspath(head)
    if not tail:
        head, tail = pth.split(head)
        name, ext_ = pth.splitext(tail)
    elif tail.startswith(('+', '_')):
        head, tail_ = pth.split(head)
        name, ext_ = pth.splitext(tail_)
        name = name + tail[1:]

    if not ext:
        ext = ext_

    ext = '.' + ext if not ext.startswith('.') else ext

    if append_date:
        timestr = strftime("%Y%m%d_%H%M%S")
        name = name + "_" + timestr

    fpath = pth.join(head, name + ext)
    if pth.isfile(fpath):
        print(fpath + ": Warning:File already exists!")
    return fpath

def get_fpath(head, tail, ext, append_date = False):
    try:
        return glob(make_fpath(head, tail, ext, append_date = False)).pop()
    except IndexError as e:
        return []
