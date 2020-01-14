import subprocess

import numpy as np


def clean_files(filename):
    try:
        subprocess.check_output("rm -f %s" % filename, shell=True)
    except OSError as e:
        print(e)
        pass


def load_emb(filename):
    f = open(filename, "r")
    dat = [_.strip() for _ in f]
    f.close()
    if len(dat[0].split()) == 2:
        dat = dat[1:]
    dim = len(dat[0].split()) - 1
    m = np.zeros((len(dat), dim))
    for i, _ in enumerate(dat):
        d = _.split()
        w = d[0]
        v = d[1:]
        m[i] = v
    return m
