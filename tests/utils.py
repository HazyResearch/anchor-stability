import subprocess

import numpy as np


def clean_files(filename):
    try:
        subprocess.check_output("rm -f %s" % filename, shell=True)
    except OSError as e:
        print(e)
        pass
