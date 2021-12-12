# -*- coding: utf-8 -*-

import base64 as _base64
import fcntl as _fcntl
import mimetypes as _mimetypes
import os.path as _ospath
import struct as _struct

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import sklearn.decomposition
import sklearn.preprocessing
import sklearn.cluster

from elice_utils import EliceUtils  # isort:skip

elice_utils = EliceUtils()

def display_digits(X, title):
    plt.clf()
    plt.figure(figsize=(4, 4))
    plt.imshow(X, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title(title)

    plt.savefig("image.svg", format="svg")
    elice_utils.send_image("image.svg")