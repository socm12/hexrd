"""Oscillation stage parameters"""
from __future__ import print_function

import numpy as np

class OscillationStage(object):

    def __init__(self, tvec, chi):
        self.tvec = np.atleast_1d(tvec).flatten()
        self.chi = chi
