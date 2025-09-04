"""
Library of commonly used functions and modules for lensing on cosmic strings project.

:Authors:
    Davide Guerra (davide.guerra@uv.es)

"""
import warnings, os, sys
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
import numpy as np
import pylab as plt
import matplotlib.pyplot as plt
import sxs
from pycbc.waveform import get_fd_waveform, get_td_waveform, TimeSeries
from pycbc.psd.analytical import aLIGOaLIGOO3LowT1800545 # O3
from pycbc.psd.analytical import aLIGOAdVO4T1800545 # O4
import pycbc.noise
import pycbc.psd
import pycbc.inject
from pycbc.filter import sigma
from scipy.special import gamma
import mpmath as mpm
from scipy.special import fresnel

import matplotlib.pylab as pylab
#'axes.linewidth': 2,
params = {'legend.fontsize': 'medium',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif')