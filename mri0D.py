# -*- coding: utf8 -*-
'''
Simulate magnetization of one group of nuclear spins "0D"
solving the Bloch equation within a frame of reference rotating with w_rf

dM/dt = g*(M x B) + relax

M: magnetization
B: applied magnetic field = B_0 + B_RF + B_G
g: gyromagnetic ratio
relax: T1, T2 relaxation terms
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import ode
# import warning


class spin:

    ''' spin class
        (add some methods later?) '''

    def __init__(self, M0=1, T1=0.200, T2=0.600, Minit=[0, 0, 1]):
        ''' constructor
        M0      equilibrium magnetization
        Minit   initial magnetization
        T1      relaxation time of substance
        T2
        '''
    # gyromagnetic ratio of protons (¹H):
        self.g = 42.6e6  # Hz/Tesla
        self.M0 = 1
        self.T1 = T1
        self.T2 = T2
        self.Minit = Minit


def pulseseq(s, t, pseq, Amp, w0, TR=0, TE=0):
    ''' compute contribution to magnetic field `Beff(t)` at time `t`
    due to static gradient `Bg`, RF pulse `Brf` and/or gradient pulse `Brfg`
    return: B'(t) = Bg + Brf(t) + Brfg(t)
                  = [Bx, By, Bz]
    '''
    if pseq == 'flip90':
        # nothing
        print('90')
    elif pseq == 'continuous':
        Bp = Amp*np.array([np.cos(w0,t), -np.sin(w0,t), 0])
    else:
        Bp = np.array([0, 0, 0])
    return Bp


def bloch(s, tend=1, nsteps=1000, backend='vode', pseq=''):
    ''' solve Bloch equations for spin `s` '''

    def rhs(t, y, *args):
        

