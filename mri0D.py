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
        self.gm = 42.6e6  # Hz/Tesla
        self.M0 = 1
        self.T1 = T1
        self.T2 = T2
        self.Minit = Minit


def pulseseq(t, params):
    ''' compute contribution to magnetic field `Beff(t)` at time `t`
    due to static gradient `Bg`, RF pulse `Brf` and/or gradient pulse `Brfg`
    return: B'(t) = Bg + Brf(t) + Brfg(t)
                  = [Bx, By, Bz]
    '''
    Amp = params.get('amp')
    w0 = params.get('w0')
    TR = params.get('TR')
    TE = params.get('TE')
    pseq = params.get('pseq')

    if pseq == 'flip90':
        # nothing
        print('90')
    elif pseq == 'continuous':
        if CONTINUE HERE
        Bp = B1*np.array([np.cos(w0*t), 0, 0])
    else:
        Bp = np.array([0, 0, 0])
    return Bp


def bloch(s, tend=1, nsteps=1000, backend='vode', pulse_params={},
          B0=3, dw_rot=0, dw_rf=0, atol=1e-6):
    ''' solve Bloch equations for spin `s` '''

    w0 = -s.gm*B0
    pulse_params['w0'] = w0 + dw_rf

    def rhs(t, y, s, pulse_params, B0, w0, dw_rot):
        B = np.array([0, 0, B0])                # static
        B += pulseseq(t, pulse_params)    # RF
        B += np.array([0, 0, (w0+dw_rot)/s.gm])   # rotating frame with `w+dw`
        R = np.array([y[0]/s.T2, y[1]/s.T2, y[2]-s.M0])  # relax
        return s.gm*np.cross(y, B) - R

    ''' VAR 1 ##   automatic step size control '''
    if backend == 'vode':
        sol = []
        t = []
        solver = ode(rhs).set_integrator('vode', atol=atol)
        solver.set_initial_value(s.Minit, 0)
        solver.set_f_params(s, pulse_params, B0, w0, dw_rot)
        while solver.successful() and solver.t < tend:
            solver.integrate(tend, step=True)
            t.append(solver.t)
            sol.append(solver.y)
            print("%g/%g" % (solver.t, tend))
    elif backend == 'dopri5':
        ''' VAR 2 ## automatic step size with time grid '''
        sol = []
        t = []
        # t = np.linspace(0, self.tend, nsteps)
        dt = tend/nsteps
        solver = ode(rhs).set_integrator('dopri5', atol=atol)
        solver.set_initial_value(s.Minit, 0)
        solver.set_f_params(s, pulse_params, B0, w0, dw_rot)
        while solver.successful() and solver.t < tend:
            solver.integrate(solver.t+dt)
            t.append(solver.t)
            sol.append(solver.y)
            print("%g/%g" % (solver.t, tend))

    return np.array(t), np.array(sol)


def plot3Dtime(t, M, skip=10):
    import time
    plt.ion()
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.axis([-1, 1, -1, 1])
    ax.plot([0, 0], [0, 0], [-1,  1], '-.k')
    ax.plot([-1, 1], [0, 0], [0, 0], '-.k')
    ax.plot([0, 0], [-1, 1], [0, 0], '-.k')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    for i in range(len(t[::skip])):
        ax.plot([0, M[i, 0]], [0, M[i, 1]], [0, M[i, 2]],
                '-<r')
        plt.draw()
        time.sleep(0.05)


def plot_relax(t, M):
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    Mt = np.sqrt(M[:, 0]**2 + M[:, 1]**2)
    ax1.plot(t, Mt)
    ax1.set_xlabel('time in ms')
    ax1.set_ylabel('$|M|$')
    ax1.set_title('T1 relaxation')
    ax2.plot(t, M[:, 2])
    ax2.set_title('T2 relaxation')


if __name__ == '__main__':
    s = spin
    t, y = s.solve(backend='vode', nsteps=1000, atol=1e-6)

    # plot_relax(t, y)
    plot3Dtime(t, y)
