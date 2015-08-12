# -*- coding: utf8 -*-
'''
Simulate magnetization of one group of nuclear spins "0D"
solving the Bloch equation within a frame of reference rotating with w_rf

dM/dt = g*(M x B) + relax

M: magnetization
B: applied magnetic field = B_0 + B_RF + B_G
g: gyromagnetic ratio
relax: T1, T2 relaxation terms '''

# TODO: [ ]  spin echo sequence: 90y - TE/2 - 180x - TE - 180x - ..
#       [ ]  compute MRI signal
#       [ ]  compare to analytical solution
#       [ ]            and matrix formulism

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import ode
from utils import progressbar
# import warning


# class spin:

#     ''' spin class
#         (add some methods later?) '''

#     def __init__(self, M0=1, T1=0.200, T2=0.600, Minit=[0, 0, 1]):
#         ''' constructor
#         M0      equilibrium magnetization
#         Minit   initial magnetization
#         T1      relaxation time of substance
#         T2
#         '''
#     # gyromagnetic ratio of protons (¹H):
#         self.gm = 42.6e6  # Hz/Tesla
#         self.M0 = 1
#         self.T1 = T1
#         self.T2 = T2
#         self.Minit = Minit


def pulseseq(t, s, params, it):
    ''' compute contribution to magnetic field `Beff(t)` at time `t`
    due to static gradient `Bg`, RF pulse `Brf` and/or gradient pulse `Brfg`
    return: B'(t) = Bg + Brf(t) + Brfg(t)
                  = [Bx, By, Bz]
    '''
    B1 = params.get('amp')
    w0 = params.get('w0')
    TR = params.get('TR')
    TE = params.get('TE')
    pseq = params.get('pseq')
    dphi = params.get('dephase')  # dephase angle in rad: by how much will
#    magnetization be dephased between P1 and P2 ?

    if pseq == 'flip90':
        tp = np.pi/(2*B1*s['gm'])

        dt = TE/2
        dB = dphi/s['gm']/dt

        if np.mod(t, TR) <= tp:  # 90° flip
            Bp = B1*np.array([np.cos(w0*t), 0, -dB])
        else:
            Bp = np.array([0, 0, -dB])

    elif pseq == 'continuous':
        Bp = B1*np.array([np.cos(w0*t), 0, 0])
    elif pseq == 'pulsed':
        if np.mod(t, TR) < TE:  # echo!
            Bp = B1*np.array([np.cos(w0*t), 0, 0])
        else:
            Bp = np.array([0, 0, 0])

    elif pseq == 'spinecho':
        ''' - one pulse of length tp flips M by pi/2
            - magnetization is dephased due to field inhomogeinities
                    (specify angle in rad!!)
            - refocus pulse after \tau -> pi flip
            - phase coherence restored after 2\tau
            cf. Slichter
        '''
        # pulse duration pi flip
        tp = np.pi/(2*B1*s['gm'])
        dt = TE/2
        dB = dphi/s['gm']/dt

        if np.mod(t, TR) <= tp:  # 90° flip
            Bp = B1*np.array([np.cos(w0*t), 0, -dB])
        elif np.mod(t, TR) <= tp + TE/2:  # dephase!
            Bp = np.array([0, 0, -dB])
        elif np.mod(t, TR) <= TE/2+3*tp:  # 180° flip
            Bp = B1*np.array([np.cos(w0*t), 0, -dB])
        else:
            Bp = np.array([0, 0, -dB])
    else:
        Bp = np.array([0, 0, 0])
    return Bp


def bloch(s, tend=1, nsteps=1000, backend='vode', pulse_params={},
          B0=3, dw_rot=0, dw_rf=0, rtol=1e-6):
    ''' solve Bloch equations for spin `s` in the ROTATING FRAME OF REFERENCE
        rotating with the Larmor frequency plus a shift `dw_rot` (default: 0)
        setting dw_rot = None (-> -w0) corresponds to the laboratory frame.

        dw_fr: frequency shift for off resonance excitation
    '''

    w0 = -s['gm']*B0
# RF freq in rotating frame of reference is  `w - w_fr`,
# so just the "off resonance" freq (=w_0-w_rf) plus the
# difference in frequency between wf_fr and w_0
    if dw_rot is None:
        dw_rot = -w0
    pulse_params['w0'] = dw_rot + dw_rf

    def rhs(t, y, s, pulse_params, B0, w0, dw_rot, it):
        B = np.array([0, 0, B0])                # static
        B = B + pulseseq(t, s, pulse_params, it)    # RF
        B = B + np.array([0, 0, (w0+dw_rot)/s['gm']])  # rotating frame with w+dw
        R = np.array([y[0]/s['T2'], y[1]/s['T2'], (y[2]-s['M0'])/s['T1']])  # relax
        return s['gm']*np.cross(y, B) - R

    ''' VAR 1 ##   automatic step size control '''
    it = 1
    sol = []
    t = []
    dt = tend/nsteps
    solver = ode(rhs).set_integrator(backend, rtol=rtol)
    solver.set_initial_value(s['Minit'], 0)
    solver.set_f_params(s, pulse_params, B0, w0, dw_rot, it)
    while solver.successful() and solver.t < tend:
        # works only with vode!! not recommended:
        # solver.integrate(tend, step=True)
        solver.integrate(solver.t+dt)
        t.append(solver.t)
        sol.append(solver.y)
        it = it + 1
        progressbar(solver.t, tend, 'solve')

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

    for i in range(0, len(t), skip):
        ax.plot([0, M[i, 0]], [0, M[i, 1]], [0, M[i, 2]],
                '-<r')
        # print('%i \t t = %g s' % (i, t[i]))
        progressbar(t[i], t.max(), s='plot')
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


def plot_pulse(t, M, params, s):
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    # plot magnetization components
    ax1.plot(t, M)
    ax1.legend(('$M_x$', '$M_y$', '$M_z$'))
    ax1.set_xlabel('time in ms')
    ax1.set_ylabel('$M$')
    ax1.set_title('Magnetization')
    plt.draw()

    # plot pulse train
    pseq = params.get('pseq')
    if pseq == 'spinecho' or pseq == 'flip90':
        TE = params.get('TE')
        TR = params.get('TR')
        B1 = params.get('amp')
        N = int(np.ceil(t[-1]/TR))   # number of periods
        tp = np.pi/(2*B1*s['gm'])
        # draw polygone of one period:
        if pseq == 'spinecho':
            p1 = [0, 1, 1, 0, 0, 1, 1, 0, 0]
            tp1 = np.array([0, 0, tp, tp, tp+TE/2, tp+TE/2, TE/2+3*tp,
                            TE/2+3*tp, TR])
        elif pseq == 'flip90':
            p1 = [0, 1, 1, 0, 0]
            tp1 = np.array([0, 0, tp, tp, TR])
        p, tp = [], []
        for i in range(N):
            tp.extend(tp1+i*TR)
            p.extend(p1)

        ax2.plot(tp, p)
        ax2.set_ylim([-0.2, 1.2])
        ax1.set_xlim([0, t.max()])
        plt.draw()


if __name__ == '__main__':

    B0 = 3
# spin dict
    s = {
        'M0': 1,
        'T1': 0.100,
        'T2': 0.600,
        'Minit': [0, 0, 1],
        'gm': 42.6e6
        }
# pulse dict
    pulse = {
        'TE': 0.050,
        'TR': 1.000,
        'amp': 1.75e-5,         # B1 = 1.75e-5 taken from Yuan1987
        'pseq': 'spinecho',
        'dephase': .1
        }
    w0 = s['gm']*B0
    nsteps = 1e4
    # t, M = bloch(s, tend=0.2, backend='dopri5', pulse_params=pulse, dw_rot=0,
    #              dw_rf=0, rtol=1e-6, nsteps=nsteps, B0=B0)
    # Mc = M[:, 0] + 1j*M[:, 1]

# MANY SPINS EXPERIMENT
    N = 20
    r = 2*np.random.rand(N) - 1
    dw_off = r*100   # frequency shift between +-100 Hz
    dphi = r*B0*0.5  # max angle (rad) of dephasing during TE/2

    var = dphi  # dw_off

    M = []
    i = 0
    Mc = np.zeros((nsteps, N), dtype=complex)
    for x in var:
        print('\nrun %i/%i \t shift %.2f' % (i+1, len(var), x))
        pulse['dephase'] = x
        t, H = bloch(s, tend=0.2, backend='dopri5', pulse_params=pulse, dw_rot=0,
                     dw_rf=0, rtol=1e-6, nsteps=1e5, B0=B0)
        M.append(H)
        Mc[:, i] = H[:, 0] + 1j*H[:, 1]
        i += 1

    M = np.array(M)
# integrate Mt to get signal

    def plot_cplx(t, Mc):
        plt.figure()
        plt.ion()
        plt.plot(t, np.real(Mc), '-', t, np.imag(Mc), ':')

    def plot_signal(t, M):
        signal = np.sum(M, 0)[:, 0:2]
        plt.figure()
        plt.ion()
        plt.plot(t, signal)
        plt.plot(t, signal[:, 0]+signal[:, 1])
        plt.legend(('x', 'y', 'sum'))

        
        


# *** BENCHMARK: COMPARE ODE BACKENDS
    # Mloop = []
    # for be in ['vode', 'lsoda', 'dopri5', 'dop853']:
    #     t, M = bloch(s, tend=0.1, backend=be, pulse_params=pulse, dw_rot=0,
    #                  dw_rf=0, rtol=1e-6, nsteps=1e5, B0=B0)
    #     Mloop.append(M)


# *** EXAMPLE:  continuous excitation, M -> 2pi turn
    # pulse = {'TE': 20, 'TR': 50, 'amp': 1, 'pseq': 'continuous'}
    # t1 = 2*np.pi/s.gm/1
    # t, M = bloch(s, tend=t1, backend='vode', pulse_params=pulse, dw_rot=0,
    #              rtol=1e-6, nsteps=1e3, B0=B0)

# *** EXAMPLE: free precession, relaxed
    # pulse = {'pseq': 'none'}
    # s = spin(Minit=[0.7, 0, 0.8])
# laboratory frame (insane)
    # t, M = bloch(s, backend='dopri5', tend=0.01, nsteps=1e4,
    #              pulse_params=pulse, dw_rot=None, rtol=1e-3, B0=3)
# rotating reference frame (sensible)
    # t, M = bloch(s, backend='vode', nsteps=1e3, pulse_params=pulse,
    #              dw_rot=100, rtol=1e-6, B0=3)

# ** BENCHMARK **  dopri5 (RK45): 1 loops, best of 3: 346 ms per loop
#                    vode (ABF): 10 loops, best of 3: 77.1 ms per loop
#       command: %timeit %run mri0D.py

    # plot_relax(t, y)
    # plot3Dtime(t, y)
