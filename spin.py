# -*- coding: utf8 -*-
'''
Simulate magnetization of one group of nuclear spins
solving the Bloch equation within a frame of reference rotating with w0

dM/dt = G*(M x B) + relax

M: magnetization
B: applied magnetic field, B_stat + B_rf == (B1x, B1y, B0+Bgz),
    static B_stat = (0,0,B0+Bgz)   (Bgz: gradient in z)
    oscill B_rf = (B1x,B1y,0)
G: gyromagnetic ratio
relax: T1, T2 relaxation terms
w0: Larmor frequency w0 = -G*B0
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import ode
# import warning


class spin:

    ''' spin main class. '''

    def __init__(self, M0=1, B0=3, w0=0, dw=0, tend=1e-4, T1=0.200, T2=0.600,
                 Minit=[0.6, 0, 0.8]):
        ''' constructor
        M0      equilibrium magnetization
        Minit   initial magnetization
        B0      static magnetic field (along $z$)
        w0  larmor frequency of spin
        dw  frequency of rot. frame of ref.
        T1      relaxation time of substance
        T2
        '''
    # gyromagnetic ratio of protons (¹H):
        self.g = 42.6e6  # Hz/Tesla
        self.M0 = M0
        self.B0 = B0
        if w0 == 0:
            self.w0 = -self.g*B0
        else:
            self.w0 = w0
        self.dw = dw
        self.tend = tend
        self.T1 = T1
        self.T2 = T2
        self.Minit = Minit
        self.ptype = ''

    def set_pulseseq(self, ptype, TR=0, TE=0):
        self.ptype = ptype
        self.TR = TR
        self.TE = TE

    def pulseseq(self, t):
        ''' define RF pulse sequences 
        
        TODO: implement pulses according to [1],[2]
            * Gradient Echo
            * Inversion Recovery
            * Spin Echo

            * Phase Constrast
            * Echo Planar Imaging (Echo Train Pulse Seq.)
            * Diffusion Imaging
        
            * what about gradients?
            -- imaging gradients:
            --- frequency-encoding gradients
            --- phase-encoding gradients
            --- slice selection gradients
            -- motion sensitizing gradients
            --- diffusion-weighting gradients
            --- flow-encoding gradients
            -- correction gradients
            --- concomitant-field correction gradients
            --- crusher gradients
            --- eddy-current compensation
            --- gradient moment nulling
            --- spoiler gradients
            --- twister gradients
            
        -------
        [1] Bernstein (2004), Handbook of MRI Pulse Sequences
        [2] Doorly (1997)
        '''

        if self.ptype == '':
            p = np.array([0, 0, 0])

        elif self.ptype == 'saturation_recovery':
            p = np.array([0, 0, 0])

        elif self.ptype == 'spin_echo':
            p = np.array([0, 0, 0])

        elif self.ptype == '':
            p = np.array([0, 0, 0])

        return p

    def solve(self, backend='vode', nsteps=1000, atol=1e-3):
        ''' solve with scipy.integrate.ode

        backend     vode, dopri5, dop853
        nsteps
        atol        required accuracy
        '''

        def rhs(t, y, self):
            ''' assemble rhs of Bloch equation
            arg: self, B

            B       effective magnetic oscil. field
            R       relaxation terms
            '''
            # self = arg[0]
            T2 = self.T2
            T1 = self.T1
            M0 = self.M0
            B0 = self.B0

            omega = -self.w0+self.dw
            Brot = np.array([0, 0, omega/self.g])
            B = np.array([0, 0, B0]) + self.pulseseq(t) - Brot
            # print(B[2])
            R = np.array([
                y[0]/T2,
                y[1]/T2,
                (y[2]-M0)/T1
                ])
            # print(self.g*np.cross(y, B) - R)

            return self.g*np.cross(y, B) - R

        # solver = ode(rhs).set_integrator(backend, nsteps=nsteps, atol=atol)

        ''' VAR 1 ##   automatic step size control '''
        if backend == 'vode':
            sol = []
            t = []
            solver = ode(rhs).set_integrator('vode', atol=atol)
            solver.set_initial_value(self.Minit, 0)
            solver.set_f_params(self)
            while solver.successful() and solver.t < self.tend:
                solver.integrate(self.tend, step=True)
                t.append(solver.t)
                sol.append(solver.y)
                print("%g/%g" % (solver.t, self.tend))
        elif backend == 'dopri5':
            ''' VAR 2 ## automatic step size with time grid '''
            sol = []
            t = []
            # t = np.linspace(0, self.tend, nsteps)
            dt = self.tend/nsteps
            solver = ode(rhs).set_integrator('dopri5', atol=atol)
            solver.set_initial_value(self.Minit, 0)
            solver.set_f_params(self)
            while solver.successful() and solver.t < self.tend:
                solver.integrate(solver.t+dt)
                t.append(solver.t)
                sol.append(solver.y)
                print("%g/%g" % (solver.t, self.tend))

        self.t = np.array(t)
        self.M = np.array(sol)
        return self.t, self.M

    def relaxation(self, dt=1):
        ''' calculate and plot T1/T2 relaxation during free precession, within a
        rotating frame of reference, with w == w_0
        t   [t0, tend] in ms
            dt  timestep
                freq.
        '''
        # import time

# T1, T2 relaxation
        t = np.linspace(0, self.tend, self.tend/dt, endpoint=True)
        self.MR = np.array(np.zeros((len(t), 3)))
        # E1 = np.exp(-t/self.T1)
        # E2 = np.exp(-t/self.T2)
        # R = np.array([E2, 0 0; 0, E2, 0; 0, 0, E1])
        # A = np.array([0, 0, self.M0*(1-E1)])
        self.MR[:, 0] = self.Minit[0]*np.exp(-t/self.T2)
        self.MR[:, 1] = self.Minit[1]*np.exp(-t/self.T2)
        self.MR[:, 2] = self.M0 + (self.Minit[2] - self.M0)*np.exp(-t/self.T1)

        plt.ion()
        fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # ax.axis([-1, 1, -1, 1])
        # ax.plot([0, 0], [0, 0], [-1,  1], '-.k')
        # ax.plot([-1, 1], [0, 0], [0, 0], '-.k')
        # ax.plot([0, 0], [-1, 1], [0, 0], '-.k')
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        # ax.set_zlabel('z')

        # skip = 10
        # for i in range(len(t[::skip])):
        #     ax.plot([0, self.M[i, 0]], [0, self.M[i, 1]], [0, self.M[i, 2]],
        #             '-<r')
        #     plt.draw()
        #     time.sleep(0.1)

        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        ax1.plot(t, self.MR[:, 0])
        ax1.set_xlabel('time in ms')
        ax1.set_ylabel('$M(t)$')
        ax1.set_title('T1 relaxation')
        ax2.plot(t, self.MR[:, 2])
        ax2.set_title('T2 relaxation')


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
    # s = spin(dw=1e6, tend=1e-2, T1=2e-3, T2=6e-3)
    s = spin(dw=0, tend=1, T1=2e-1, T2=6e-1)
    s.set_pulseseq('saturation_recovery', TE=1, TR=5)
    t, y = s.solve(backend='vode', nsteps=1000, atol=1e-6)

    plot_relax(t, y)
    # plot3Dtime(t, y)
