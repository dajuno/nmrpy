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
import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import ode, odeint
# import warning


class spin:

    """ spin main class. """

    def __init__(self, M0=1, B0=3, w0=0, dw=0, tend=1, T1=0.200, T2=0.600,
                 Minit=[0.6, 0, 0.8]):
        """ constructor
        M0      equilibrium magnetization
        Minit   initial magnetization
        B0      static magnetic field (along $z$)
        w0  larmor frequency of spin
        dw  frequency of rot. frame of ref.
        T1      relaxation time of substance
        T2
        """
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

    def pulse_seq(self, t):
        """ define RF pulse sequences """
        return np.array([0, 0, 0])

    def solve(self, backend='vode', nsteps=100, atol=1e-3):
        """ solve with scipy.integrate.ode

        backend     vode, dopri5, dop853
        nsteps
        atol        required accuracy
        """

        def rhs(t, y, self):
            """ assemble rhs of Bloch equation
            arg: self, B

            B       effective magnetic oscil. field
            R       relaxation terms
            """
            # self = arg[0]
            T2 = self.T2
            T1 = self.T1
            M0 = self.M0
            B0 = self.B0

            omega = self.w0+self.dw
            Brot = np.array([0, 0, omega/self.g])
            B = np.array([0, 0, B0]) + self.pulse_seq(t) - Brot
            R = np.array([
                y[0]/T2,
                y[1]/T2,
                (y[2]-M0)/T1
                ])
            print(self.g*np.cross(y, B) - R)

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

        return t, sol

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
        time.sleep(0.1)


def plot_relax(t, M):
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.plot(t, M[:, 0], label='$M_x$')
    ax1.plot(t, M[:, 1], label='$M_y$')
    ax1.legend()
    ax1.set_xlabel('time in ms')
    ax1.set_ylabel('$M(t)$')
    ax1.set_title('T1 relaxation')
    ax2.plot(t, M[:, 2])
    ax2.set_title('T2 relaxation')

if __name__ == '__main__':
    s = spin(dw=100)
    t, y = s.solve(backend='dopri5', atol=1e-3)

    # plot_relax(t, y)
    # plot3Dtime(t, y)



# ######### OLD :::: DELETE BELOW ############
"""
def F(t,y,arg): #gm,M,B0,B1,w):
    Beff = arg[0]
    Frelax = arg[1]
    # gm = args[0]; w = args[1]; B0 = args[2]; B1 = args[3]
    # M = y
    # Beff = np.zeros(M.size)
    # Beff[2] = gm*B0-w
    # Beff[0] = gm*B1    
    if not Frelax:
        return np.cross(y,Beff)
    else:
        T2 = Frelax[0]
        T1 = Frelax[1]
        M0 = Frelax[2]
        yrel = np.array([-y[0]/T2, -y[1]/T2, -(y[2]-M0)/T1])
        return np.cross(y,Beff) + yrel


# In[3]:

t0 = 0
t1 = 20.
gm = 200 #.66e4            # Bloch (1946)
B1 = 10.  # in gauss
B0 = 5.   
w = 100.
w = gm*B0   # resonance !

T1 = 1e1
T2 = 5e-0
M0 = .6

Frelax = []
Frelax = [T2,T1,M0] # T2 T1 M0

Beff = np.array([gm*B1, 0, gm*B0-w])
Bnorm = sp.linalg.norm(Beff)
Minit = np.array([0.5, 0.0, 1.])


# In[4]:

# plt.ion()
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot([M0[0]],[M0[1]],[M0[2]],'o')
# ax.axis([-1,1,-1,1])
# plt.draw()


# In[5]:

#plt.close('all')


# In[ ]:

#backend = 'vode'
backend = 'dopri5'
#backend = 'dop853'

solver = ode(F).set_integrator(backend, nsteps=1,atol=1e-3)
solver.set_initial_value(Minit, t0).set_f_params((Beff,Frelax))
# suppress Fortran-printed warning
solver._integrator.iwork[2] = -1

sol = []
warnings.filterwarnings("ignore", category=UserWarning)
it = 0


plt.ion()
fig = plt.figure()
ax = fig.gca(projection='3d')
#ax.plot([M0[0]],[M0[1]],[M0[2]],'o')
ax.axis([-1,1,-1,1])
ax.plot([0,0],[0,0],[-1, 1],'-.k')
ax.plot([-1,1],[0,0],[0,0],'-.k')
ax.plot([0,0],[-1,1],[0,0],'-.k')
ax.plot([0,Beff[0]/Bnorm],[0,Beff[1]/Bnorm],[0,Beff[2]/Bnorm],'-<r')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

while solver.t < t1:
    it += 1
    solver.integrate(t1, step=True)
    sol.append([solver.t, solver.y])
    if np.mod(it,100) == 0:
        print("%g" % solver.t)
        Mt = solver.y
        ax.plot([0,Mt[0]],[0,Mt[1]],[0,Mt[2]], 'b.')
        plt.draw()
warnings.resetwarnings()


# In[101]:

Mtmp = []
t = []
for x in sol:
    tmp = x
    t.append(x[0])
    Mtmp.append(x[1])
t = np.array(t)
M = np.array(Mtmp)
print(M.shape)


# In[104]:

# plt.ion()
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# #ax.plot([M0[0]],[M0[1]],[M0[2]],'o')
# ax.axis([-1,1,-1,1])
# ax.plot(M[0::5000,0],M[0::5000,1],M[0::5000,2], 'b.-')
# ax.plot([0,0],[0,0],[-1, 1],'-.k')
# ax.plot([-1,1],[0,0],[0,0],'-.k')
# ax.plot([0,0],[-1,1],[0,0],'-.k')

# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# plt.draw()


# # In[89]:

# #plt.close('all')

"""
