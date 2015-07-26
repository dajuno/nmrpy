# -*- coding: utf8 -*-
'''
Simulate magnetization of one group of nuclear spins
solving bloch equation
'''

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import ode
import warnings


class spin:

    """ spin main class. """

    def __init__(self,M0,B0,omega,gm,T1,T2):
        """ initialize
        M0      initial magnetization
        B0      static magnetic field (along $z$)
        omega   larmor frequency of spin
        gm      gyromagnetic constant \gamma
        T1      relaxation constant of substance
        T2
        """
        self.M0 = M0
        self.B0 = B0
        self.omega = omega
        self.gm = gm
        self.T1 = T1
        self.T2 = T2

        

    def pulse_seq(self):
        """ define RF pulse sequences """

    def solve(self,backend='dopri5',nsteps=1,atol=1e-3):
        """ solve with scipy.integrate.ode

        backend     dopri5, vode, dop853
        nsteps      should be 1 for adaptive step size
        atol        required accuracy
        """


        def rhs(t,y,arg):
            """ assemble rhs of Bloch equation 
            
                    effective magnetic oscil. field 
                    relaxation terms
            """

            
        solver = ode(rhs).set_integrator(backend, nsteps=nsteps, atol=atol)
        solver.set_initial_value(M0, 0)
        solver.set_f_params((X,Y))
# suppress Fortran-printed warning
        # solver._ingerator.iwork[2] = -1
        # warnings.filterwarnings("ignore", category=UserWarning)
                

########## OLD :::: DELETE BELOW ############

def F(t,y,arg): #gm,M,B0,B1,omega):
    Beff = arg[0]
    Frelax = arg[1]
    # gm = args[0]; omega = args[1]; B0 = args[2]; B1 = args[3]
    # M = y
    # Beff = np.zeros(M.size)
    # Beff[2] = gm*B0-omega
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
omega = 100.
omega = gm*B0   # resonance !

T1 = 1e1
T2 = 5e-0
M0 = .6

Frelax = []
Frelax = [T2,T1,M0] # T2 T1 M0

Beff = np.array([gm*B1, 0, gm*B0-omega])
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


