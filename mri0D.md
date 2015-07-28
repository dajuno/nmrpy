# mri0D.py

* `class spin` with attributes 
    * physical const. gamma, T1, T2
    * equilibrium magnetization M0
    * initial magnetization Minit (=?M0)

* define pulse sequences as functions of time, including RF and gradient pulses,
    * spin echos
    * inversion recovery
    * phase contrast
    * etc

* function `bloch(s)` that solves the Bloch equations for an object `s` of class
`spin` 
    * parameters: tend, nsteps, ode backend (vode, dopri5), pulse sequence
    * function `rhs(t, y, *args)`, 
      args: B_eff(t), spin (T1,T2,M0 -> relaxation)

* plotting functions
    * T1, T2 relaxation diagrams
    * magnetization vector 2D
