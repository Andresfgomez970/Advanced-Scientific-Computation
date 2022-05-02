In this directory a set of routines we made. In the following lines I shortly
  specify the purpose of each one.

# isign.ccp ising.h

General functions needed to modelate the dynamical evolution of a spin square
  grid system.

# isign_temp.cpp

This will the evolution of the spins, the parameters for the system desired to
  be modelated must be specified.

Compiled as: 
g++ ising.cpp ising_temp.cpp -o out

It recives 3 parameters, as an example we have

./out 2 10000 1 1

the parameters means: 

L = 2, n_steps = 1000, periodic == 1, Energy moments == 1

# Analysis.py

This plots the cv graphs, the files constructed are the one with 640M of points
    and the L specified in their names. The 50 refer to the density of points
    that is 50 by default.

# mc_integration.py 

general routine constrcuted to integrate a Nth dimensional function, with the
  convergence desired; examples of integral  are given in the main of it.


# potentail.py

The example of the potential being integrated is given in potential.py; this
  was done before general Nth dimensinal integration. Simply run 
  python potential.py