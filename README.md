# pycrn
A python version of crn++
This project is a Python counterpart to the CRN++ Mathematica package originally developed by David Soloveichik (see: https://github.com/marko-vasic/crnPlusPlus ).

# How are we simulating ODES

First we sort species, the purpose is to map each specie with its concentration
here smap stores according to position like {'A':0, 'B':1, 'C':2, ..., 'trash4':19}
After filling intial values and inputs provided we start our oscillator.
We add the odes to solve with solve_ivp using mass action.
```
A + B → 2C
dA/dt = -rate
dB/dt = -rate
dC/dt = +2*rate
```
We are using BDF solver which moves slow, good for stiff systems takes tiny steps to
remain stable.



