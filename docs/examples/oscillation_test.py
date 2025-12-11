import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import time

#  instead of trying to add all the odes manually, we define them in a function
#  using crnpy its easier to add odes automatically and solve them combined

def all_differential_equations(t, y, s_map):
    a = y[s_map['a']]
    b = y[s_map['b']]
    e = y[s_map['e']]
    c = y[s_map['c']]
    d = y[s_map['d']]
    
    o1 = y[s_map['o1']]
    o2 = y[s_map['o2']]
    o3 = y[s_map['o3']]
    o4 = y[s_map['o4']]
    o5 = y[s_map['o5']]
    
    k_osc = 1.0    
    k_compute = 1.0
    k_trash = 1.0   

    
    # o1 + o2 -> 2*o2
    # o2 + o3 -> 2*o3
    # ...
    # o5 + o1 -> 2*o1 (This makes it a ring)

    rate_o1_o2 = k_osc * o1 * o2
    rate_o2_o3 = k_osc * o2 * o3
    rate_o3_o4 = k_osc * o3 * o4
    rate_o4_o5 = k_osc * o4 * o5
    rate_o5_o1 = k_osc * o5 * o1
    
    # = a * b
    # o3 + a + b -> o3 + a + b + c
    rate_c_prod = k_compute * o3 * a * b
    # o3 + c -> o3 + trash1



    rate_c_decay = k_trash * o3 * c
    
    # d = e + c
    # o5 + e -> o5 + e + d
    rate_d_prod_e = k_compute * o4 * e
    # o5 + c -> o5 + c + d
    rate_d_prod_c = k_compute * o4 * c
    # o5 + d -> o5 + trash2
    rate_d_decay = k_trash * o4 * d
    dydt = np.zeros_like(y)
    
    dydt[s_map['a']] = 0
    dydt[s_map['b']] = 0
    dydt[s_map['e']] = 0
    
    # do1/dt = (production from o5) - (consumption by o2)
    # do2/dt = (production from o1) - (consumption by o3)
    # ..
    # do5/dt = (production from o4) - (consumption by o1)
    dydt[s_map['o1']] = rate_o5_o1 - rate_o1_o2
    dydt[s_map['o2']] = rate_o1_o2 - rate_o2_o3
    dydt[s_map['o3']] = rate_o2_o3 - rate_o3_o4
    dydt[s_map['o4']] = rate_o3_o4 - rate_o4_o5
    dydt[s_map['o5']] = rate_o4_o5 - rate_o5_o1
    
    # dc/dt = (production) - (decay)
    dydt[s_map['c']] = rate_c_prod - rate_c_decay
    
    # dd/dt = (production from e) + (production from c) - (decay)
    dydt[s_map['d']] = rate_d_prod_e + rate_d_prod_c - rate_d_decay
    
    dydt[s_map['trash1']] = rate_c_decay
    dydt[s_map['trash2']] = rate_d_decay
    
    return dydt

species_list = [
    'a', 'b', 'e',
    'c', 'd',     
    'o1', 'o2', 'o3',
    'o4', 'o5',     
    'trash1', 'trash2'
]
s_map = {name: i for i, name in enumerate(species_list)}
num_species = len(species_list)

y0 = np.full(num_species, 1e-10) # Start all at near-zero

y0[s_map['a']] = 3.0
y0[s_map['b']] = 4.0
y0[s_map['e']] = 5.0

# final answers:
# c = a * b = 12
# d = e + c = 5 + 12 = 17

y0[s_map['o1']] = 1.0
y0[s_map['o2']] = 1.0

t_span = [0, 200] # Simulation time
t_eval = np.linspace(t_span[0], t_span[1], 1000)

sol = solve_ivp(
    fun=all_differential_equations, 
    t_span=t_span, 
    y0=y0, 
    t_eval=t_eval,
    method='BDF', # 'BDF' is good for stiff systems like CRNs
    args=(s_map,)
)
print("Simulation complete.")

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(sol.t, sol.y[s_map['o3']], label='o3 (Clock for Phase 1: c=a*b)', lw=2)
plt.plot(sol.t, sol.y[s_map['o4']], label='o4 (Clock for Phase 2: d=e+c)', lw=2)
plt.title('Oscillator Clock Pulses')
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(sol.t, sol.y[s_map['c']], label='c (output of a*b)', lw=2)
plt.plot(sol.t, sol.y[s_map['d']], label='d (output of e+c)', lw=2)

plt.axhline(y=12.0, color='blue', linestyle='--', label='Expected c=12')
plt.axhline(y=17.0, color='orange', linestyle='--', label='Expected d=17')

plt.title('Sequential Computation')
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

