import numpy as np
import matplotlib.pyplot as plt
import time

from tqdm import tqdm

from pycrn import *
    

x, x0, y0, x1, y1 = 4, 2, 5, 7, 15
expected_result = 9.0

def dual_encode(val): 
    return (max(0, val), max(0, -val))

def dual_decode(p_val, n_val): 
    return p_val - n_val


x_p, x_n = dual_encode(x)
x0_p, x0_n = dual_encode(x0)
y0_p, y0_n = dual_encode(y0)
x1_p, x1_n = dual_encode(x1)
y1_p, y1_n = dual_encode(y1)

crn = CRNCompiler(oscillator_step=2)


crn.add_phase([
    ('add', ('x_p', x_p), ('x0_n', x0_n), 'dx_p'),
    ('add', ('x_n', x_n), ('x0_p', x0_p), 'dx_n'),

    ('add', ('x1_p', x1_p), ('x0_n', x0_n), 'span_p'),
    ('add', ('x1_n', x1_n), ('x0_p', x0_p), 'span_n'),

    ('add', ('y1_p', y1_p), ('y0_n', y0_n), 'dy_p'),
    ('add', ('y1_n', y1_n), ('y0_p', y0_p), 'dy_n'),
])

crn.add_phase([
    ('sub', 'dx_p', 'dx_n', 'dx_real'),
    ('sub', 'span_p', 'span_n', 'span_real'),
    ('sub', 'dy_p', 'dy_n', 'dy_real'),
])

crn.add_phase([
    ('div', 'dx_real', 'span_real', 'frac_real')
])

crn.add_phase([
    ('mul', 'frac_real', 'dy_real', 'offset_real')
])

crn.add_phase([
    ('add', ('y0_p', y0_p), 'offset_real', 'result_real')
])


sol, s_map = crn.simulate(t_max=300, steps=2000)
a1 = crn.get_value(sol, s_map, 'dx_p')
a2 = crn.get_value(sol, s_map, 'dx_n')

print(f"dx_p: {a1}, dx_n:{a2}")

plt.figure(figsize=(12, 12))

plt.subplot(3, 1, 1)
steps = [2, 4, 6, 8, 10]
for i, step in enumerate(steps):
    if f'o{step}' in s_map:
        plt.plot(sol.t, sol.y[s_map[f'o{step}']], label=f'Phase {i+1} (o{step})')
plt.title(f"Oscillator Clocks (step=2)")
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(sol.t, sol.y[s_map['dx_real']], label='dx_real (Exp: 2)')
plt.plot(sol.t, sol.y[s_map['span_real']], label='span_real (Exp: 5)')
plt.plot(sol.t, sol.y[s_map['frac_real']], label='frac_real (Exp: 0.4)')
plt.plot(sol.t, sol.y[s_map['offset_real']], label='offset_real (Exp: 4.0)')
plt.title("Intermediates (Decoded, Positive-Rail)")
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 3)
final_val = sol.y[s_map['result_real']]
plt.plot(sol.t, final_val, label='Final Result', color='black', lw=2)
plt.axhline(y=expected_result, color='red', linestyle=':', label='Expected (9.0)')
plt.title(f"Final Result (Value: {final_val[-1]:.4f})")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
print(f"Final Value: {final_val[-1]:.4f}")
