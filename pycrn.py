import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import time

from tqdm import tqdm

class CRNCompiler:
    def __init__(self, oscillator_step=3):
        self.phases = []                
        self.reactions = []             
        self.species = set()            
        self.oscillator_step = oscillator_step
        self.trash_counter = 0
        self.helper_counter = 0
        self.initials = {}              # { species_name: concentration }

    def _get_trash(self):
        self.trash_counter += 1
        name = f"trash{self.trash_counter}"
        self.species.add(name)
        return name
        
    def _get_helper(self):
        self.helper_counter += 1
        name = f"helper{self.helper_counter}"
        self.species.add(name)
        return name

    def add_phase(self, operations, inputs=None):
        phase1_ops = []
        phase2_ops = []
        requires_two_cycles = False

        for op in operations:
            if not isinstance(op, (list, tuple)) or len(op) < 2:
                raise ValueError("Each operation must be a tuple/list like ('add', a, b, c)")

            op_name = op[0].lower()
            new_args = []

            # process each operand after op_name

            for operand in op[1:]:

                # inline input tuple: ('species_name', value)
                if isinstance(operand, tuple) and len(operand) == 2 and isinstance(operand[0], str):
                    name, val = operand
                    if val < 0: raise ValueError(f"Concentration for '{name}' must be positive.")
                    if name not in self.initials: self.initials[name] = val
                    self.species.add(name)
                    new_args.append(name)
                else:
                    new_args.append(operand)
                    if isinstance(operand, str): self.species.add(operand)

            if op_name == 'cmp':
                requires_two_cycles = True
                phase1_ops.append(tuple(['cmp_setup'] + new_args))
                phase2_ops.append(tuple(['cmp_majority'] + new_args))
            else:
                phase1_ops.append(tuple([op_name] + new_args))

        if inputs:
            for name, val in inputs.items():
                if val < 0: raise ValueError(f"Input '{name}' must be positive.")
                if name not in self.initials: self.initials[name] = val
                self.species.add(name)

        self.phases.append(phase1_ops)
        if requires_two_cycles:
            self.phases.append(phase2_ops)

        return self

    def _compile_add(self, clock, a, b, c):
        t = self._get_trash()
        self._add_reaction([clock, a], [clock, a, c], k=1.0)
        self._add_reaction([clock, b], [clock, b, c], k=1.0)
        self._add_reaction([clock, c], [clock, t], k=1.0)

    def _compile_mul(self, clock, a, b, c):
        t = self._get_trash()
        self._add_reaction([clock, a, b], [clock, a, b, c], k=1.0)
        self._add_reaction([clock, c], [clock, t], k=1.0)

    def _compile_sub(self, clock, a, b, c):
        """ c = max(0, a - b) """
        h = self._get_helper()
        t1, t2 = self._get_trash(), self._get_trash()
        self._add_reaction([clock, a], [clock, a, c], k=1.0)
        self._add_reaction([clock, b], [clock, b, h], k=1.0)
        self._add_reaction([clock, c], [clock, t1], k=1.0)
        self._add_reaction([clock, c, h], [clock, t2], k=1.0)
        
    def _compile_div(self, clock, a, b, c):
        self._add_reaction([clock, a], [clock, a, c], k=1.0)
        self._add_reaction([clock, b, c], [clock, b], k=1.0)
    def _compile_load(self, clock, a, b):
        """
        LOAD module:
            A --→ A + B
            B --→ ∅
        """
        t = self._get_trash()

        self._add_reaction([clock, a], [clock, a, b], k=1.0)
        self._add_reaction([clock, b], [clock, t], k=1.0)

    def _compile_cmp_setup(self, clock, a, b, gt, lt):
        # Gt + B -> Lt + B  (If B is high, push Gt -> Lt)
        self._add_reaction([clock, gt, b], [clock, lt, b], k=1.0)
        # Lt + A -> Gt + A  (If A is high, push Lt -> Gt)
        self._add_reaction([clock, lt, a], [clock, gt, a], k=1.0)

    def _compile_cmp_majority(self, clock, a, b, gt, lt):
        B_int = self._get_helper()

        # Gt + Lt -> Lt + B
        self._add_reaction([clock, gt, lt], [clock, lt, B_int], k=1.0)
        # B + Lt -> 2 Lt
        self._add_reaction([clock, B_int, lt], [clock, lt, lt], k=1.0)

        # Bias for Gt 
        # Lt + Gt -> Gt + B
        self._add_reaction([clock, lt, gt], [clock, gt, B_int], k=1.0)
        # B + Gt -> 2 Gt
        self._add_reaction([clock, B_int, gt], [clock, gt, gt], k=1.0)

    def _add_reaction(self, reactants, products, k=1.0):
        # ensure reactants/products are strings (if user accidentally passed tuples, this normalizes)
        reactants_clean = []
        products_clean = []
        for r in reactants:
            if isinstance(r, tuple):  # defensive: take first element if tuple-like
                reactants_clean.append(r[0])
            else:
                reactants_clean.append(r)
        for p in products:
            if isinstance(p, tuple):
                products_clean.append(p[0])
            else:
                products_clean.append(p)

        self.reactions.append({'reactants': reactants_clean, 'products': products_clean, 'k': k})
        # update species with string names only
        for s in reactants_clean + products_clean:
            if isinstance(s, str):
                self.species.add(s)

    def compile(self):
        # print("Compiling CRN...")
        num_phases = len(self.phases)
        
        oscillator_len = self.oscillator_step * (num_phases + 1)
        osc_species = [f"o{i}" for i in range(1, oscillator_len + 1)]
        for o in osc_species:
            self.species.add(o)
        
        for i in range(oscillator_len):
            r = osc_species[i]
            p = osc_species[(i + 1) % oscillator_len]
            self._add_reaction([r, p], [p, p], k=1.0)

        for i, phase_ops in enumerate(self.phases):
            clock_index = (i + 1) * self.oscillator_step
            clock = f"o{clock_index}"
            # print(f"  Phase {i+1} (gated by {clock}):")
            for op in phase_ops:
                op_type = op[0].lower()
                args = op[1:]
                if op_type == 'add': self._compile_add(clock, *args)
                elif op_type == 'mul': self._compile_mul(clock, *args)
                elif op_type == 'sub': self._compile_sub(clock, *args)
                elif op_type == 'div': self._compile_div(clock, *args)
                elif op_type == 'load': self._compile_load(clock, *args)
                # Compare Ops (Generated by add_phase splitting)
                elif op_type == 'cmp_setup': self._compile_cmp_setup(clock, *args)
                elif op_type == 'cmp_majority': self._compile_cmp_majority(clock, *args)

    def simulate(self, t_max=100, inputs=None, steps=1000):
        """
        inputs: optional dict of initial concentrations that override recorded initials.
        """
        if not self.reactions:
            self.compile()
        self.print_ode_system()
            
        sorted_species = sorted([s for s in self.species if isinstance(s, str)])
        s_map = {name: i for i, name in enumerate(sorted_species)}
        num_species = len(sorted_species)

        y0 = np.full(num_species, 1e-12) # ZERO

        # 1) seed from stored initials
        for name, val in self.initials.items():
            if name in s_map:
                y0[s_map[name]] = val

        # 2) apply any inputs passed directly to simulate()
        if inputs:
            for name, val in inputs.items():
                if val < 0:
                    raise ValueError(f"Simulation input concentration for '{name}' must be non-negative, but got {val}")
                if name in s_map:
                    y0[s_map[name]] = val

        # 3) ensure oscillator starts
        for o in ('o1', 'o2'):
            if o in s_map and np.isclose(y0[s_map[o]], 1e-12):
                y0[s_map[o]] = 1.0
        
        reaction_list = self.reactions
        
        def dydt_func(t, y):
            # no negative concentrations allowed
            y[y < 0] = 0 
            
            derivs = np.zeros(num_species)
            for rxn in reaction_list:
                # 1. Calculate Rate
                rate = rxn['k']
                skip = False
                for r in rxn['reactants']:
                    if r not in s_map:
                        rate = 0.0
                        skip = True
                        break
                    rate *= y[s_map[r]]
                
                if skip or rate == 0.0:
                    continue

                # 2. Apply Net Changes 
                # Here we identify all unique species involved in this reaction and apply the NET change (Products - Reactants)
                unique_species = set(rxn['reactants'] + rxn['products'])
                
                for s in unique_species:
                    if s in s_map:
                        # Count occurrences in products vs reactants
                        net_change = rxn['products'].count(s) - rxn['reactants'].count(s)
                        
                        # If s is a catalyst (like o2), net_change is 0, so we don't modify derivatives
                        if net_change != 0:
                            derivs[s_map[s]] += rate * net_change

            return derivs

        # print(f"\nSimulating {len(self.phases)} phases with {len(sorted_species)} species...")
        start_time = time.time()
        sol = solve_ivp(dydt_func, [0, t_max], y0, t_eval=np.linspace(0, t_max, steps), method='BDF')
        
        return sol, s_map

    def print_ode_system(self):
        if not self.reactions:
            print("No reactions compiled yet. Run compile() first.")
            return
        
        print("\nFINAL ODE SYSTEM")
        # Sort species for consistency (strings only)
        sorted_species = sorted([s for s in self.species if isinstance(s, str)])

        def rate_expr(rxn):
            if len(rxn['reactants']) == 0:
                return f"{rxn['k']}"
            term = " * ".join(rxn['reactants'])
            return f"{rxn['k']} * {term}"

        for X in sorted_species:
            terms = []
            for rxn in self.reactions:
                # Calculate Net Stoichiometry
                r_count = rxn['reactants'].count(X)
                p_count = rxn['products'].count(X)
                net_change = p_count - r_count
                
                # Only print if there is an actual NET change.
                # Catalysts (Reactant 1, Product 1) result in 0 and are skipped.
                if net_change != 0:
                    rate = rate_expr(rxn)
                    if net_change > 0:
                        terms.append(f"+ {net_change} * ({rate})")
                    else:
                        # Print negative change
                        terms.append(f"- {abs(net_change)} * ({rate})")
            
            rhs = " ".join(terms)
            if rhs.strip() == "":
                rhs = "0"
            print(f"d{X}/dt = {rhs}")

    def get_value(self, sol, s_map, species_id, t_index=-1):
        """
        Return the concentration of species_id from the simulation result.

        sol: it is the object
        s_map: species : index map
        Above two are returned by simulate()
        species_id: e.g. 'A1'
        t_index: which timestep to read (default = final (last))
        """
        if species_id not in s_map:
            raise ValueError(f"Species '{species_id}' not found in s_map.")

        idx = s_map[species_id]
        return sol.y[idx][t_index]
    
    
