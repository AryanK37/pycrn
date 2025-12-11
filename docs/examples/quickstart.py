from pycrn import CRNCompiler

compiler = CRNCompiler()
compiler.add_phase([("add", "A", "B", "C")], inputs={"A": 5, "B": 3})

sol, s_map = compiler.simulate()

print("C final:", compiler.get_value(sol, s_map, "C"))
