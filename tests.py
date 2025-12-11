import unittest
import numpy as np
from pycrn import CRNCompiler

TOLERANCE = 0.05

class TestCRNArithmetic(unittest.TestCase):

    def test_compare(self):
        print("\n Testing COMPARE (A=0.6, B=0.4; Expect Gt wins) ")
        compiler = CRNCompiler(oscillator_step=3)
        compiler.add_phase([
            ('cmp', 'A', 'B', 'Gt', 'Lt')
        ], inputs={'A': 0.6, 'B': 0.4, 'Gt': 0.5, 'Lt': 0.5})

        sol, s_map = compiler.simulate(t_max=100)

        gt_final = compiler.get_value(sol, s_map, 'Gt')
        lt_final = compiler.get_value(sol, s_map, 'Lt')
        print("Final Gt:", gt_final) 
        print("Final Lt:", lt_final)
        
        assert gt_final > 0.9, "Gt did not win!"
        assert lt_final < 0.1, "Lt did not lose!"
    
    def test_addition(self):
        print("\n Testing ADD (5 + 3 = 8) ")
        compiler = CRNCompiler(oscillator_step=1) 
        compiler.add_phase([
            ('add', 'A', 'B', 'C')
        ], inputs={'A': 5.0, 'B': 3.0})
        sol, s_map = compiler.simulate(t_max=50, steps=1000)
        result = compiler.get_value(sol, s_map, 'C')
        print(f"  Result C: {result:.4f}")
        
        self.assertAlmostEqual(result, 8.0, delta=TOLERANCE)

    def test_multiplication(self):
        print("\n Testing MUL (4 * 3 = 12) ")
        compiler = CRNCompiler(oscillator_step=1)
        compiler.add_phase([
            ('mul', 'A', 'B', 'C')
        ], inputs={'A': 4.0, 'B': 3.0})
        sol, s_map = compiler.simulate(t_max=50, steps=1000)
        result = compiler.get_value(sol, s_map, 'C')
        print(f"  Result C: {result:.4f}")
        
        self.assertAlmostEqual(result, 12.0, delta=TOLERANCE)

    def test_subtraction(self):
        print("\n Testing SUB (10 - 4 = 6) ")
        compiler = CRNCompiler(oscillator_step=1)
        compiler.add_phase([
            ('sub', 'A', 'B', 'C')
        ], inputs={'A': 10.0, 'B': 4.0})
        sol, s_map = compiler.simulate(t_max=60, steps=1000)
        result = compiler.get_value(sol, s_map, 'C')
        print(f"  Result C: {result:.4f}")
        
        self.assertAlmostEqual(result, 6.0, delta=TOLERANCE)

    def test_subtraction_zero(self):
        print("\n Testing SUB ReLU (4 - 10 = 0) ")
        compiler = CRNCompiler(oscillator_step=1)
        compiler.add_phase([
            ('sub', 'A', 'B', 'C')
        ], inputs={'A': 4.0, 'B': 10.0})
        sol, s_map = compiler.simulate(t_max=60, steps=1000)
        result = compiler.get_value(sol, s_map, 'C')
        print(f"  Result C: {result:.4f}")
        self.assertTrue(result < 0.02, f"Expected Approx. 0, got {result}")

    def test_division(self):
        print("\n Testing DIV (10 / 2 = 5) ")
        compiler = CRNCompiler(oscillator_step=1)
        compiler.add_phase([
            ('div', 'A', 'B', 'C')
        ], inputs={'A': 10.0, 'B': 2.0})

        sol, s_map = compiler.simulate(t_max=100, steps=2000)
        
        result = compiler.get_value(sol, s_map, 'C')
        print(f"  Result C: {result:.4f}")
        
        self.assertAlmostEqual(result, 5.0, delta=TOLERANCE)

    def test_load(self):
        print("\n Testing LOAD (Copy 7 -> Target) ")
        compiler = CRNCompiler(oscillator_step=1)
        compiler.add_phase([
            ('load', 'Source', 'Target')
        ], inputs={'Source': 7.0})
        sol, s_map = compiler.simulate(t_max=50, steps=1000)
        result = compiler.get_value(sol, s_map, 'Target')
        print(f"  Result Target: {result:.4f}")
        
        self.assertAlmostEqual(result, 7.0, delta=TOLERANCE)

if __name__ == '__main__':
    unittest.main()