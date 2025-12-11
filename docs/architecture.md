# pycrn Overview

Like the Mathematica package developed by David Soloveichik, we convert high-level operations into the corresponding CRN reactions.

## Main Components

### Compiler
- Converts operations into parallel chemical reactions.
- It validates if inputs are correct, maps them to predefined reactions, and also manages extra resources required like trash helpers.

### Oscillator
- Provides the clock signals:  
  o1 → o2 → ... → oN
- Each phase is catalyzed by a specific oscillator species (e.g., o3).
- Reactions for that phase only turn on when the corresponding species is high.

### Simulator
- Converts chemical reactions into ODEs.
- Integrates them over time using scipy.integrate.solve_ivp (BDF method).

---
