"""
__init__.py for pde package.

Imports relevant PDE classes and solver functions for convenience.
"""

from .cable_equation import CableConfig, init_domain, initial_condition
from .solver_crank_nicolson import crank_nicolson_step, solve_cable_crank_nicolson
from .solver_utils import build_tridiag, apply_neumann_bc, apply_dirichlet_bc
