import numpy as np
from typing import List, Tuple

def generate_villain(Lx: int, Ly: int, j_magnitude: float = 1.0) -> np.ndarray:
    """
    Generate coupling values for a fully frustrated lattice (Villain model).
    
    The strategy sets all horizontal bonds to +J. The vertical bonds alternate
    sign by column: even columns have +J, odd columns have -J. This ensures
    that the product of J signs around every plaquette is -1, making the
    lattice fully frustrated.
    
    Args:
        Lx: Number of sites in the x-direction.
        Ly: Number of sites in the y-direction.
        j_magnitude: Magnitude of the coupling J.
        
    Returns:
        np.ndarray: Array of coupling values corresponding to the bonds.
    """
    num_h_bonds = (Lx - 1) * Ly
    num_v_bonds = Lx * (Ly - 1)
    J = np.zeros(num_h_bonds + num_v_bonds)
    
    idx = 0
    # Horizontal bonds: (x,y) to (x+1,y) for y in 0..Ly-1, x in 0..Lx-2
    for y in range(Ly):
        for x in range(Lx - 1):
            J[idx] = j_magnitude
            idx += 1
            
    # Vertical bonds: (x,y) to (x,y+1) for x in 0..Lx-1, y in 0..Ly-2
    for x in range(Lx):
        for y in range(Ly - 1):
            if x % 2 == 0:
                J[idx] = j_magnitude
            else:
                J[idx] = -j_magnitude
            idx += 1
            
    return J

def generate_ea_bimodal(Lx: int, Ly: int, j_magnitude: float = 1.0, seed: int = 42) -> np.ndarray:
    """
    Generate Edwards-Anderson bimodal couplings (+J or -J with equal probability).
    
    Args:
        Lx: Number of sites in the x-direction.
        Ly: Number of sites in the y-direction.
        j_magnitude: Magnitude of the coupling J.
        seed: Random seed.
        
    Returns:
        np.ndarray: Array of coupling values.
    """
    rng = np.random.default_rng(seed)
    num_bonds = (Lx - 1) * Ly + Lx * (Ly - 1)
    return rng.choice([-j_magnitude, j_magnitude], size=num_bonds)

def generate_gaussian(Lx: int, Ly: int, j_magnitude: float = 1.0, seed: int = 42) -> np.ndarray:
    """
    Generate Gaussian distributed couplings N(0, J^2).
    
    Args:
        Lx: Number of sites in the x-direction.
        Ly: Number of sites in the y-direction.
        j_magnitude: Standard deviation of the Gaussian distribution.
        seed: Random seed.
        
    Returns:
        np.ndarray: Array of coupling values.
    """
    rng = np.random.default_rng(seed)
    num_bonds = (Lx - 1) * Ly + Lx * (Ly - 1)
    return rng.normal(0.0, j_magnitude, size=num_bonds)

def frustration_ratio(J: np.ndarray, bonds: List[Tuple[int, int]], Lx: int, Ly: int) -> float:
    """
    Calculate the fraction of frustrated plaquettes in the lattice.
    
    A plaquette is frustrated if the product of the coupling signs around
    it is negative.
    
    Args:
        J: Array of coupling values.
        bonds: List of bonds as tuples of qubit indices (u, v).
        Lx: Number of sites in the x-direction.
        Ly: Number of sites in the y-direction.
        
    Returns:
        float: Fraction of frustrated plaquettes.
    """
    bond_to_idx = {tuple(sorted((u, v))): idx for idx, (u, v) in enumerate(bonds)}
    
    def q_idx(x: int, y: int) -> int:
        return y * Lx + x
        
    num_plaquettes = (Lx - 1) * (Ly - 1)
    if num_plaquettes == 0:
        return 0.0
        
    frustrated = 0
    for y in range(Ly - 1):
        for x in range(Lx - 1):
            # The 4 bonds forming the plaquette
            b1 = tuple(sorted((q_idx(x, y), q_idx(x+1, y))))
            b2 = tuple(sorted((q_idx(x+1, y), q_idx(x+1, y+1))))
            b3 = tuple(sorted((q_idx(x, y+1), q_idx(x+1, y+1))))
            b4 = tuple(sorted((q_idx(x, y), q_idx(x, y+1))))
            
            j1 = J[bond_to_idx[b1]]
            j2 = J[bond_to_idx[b2]]
            j3 = J[bond_to_idx[b3]]
            j4 = J[bond_to_idx[b4]]
            
            prod = np.sign(j1) * np.sign(j2) * np.sign(j3) * np.sign(j4)
            if prod < 0:
                frustrated += 1
                
    return frustrated / num_plaquettes
