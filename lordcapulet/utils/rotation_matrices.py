#%%
import numpy as np

def spherical_to_cubic_rotation(dim=5, convention='qe'):
    """
    Returns the 5x5 transformation matrix T that rotates from the spherical
    harmonic basis (Y_2^-2, Y_2^-1, Y_2^0, Y_2^1, Y_2^2) to the Quantum ESPRESSO
    cubic harmonic basis (r^2-3z^2, xz, yz, xy, x^2-y^2) for d-orbitals.

    The transformation is applied as: Cubic_Vector = T @ Spherical_Vector
    """
    if convention != 'qe':
        raise ValueError("Only 'qe' convention is supported for now.")
    if dim != 5:
        raise ValueError("Only dimension 5 is supported for d-orbitals.")
    

    if convention == 'qe':
        if dim == 5:
            # Quantum ESPRESSO cubic harmonics for d-orbitals
            T = np.array([
                [0, 0, 1, 0, 0],                               # r^2-3z^2 ~ Y_2^0
                [0, -1j/np.sqrt(2), 0, 1j/np.sqrt(2), 0],      # xz ~ (Y_2^1 - Y_2^-1)
                [0, 1/np.sqrt(2), 0, 1/np.sqrt(2), 0],         # yz ~ (Y_2^1 + Y_2^-1)
                [-1j/np.sqrt(2), 0, 0, 0, 1j/np.sqrt(2)],      # xy ~ (Y_2^2 - Y_2^-2)
                [1/np.sqrt(2), 0, 0, 0, 1/np.sqrt(2)]          # x^2-y^2 ~ (Y_2^2 + Y_2^-2)
            ])
            return T  

