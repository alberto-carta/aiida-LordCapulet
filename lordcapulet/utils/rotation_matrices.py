#%%
import numpy as np
from scipy.linalg import expm


def spherical_to_cubic_rotation(dim=5, convention='qe'):
    """
    Returns the 5x5 transformation matrix T that rotates from the spherical
    harmonic basis (Y_2^-2, Y_2^-1, Y_2^0, Y_2^1, Y_2^2) to the Quantum ESPRESSO
    cubic harmonic basis (r^2-3z^2, xz, yz, xy, x^2-y^2) for d-orbitals.

    The transformation is applied as: Cubic_Vector = T @ Spherical_Vector
    For a matrix M in spherical basis, the rotated matrix in cubic basis is:
    M_cubic = T @ M @ T.T.conj()
    """
    sqrt2_inv = 1 / np.sqrt(2)
    if convention != 'qe':
        raise ValueError("Only 'qe' convention is supported for now.")
    if dim != 5:
        raise ValueError("Only dimension 5 is supported for d-orbitals.")
    

    if convention == 'qe':
        if dim == 5:
    
            # Rows: r^2-3z^2, xz, yz, xy, x^2-y^2 (QE order)
            # Cols: Y_2^-2, Y_2^-1, Y_2^0, Y_2^1, Y_2^2
            T = np.array([
                [0, 0, 1, 0, 0],                                     # r^2-3z^2 ~ Y_2^0
                [0, sqrt2_inv, 0, -sqrt2_inv, 0],                    # xz ~ 1/sqrt(2) * (Y_2^-1 - Y_2^1)
                [0, 1j * sqrt2_inv, 0, 1j * sqrt2_inv, 0],           # yz ~ i/sqrt(2) * (Y_2^-1 + Y_2^1)
                [1j * sqrt2_inv, 0, 0, 0, -1j * sqrt2_inv],          # xy ~ i/sqrt(2) * (Y_2^-2 - Y_2^2)
                [sqrt2_inv, 0, 0, 0, sqrt2_inv]                      # x^2-y^2 ~ 1/sqrt(2) * (Y_2^-2 + Y_2^2)
            ])
            # guoyuan points out that there might be some internal reordering in QE
            # Francesco also noticed something similar, we need to check  this
            # check and one might want to swap the last two rows
            return T  

def get_angular_momentum_operators(l):
    """
    Generates the angular momentum operators Lx, Ly, Lz, L+, L- for a given
    orbital angular momentum quantum number l.

    The matrices are represented in the |l, m> basis, where rows and columns
    are ordered by *increasing* m values (m = -l, -l+1, ..., l).
    hbar is set to 1 for simplicity.

    Args:
        l (int or float): The orbital angular momentum quantum number.
                          Must be a non-negative integer or half-integer.

    Returns:
        tuple: A tuple containing five NumPy arrays: (Lx, Ly, Lz, Lp, Lm)
               where:
               - Lx (ndarray): The x-component angular momentum operator.
               - Ly (ndarray): The y-component angular momentum operator.
               - Lz (ndarray): The z-component angular momentum operator.
               - Lp (ndarray): The raising operator (L+).
               - Lm (ndarray): The lowering operator (L-).

    Raises:
        ValueError: If l is negative or not a valid quantum number.
    """
    if not isinstance(l, (int, float)) or l < 0 or (l * 2) % 1 != 0:
        raise ValueError("l must be a non-negative integer or half-integer.")
    if l != int(l):
        print("Warning: For orbital angular momentum, l is typically an integer. "
              "Generating operators for half-integer l as per general angular momentum.")

    dim = int(2 * l + 1)
    
    # Initialize operators as zero matrices
    Lz = np.zeros((dim, dim), dtype=complex)
    Lp = np.zeros((dim, dim), dtype=complex) # L+
    Lm = np.zeros((dim, dim), dtype=complex) # L-

    # Populate Lz, L+, L- matrices
    for i in range(dim):
        m = -l + i  # Current m value for this row/column index (from -l to l)

        # Lz operator (diagonal)
        # The diagonal element at index i corresponds to the m value -l + i
        Lz[i, i] = m

        # L+ operator (raising operator)
        # L+ |l, m> = sqrt(l(l+1) - m(m+1)) |l, m+1>
        # This operator connects |m> to |m+1>.
        # If current state is |l, m> (at index i), the next state is |l, m+1> (at index i+1).
        if m < l: # L+ cannot act on the highest m state
            # The element is at row (index for m+1), column (index for m)
            Lp[i + 1, i] = np.sqrt(l * (l + 1) - m * (m + 1))

        # L- operator (lowering operator)
        # L- |l, m> = sqrt(l(l+1) - m(m-1)) |l, m-1>
        # This operator connects |m> to |m-1>.
        # If current state is |l, m> (at index i), the next state is |l, m-1> (at index i-1).
        if m > -l: # L- cannot act on the lowest m state
            # The element is at row (index for m-1), column (index for m)
            Lm[i - 1, i] = np.sqrt(l * (l + 1) - m * (m - 1))

    # Derive Lx and Ly from L+ and L-
    Lx = 0.5 * (Lp + Lm)
    Ly = (0.5 / 1j) * (Lp - Lm)

    return Lx, Ly, Lz, Lp, Lm


# give a normalized axis direction and angle, return the rotation matrix
def get_rotation_matrix(angle, axis, Lx, Ly, Lz):
    """
    Returns the rotation matrix for a given angle and axis direction.
    
    Args:
        angle (float): The rotation angle in radians.
        axis (array-like): The axis direction as a 3-element array.
        Lx, Ly, Lz (ndarray): Angular momentum operators for the system.

    Returns:
        ndarray: The rotation matrix.
    """
    # assert that axis is a 3-element vector
    if len(axis) != 3:
        raise ValueError("Axis must be a 3-element vector.")

    # Normalize the axis
    axis = np.array(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm == 0:
        raise ValueError("Axis cannot be the zero vector.")
    axis /= norm

    # exponentiate the Lx, Ly, Lz operators
    R = np.zeros_like(Lx, dtype=complex)

    R = expm(-1j * angle * (axis[0] * Lx + axis[1] * Ly + axis[2] * Lz))


    return R


def rotate_QE_matrix(rho_qe, angle, axis):
    """
    Rotate a Quantum ESPRESSO state matrix using a rotation matrix.

    Args:
        rho_qe (ndarray): The Quantum ESPRESSO state matrix to be rotated.
        angle (float): The rotation angle in radians.
        axis (array-like): The axis direction as a 3-element array.

    Returns:
        ndarray: The rotated state matrix.
    """
    # get angular momentum operator from the size of rho_qe
    dim = rho_qe.shape[0]
    l_angular_momentum = (dim - 1) / 2
    Lx, Ly, Lz, _, _ = get_angular_momentum_operators(l_angular_momentum)

    # get the rotation matrix
    R = get_rotation_matrix(angle, axis, Lx, Ly, Lz)
    # Convert R to cubic basis
    C = spherical_to_cubic_rotation(dim=dim, convention='qe')
    R_cubic = C @ R @ C.T.conj()

    # Rotate the density matrix
    rho_rotated = R_cubic @ rho_qe @ R_cubic.T.conj()

    return rho_rotated

#%% test 
# Lx, Ly, Lz, _,_ = get_angular_momentum_operators(2)

# R = get_rotation_matrix(np.pi, [0, 0, 1], Lx, Ly, Lz)

# with np.printoptions(precision=3, suppress=True):
#     pass
#     # print("Rotation Matrix R:")
#     # print(R)
#     # print("Rotate and rotate back should yield identity:")
#     # print(R.T.conj()@R)  # Should be close to identity matrix

# # let's rotate a quantum espresso state matrix
# rho_qe = np.array([
#     [1, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0],
#     [0, 0, 1, 0, 0],
#     [0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 1]
# ], dtype=complex)

# rho_rotated = rotate_QE_matrix(rho_qe, np.pi/4, [0, 0, 1])

# with np.printoptions(precision=3, suppress=True):
#     print("Rotated Quantum ESPRESSO state matrix:")
#     print(rho_rotated.real)

# %%
# # get a random rotation matrix for d orbitals
# from scipy.stats import uniform as uniform_direction
# random_direction = uniform_direction.rvs(size=3)
# random_angle = np.random.uniform(0, 2 * np.pi)

# Lx, Ly, Lz, _, _ = get_angular_momentum_operators(2)

# rot_mat = get_rotation_matrix(random_angle, random_direction, Lx, Ly, Lz)

# with np.printoptions(precision=3, suppress=True):
#     print("Random Rotation Matrix for d-orbitals:")
#     print(rot_mat)


# C = spherical_to_cubic_rotation(dim=5, convention='qe')
# # C = get_spherical_to_cubic_d_orbital_transformation_matrix()
# rot_mat_qe = C @ rot_mat @ C.T.conj()

# with np.printoptions(precision=3, suppress=True):
#     print("Quantum ESPRESSO Rotation Matrix:")
#     print(rot_mat_qe)

# # now change it to a Quantum ESPRESSO rotation matrix
# # consider a density matrix with just a 1

# rho_trial = np.array([
#     [1, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0], 
#     [0, 0, 0, 0, 0]], dtype=complex)
# rho_rotated = rotate_QE_matrix(rho_trial, random_angle, random_direction)

# with np.printoptions(precision=3, suppress=True):
#     print("Rotated Quantum ESPRESSO state matrix:")
#     print(rho_rotated)
#     print("Trace of rotated matrix:", np.trace(rho_rotated).real)
# %%
