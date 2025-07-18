
import numpy as np
import json
from aiida.orm import Dict, Code, KpointsData, load_node
from aiida.engine import WorkChain, run


from aiida.orm import Dict, List, Int, Float, Str
from aiida.engine import calcfunction

from .proposal_modes import propose_random_constraints

@calcfunction
def aiida_propose_occ_matrices_from_results(pk_list: List, N: int = 8, debug: bool = False,
                                             mode: str = 'random', **kwargs):

    """
    AiiDA calcfunction that takes a list of PKs
    and returns a list of PKs of Dict nodes that are themselves stored
    and contain the occupation matrices. 

    This function wraps `propose_new_constraints` to create the Dict nodes.
    
    :param pk_list: List of PKs to load the occupation matrices from a AFMScanWorkChain or ConstrainedScanWorkChain.
    :param N: Int, number of dictionaries to return.
    :param debug: Bool, whether to print debug information.
    :param mode: Mode for selecting the dictionaries, e.g., 'random' or 'read'.
    :param kwargs: Additional keyword arguments for `propose_new_constraints`.

    :return: List of Dict nodes containing the occupation matrices.

    !!! WARNING PRINT STATEMENTS !!!

    This function uses print statements to log debug information.
    This is because it is a calcfunction wrapping AiiDA agnostic code.
    The print statements will be captured in the AiiDA report log.
    """

    # load the nodes from the PKs
    occ_matrices = []
    for pk in pk_list.get_list():
        node = load_node(pk)
        if node.__class__.__name__ == "Dict":
            occupation_matrix = node.get_dict()
            occ_matrices.append(occupation_matrix)

    # now get the N dictionaries from the list


    # Convert AiiDA data types to native Python types for the internal function call
    # This is necessary because propose_new_constraints expects standard Python types,
    # but AiiDA calcfunctions receive AiiDA node types (Dict, List, Int, Float, Str)
    kwargs_internal = {}
    for key, value in kwargs.items():
        # Handle AiiDA Dict and List nodes by extracting their content
        if isinstance(value, (Dict, List)):
            # For List nodes: get_list() returns the Python list
            # For Dict nodes: get_dict() returns the Python dictionary
            kwargs_internal[key] = value.get_list() if isinstance(value, List) else value.get_dict()
        # Handle AiiDA numeric and string nodes by extracting their .value attribute
        elif isinstance(value, (Int, Float, Str)):
            kwargs_internal[key] = value.value
        # Raise error for any unsupported AiiDA node types
        else:
            raise ValueError(f"Unsupported AiiDA node type for key '{key}': {type(value)}. "
                           f"Only Dict, List, Int, Float, and Str nodes are supported.")
    # check if this ran in the debug mode
    if debug:
        print(f"Loaded {len(occ_matrices)} occupation matrices from nodes with PKs: {pk_list.get_list()}")
        print(f"Using proposal mode: {mode.value} with N = {N.value} samples per generation")

    # magic happens here
    proposals = propose_new_constraints(occ_matrices,
                                    N= N.value,
                                    debug=debug.value,
                                    mode=mode.value,
                                    **kwargs_internal
                                    )

    # create Dict nodes for each proposal
    dict_nodes = []
    for proposal in proposals:
        # if debug:
            # print(f"Creating Dict node for proposal: {proposal}")
        dict_node = Dict(dict=proposal)
        dict_node.store()
        dict_nodes.append(dict_node)

    # return a list of the PKs of the Dict nodes
    return List(list=[node.pk for node in dict_nodes])


# define a function that takes a list of dictionaries and N and returns a list of N dictionaries
def propose_new_constraints(occ_matr_list, N, mode='random', debug=True, **kwargs):
    """
    !!IMPORTANT!! THIS FUNCTION SHOULD NOT GET ANY AIIDA TYPES AS INPUT
    
    Returns a list of N dictionaries from a list of dictionaries.
    
    This will be a giant function with a lot of logic
    and it is better that everything non trivial gets its own function and wrapped here

    
    :param occ_matr_list: List of dictionaries to choose from.
    :param N: Number of dictionaries to return.
    :return: List of N dictionaries.
    """
    # make sure that N is > 1

    atom_names = list(occ_matr_list[1].keys())
    natoms = len(atom_names)
    spin_names = ['up', 'down']
    nspin = 2 # up and down spin

    norbitals = np.array(occ_matr_list[1][atom_names[0]]['spin_data']['up']['occupation_matrix']).shape[0] 

    # structure of an output dictionary
    # { 'matrix': array of shape (natoms, nspin, norbitals, norbitals) }'
    if debug:
        print(f"Number of atoms: {natoms}")
        print(f"Number of spins: {nspin}")
        print(f"Number of orbitals for atom 1: {norbitals}")


    if N < 1:
        raise ValueError("N must be greater than or equal to 1")

    # implement case switch for mode
    match mode:

        case 'random':
            proposals = propose_random_constraints(occ_matr_list, natoms,  N, debug=debug, **kwargs)

        case 'read':
            # check if there is readfile in kwargs
            if 'readfile' not in kwargs:
                raise ValueError("readfile must be provided in kwargs for read mode")
            readfile = kwargs['readfile']
            # read the json file
            with open(readfile, 'r') as f:
                loaded_data = json.load(f)
            # assert that the loaded list is longer than N
            if len(loaded_data) < N:
                raise ValueError(f"Loaded data has only {len(loaded_data)} dictionaries, but N is {N}")

            # now loaded_data[iteration]['occupation_numbers'][iatom][ispin] is a norbital x norbital matrix


            #create now a list of dictionaries
            target_matrix_np = np.zeros((natoms, nspin, norbitals, norbitals))

            proposals = []
            for iteration in range(0, N):
                proposal = {}
                for iatom in range(natoms):
                    for ispin in range(nspin):
                        target_matrix_np[iatom, ispin] = \
                            np.array(loaded_data[iteration]['occupation_numbers'][iatom][ispin])
                proposal['matrix'] = target_matrix_np.tolist()
                proposals.append(proposal)
            if debug:
                print(f"Reading {N} dictionaries from file")

        
        
    
    return proposals