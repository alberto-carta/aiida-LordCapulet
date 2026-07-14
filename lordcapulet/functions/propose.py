
import numpy as np
import json
import contextlib
import io
from aiida.orm import Dict, Code, KpointsData, load_node, JsonableData
from aiida.engine import WorkChain, run
from aiida.orm import Dict, List, Int, Float, Str
from aiida.engine import calcfunction, Process

from .proposal_modes import propose_random_constraints, propose_random_so_n_constraints
from .proposal_modes import propose_gaussian_process_constraints
from .proposal_modes import propose_linear_bandit_constraints, propose_forest_bandit_constraints
from lordcapulet.data_structures import OccupationMatrixData, extract_occupations_from_calc, filter_atoms_by_species


# This calcfunction must be reworked to accept also a list of calculations pks
# in addition to the occupation_matrix_pk list


@calcfunction
def aiida_propose_occ_matrices_from_results(
    occ_matr_pks,
    calc_pks,
    N=8,
    debug=False,
    mode='random',*,
    reporter_type='aiida_process',
    hubbard_corr_atoms=None, **kwargs):

    """
    AiiDA calcfunction that takes a list of PKs
    and returns a list of PKs of Dict nodes that are themselves stored
    and contain the occupation matrices. 

    This function wraps `propose_new_constraints` to create the Dict nodes.

    Importantly: ALL THE AIIDA STUFF IS HANDLED HERE.
    The internal function `propose_new_constraints` should not receive any
    AiiDA specific logic and/or types.
    
    :param occ_matr_pks: List of PKs to load the occupation matrices from a StandardMagneticScanWorkChain or ConstrainedScanWorkChain.
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

    if reporter_type == None:
        def reporter(msg):
            print(msg)
    elif reporter_type == 'aiida_process':
        process  = Process.current()
        def reporter(msg):
            process.logger.report(msg)
    elif reporter_type == 'aiida+print':
        process  = Process.current()
        def reporter(msg):
            process.logger.report(msg)
            print(msg)

    # load the nodes from the PKs and convert to unified format
    occ_matrices = []
    for pk in occ_matr_pks.get_list():
        node = load_node(pk)
        
        # Handle JsonableData nodes containing OccupationMatrixData (preferred)
        if hasattr(node, 'obj') and hasattr(node.obj, 'as_dict'):
            # This is a JsonableData node containing our OccupationMatrixData
            occupation_matrix_data = node.obj
            occ_matrices.append(occupation_matrix_data)

            if debug and reporter is not None:
                reporter(f"Loaded occupation matrix from JsonableData node with PK {pk}")
        
        # Legacy support for saved matrix as Dict
        elif node.__class__.__name__ == "Dict":
            legacy_dict = node.get_dict()
            occupation_matrix_data = OccupationMatrixData.from_legacy_dict(legacy_dict)
            occ_matrices.append(occupation_matrix_data)
            # print a deprecated warning
            if debug and reporter is not None:
                reporter(f"Warning: Loaded occupation matrix from Dict node with PK {pk}. This is deprecated, please use JsonableData wrapping OccupationMatrixData.")
        
        # Handle calculation nodes directly (for backward compatibility)
        elif hasattr(node, 'process_type') and ('aiida.calculations:quantumespresso.pw' in node.process_type or 'aiida.calculations:lordcapulet.constrained_pw' in node.process_type):
            # try to get the output occupation matrix from the CalcJobNode using unified extractor
            try:
                occupation_matrix_data = extract_occupations_from_calc(node)
                occ_matrices.append(occupation_matrix_data)
                if debug and reporter is not None:
                    reporter(f"Warning: Extracted occupation matrix directly from CalcJobNode with PK {pk}. Consider using workchain outputs instead.")
            except Exception as e:
                raise ValueError(f"CalcJobNode with PK {pk}, error in parsing occupation_matrix: {e}")
        
        else:
            raise ValueError(f"Unsupported node type for PK {pk}: {type(node)}. Expected JsonableData(OccupationMatrixData), Dict, or CalcJobNode.")
        

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
    if debug and reporter is not None:
        reporter(f"Loaded {len(occ_matrices)} occupation matrices from nodes with PKs: {occ_matr_pks.get_list()}")
        reporter(f"Using proposal mode: {mode.value} with N = {N.value} samples per generation")


    # if the mode is 'gp' or 'gaussian_process' or a bandit mode,
    # we need to also pass the total energies from the calculation pks
    _energy_modes = {'gp', 'gaussian_process', 'linear_bandit', 'forest_bandit', 'forest', 'rf'}
    if mode.value in _energy_modes:
        energies = [ load_node(pk).outputs.output_parameters.get_dict().get('energy') for pk in calc_pks.get_list() ]
        kwargs_internal['energies'] = energies



    # Filter atoms by species if hubbard_corr_atoms is provided
    if hubbard_corr_atoms is not None:
        hubbard_corr_atoms_list = hubbard_corr_atoms.get_list() if hasattr(hubbard_corr_atoms, 'get_list') else hubbard_corr_atoms
        filtered_matrices = []
        for occ_matrix_data in occ_matrices:
            filtered_data = filter_atoms_by_species(occ_matrix_data, hubbard_corr_atoms_list)
            filtered_matrices.append(filtered_data)
        occ_matrices = filtered_matrices

    if debug:
        reporter(f"Loaded {len(occ_matrices)} OccupationMatrixData objects")
        for i, occ_data in enumerate(occ_matrices):
            reporter(f"  Matrix {i+1}: {len(occ_data)} atoms, species: {occ_data.get_atom_species()}")

    # Generate proposals using OccupationMatrixData directly (no conversion to legacy format)
    proposals = propose_new_constraints(
                                    occ_matr_list=occ_matrices,
                                    N=N.value,
                                    debug=debug.value,
                                    mode=mode.value,
                                    reporter=reporter,
                                    **kwargs_internal
                                    )

    # Store proposals as JsonableData nodes
    # Proposals are already OccupationMatrixData objects with metadata preserved
    dict_nodes = []
    for proposal in proposals:
        # Store as JsonableData
        json_node = JsonableData(proposal)
        json_node.store()
        dict_nodes.append(json_node)

    # return a list of the PKs of the Dict nodes
    return List(list=[node.pk for node in dict_nodes])


def propose_new_constraints(occ_matr_list, N, mode='random', debug=True, reporter=None, **kwargs):
    """
    Generate N new occupation matrix proposals from existing data.
    
    !!IMPORTANT!! THIS FUNCTION SHOULD NOT GET ANY AIIDA TYPES AS INPUT
    
    :param occ_matr_list: List of OccupationMatrixData objects to use as reference
    :param N: Number of proposals to generate
    :param mode: Proposal generation mode ('random', 'random_so_n', 'gaussian_process', or 'read')
    :param debug: Whether to print debug information
    :param reporter: Optional callable for logging (if None, uses print)
    :param kwargs: Additional mode-specific parameters
    :return: List of N OccupationMatrixData objects (proposals)
    """
    # Setup reporter
    if reporter is None:
        reporter = print
    
    if N < 1:
        raise ValueError("N must be greater than or equal to 1")
    
    # Get dimensions from first occupation matrix
    first_occ_data = occ_matr_list[0]
    natoms = len(first_occ_data)
    first_atom_label = first_occ_data.get_atom_labels()[0]
    norbitals = len(first_occ_data.get_occupation_matrix(first_atom_label, 'up'))
    nspin = 2  # up and down spin

    if debug:
        reporter(f"Number of atoms: {natoms}")
        reporter(f"Number of spins: {nspin}")
        reporter(f"Number of orbitals: {norbitals}")
        reporter(f"Atom labels: {first_occ_data.get_atom_labels()}")
        reporter(f"Atom species: {first_occ_data.get_atom_species()}")

    # implement case switch for mode
    match mode:

        case 'random':
            proposals = propose_random_constraints(occ_matr_list, natoms,  N, debug=debug, **kwargs)

        case 'random_so_n':
            proposals = propose_random_so_n_constraints(occ_matr_list, natoms, N, debug=debug, **kwargs)

        case 'gaussian_process' | 'gp':
            # TODO: merge with linear_bandit/forest_bandit — identical structure
            energies = kwargs.pop('energies', None)
            if energies is None:
                raise ValueError("Energies must be provided for Gaussian Process proposal mode")
            
            gp_config = kwargs.pop('gp_config', None)

            if debug:
                reporter(f"Energies provided: {energies}")
                reporter(f"Remaining kwargs keys: {list(kwargs.keys())}")

            current_generation = kwargs.pop('current_generation', None)
            assert current_generation is not None, "current_generation must be provided in kwargs for GP mode"

            if current_generation == 0:
                N_initial_random = kwargs.get('N_initial_random', N)
                reporter(f"Current generation is {current_generation}, proposing {N_initial_random} random constraints for initial GP training")
                proposals = propose_random_constraints(occ_matr_list, natoms,  N_initial_random, debug=debug, **kwargs)
            else:
                reporter(f"Current generation is {current_generation}, proposing {N} constraints using Gaussian Process")
                try:
                    proposals = propose_gaussian_process_constraints(
                        occ_matr_list, energies, natoms, N, gp_config=gp_config,
                        debug=debug, reporter=reporter, **kwargs
                    )
                except Exception as e:
                    reporter(f"Error in Gaussian Process proposal generation: {e}")
                    import traceback
                    reporter(traceback.format_exc())
                    reporter("Falling back to random constraint proposals")
                    proposals = propose_random_constraints(occ_matr_list, natoms,  N, debug=debug, **kwargs)

        case 'linear_bandit':
            # TODO: merge with gp/forest_bandit — identical structure
            energies = kwargs.pop('energies', None)
            if energies is None:
                raise ValueError("Energies must be provided for linear bandit proposal mode")
            
            linear_bandit_config = kwargs.pop('linear_bandit_config', None)

            if debug:
                reporter(f"Energies provided: {energies}")

            current_generation = kwargs.pop('current_generation', None)
            assert current_generation is not None, "current_generation must be provided in kwargs for linear bandit mode"

            if current_generation == 0:
                N_initial_random = kwargs.get('N_initial_random', N)
                reporter(f"Current generation is {current_generation}, proposing {N_initial_random} random constraints for initial training")
                proposals = propose_random_constraints(occ_matr_list, natoms, N_initial_random, debug=debug, **kwargs)
            else:
                reporter(f"Current generation is {current_generation}, proposing {N} constraints using linear bandit (ridge/ARD)")
                try:
                    proposals = propose_linear_bandit_constraints(
                        occ_matr_list, energies, natoms, N, gp_config=linear_bandit_config,
                        debug=debug, reporter=reporter, **kwargs
                    )
                except Exception as e:
                    reporter(f"Error in linear bandit proposal generation: {e}")
                    import traceback
                    reporter(traceback.format_exc())
                    reporter("Falling back to random constraint proposals")
                    proposals = propose_random_constraints(occ_matr_list, natoms, N, debug=debug, **kwargs)

        case 'forest_bandit' | 'forest' | 'rf':
            # TODO: merge with gp/linear_bandit — identical structure
            energies = kwargs.pop('energies', None)
            if energies is None:
                raise ValueError("Energies must be provided for forest bandit proposal mode")
            
            rf_config = kwargs.pop('rf_config', None)

            if debug:
                reporter(f"Energies provided: {energies}")

            current_generation = kwargs.pop('current_generation', None)
            assert current_generation is not None, "current_generation must be provided in kwargs for forest bandit mode"

            if current_generation == 0:
                N_initial_random = kwargs.get('N_initial_random', N)
                reporter(f"Current generation is {current_generation}, proposing {N_initial_random} random constraints for initial training")
                proposals = propose_random_constraints(occ_matr_list, natoms, N_initial_random, debug=debug, **kwargs)
            else:
                reporter(f"Current generation is {current_generation}, proposing {N} constraints using forest bandit (Random Forest)")
                try:
                    proposals = propose_forest_bandit_constraints(
                        occ_matr_list, energies, natoms, N, gp_config=rf_config,
                        debug=debug, reporter=reporter, **kwargs
                    )
                except Exception as e:
                    reporter(f"Error in forest bandit proposal generation: {e}")
                    import traceback
                    reporter(traceback.format_exc())
                    reporter("Falling back to random constraint proposals")
                    proposals = propose_random_constraints(occ_matr_list, natoms, N, debug=debug, **kwargs)

        case 'read':
            # raise implmementation error for now
            raise NotImplementedError("The 'read' mode needs to be implemented")

    
    return proposals