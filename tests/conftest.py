"""
Pytest configuration for LordCapulet tests.

Provides shared fixtures for unit and integration tests.
Uses aiida.tools.pytest_fixtures for ephemeral AiiDA profile management
(automatic temporary PostgreSQL via pgtest).
"""

import asyncio
import io
import os
import pathlib
import sys
import tempfile
from collections.abc import Mapping

import pytest
import numpy as np

# Add the parent directory to the path so we can import lordcapulet
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Enable AiiDA's built-in pytest plugin: ephemeral profile, aiida_localhost, aiida_code_installed
pytest_plugins = ['aiida.tools.pytest_fixtures']


# ============================================================================
# Test ordering
# ============================================================================

def _test_priority(item) -> int:
    """Return an integer priority for a collected test item (lower runs first).

    Tiers:
      1 — unit / utils          (pure Python, no AiiDA, no torch)
      2 — unit / functions      (pure Python)
      3 — unit / data_structures(pure Python)
      4 — unit / bayesian       (torch only, no AiiDA)
      5 — unit / calculations   (AiiDA, single-calc scope)
      6 — integration / workflows (AiiDA workflows)
      7 — integration / bayesian  (GP training, no @slow)
      8 — integration / bayesian  (GP pipeline, @slow)
    """
    path = str(item.fspath)
    is_slow = item.get_closest_marker('slow') is not None

    if 'unit_tests/utils' in path:
        return 1
    if 'unit_tests/functions' in path:
        return 2
    if 'unit_tests/data_structures' in path:
        return 3
    if 'unit_tests/bayesian' in path:
        return 4
    if 'unit_tests/calculations' in path:
        return 5
    if 'integration_tests/workflows' in path:
        return 6
    if 'integration_tests/bayesian' in path:
        return 8 if is_slow else 7
    return 9  # anything unexpected goes last


def pytest_collection_modifyitems(items):
    """Reorder collected tests by logical tier."""
    items.sort(key=_test_priority)


# ============================================================================
# Session-scoped Infrastructure
# ============================================================================

@pytest.fixture(scope='session', autouse=True)
def clean_asyncio_tasks():
    """Ensure clean shutdown of asyncio tasks at the end of the test session."""
    yield
    asyncio.run(asyncio.sleep(0))


@pytest.fixture(scope='session')
def filepath_tests():
    """Return the absolute filepath of the ``tests`` folder."""
    return os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def filepath_fixtures(filepath_tests):
    """Return the absolute filepath to the ``tests/fixtures`` directory."""
    return os.path.join(filepath_tests, 'fixtures')


# ============================================================================
# AiiDA Infrastructure Fixtures
# ============================================================================

@pytest.fixture
def fixture_sandbox():
    """Return an AiiDA ``SandboxFolder``."""
    from aiida.common.folders import SandboxFolder

    with SandboxFolder() as folder:
        yield folder


@pytest.fixture
def fixture_localhost(aiida_localhost):
    """Return a localhost ``Computer`` with 1 MPI proc."""
    localhost = aiida_localhost
    localhost.set_default_mpiprocs_per_machine(1)
    return localhost


@pytest.fixture
def fixture_code(aiida_code_installed):
    """Return an ``InstalledCode`` instance for the given entry point on localhost."""

    def _fixture_code(entry_point_name):
        return aiida_code_installed(
            label=f'test.{entry_point_name}',
            default_calc_job_plugin=entry_point_name,
        )

    return _fixture_code


@pytest.fixture
def generate_calc_job():
    """Fixture to instantiate a CalcJob and call ``prepare_for_submission``.

    Returns the ``CalcInfo`` object without ever running the calculation.
    """

    def _generate_calc_job(folder, entry_point_name, inputs=None):
        from aiida.engine.utils import instantiate_process
        from aiida.manage.manager import get_manager
        from aiida.plugins import CalculationFactory

        manager = get_manager()
        runner = manager.get_runner()

        process_class = CalculationFactory(entry_point_name)
        process = instantiate_process(runner, process_class, **inputs)

        return process.prepare_for_submission(folder)

    return _generate_calc_job


@pytest.fixture
def generate_calc_job_node(fixture_localhost):
    """Fixture to generate a mock ``CalcJobNode`` for testing tools/parsers.

    Creates a node with correctly linked inputs and optional retrieved ``FolderData``.
    """

    def flatten_inputs(inputs, prefix=''):
        """Flatten inputs recursively."""
        flat_inputs = []
        for key, value in inputs.items():
            if isinstance(value, Mapping):
                flat_inputs.extend(flatten_inputs(value, prefix=prefix + key + '__'))
            else:
                flat_inputs.append((prefix + key, value))
        return flat_inputs

    def _generate_calc_job_node(
        entry_point_name='base',
        computer=None,
        inputs=None,
        attributes=None,
    ):
        from aiida import orm
        from aiida.common import LinkType
        from aiida.plugins.entry_point import format_entry_point_string

        if computer is None:
            computer = fixture_localhost
        if inputs is None:
            inputs = {}

        entry_point = format_entry_point_string('aiida.calculations', entry_point_name)

        node = orm.CalcJobNode(computer=computer, process_type=entry_point)
        node.base.attributes.set('input_filename', 'aiida.in')
        node.base.attributes.set('output_filename', 'aiida.out')
        node.base.attributes.set('error_filename', 'aiida.err')
        node.set_option('resources', {'num_machines': 1, 'num_mpiprocs_per_machine': 1})
        node.set_option('max_wallclock_seconds', 1800)

        if attributes:
            node.base.attributes.set_many(attributes)

        if inputs:
            metadata = inputs.pop('metadata', {})
            options = metadata.get('options', {})

            for name, option in options.items():
                node.set_option(name, option)

            for link_label, input_node in flatten_inputs(inputs):
                input_node.store()
                node.base.links.add_incoming(
                    input_node, link_type=LinkType.INPUT_CALC, link_label=link_label
                )

        node.store()
        return node

    return _generate_calc_job_node


@pytest.fixture
def generate_workchain():
    """Generate an instance of a ``WorkChain`` without running it through the engine."""

    def _generate_workchain(entry_point, inputs):
        from aiida.engine.utils import instantiate_process
        from aiida.manage.manager import get_manager
        from aiida.plugins import WorkflowFactory

        if isinstance(entry_point, str):
            process_class = WorkflowFactory(entry_point)
        else:
            # Accept a class directly (useful for unregistered workchains)
            process_class = entry_point
        runner = get_manager().get_runner()
        return instantiate_process(runner, process_class, **inputs)

    return _generate_workchain


# ============================================================================
# Data Generation Fixtures
# ============================================================================

@pytest.fixture
def generate_structure():
    """Return a ``StructureData`` for testing."""

    def _generate_structure(structure_id='feo'):
        from aiida.orm import StructureData

        if structure_id == 'feo':
            a = 4.334
            cell = [[a, 0, 0], [0, a, 0], [0, 0, a]]
            structure = StructureData(cell=cell)
            structure.append_atom(position=(0.0, 0.0, 0.0), symbols='Fe', name='Fe')
            structure.append_atom(position=(a / 2, a / 2, a / 2), symbols='O', name='O')
        elif structure_id == 'nio':
            a = 4.17
            cell = [[a, 0, 0], [0, a, 0], [0, 0, a]]
            structure = StructureData(cell=cell)
            structure.append_atom(position=(0.0, 0.0, 0.0), symbols='Ni', name='Ni')
            structure.append_atom(position=(a / 2, a / 2, a / 2), symbols='O', name='O')
        elif structure_id == 'fe3o4':
            a = 8.394
            cell = [[a, 0, 0], [0, a, 0], [0, 0, a]]
            structure = StructureData(cell=cell)
            structure.append_atom(position=(0.0, 0.0, 0.0), symbols='Fe', name='Fe1')
            structure.append_atom(position=(a / 4, a / 4, a / 4), symbols='Fe', name='Fe2')
            structure.append_atom(position=(a / 2, a / 2, 0.0), symbols='Fe', name='Fe3')
            structure.append_atom(position=(a / 4, a / 2, a / 4), symbols='O', name='O1')
            structure.append_atom(position=(a / 2, a / 4, a / 4), symbols='O', name='O2')
            structure.append_atom(position=(0.0, a / 2, a / 2), symbols='O', name='O3')
            structure.append_atom(position=(a / 4, a / 4, a / 2), symbols='O', name='O4')
        else:
            raise KeyError(f'Unknown structure_id="{structure_id}"')
        return structure

    return _generate_structure


@pytest.fixture
def generate_kpoints_mesh():
    """Return a ``KpointsData`` node."""

    def _generate_kpoints_mesh(npoints):
        from aiida.orm import KpointsData

        kpoints = KpointsData()
        kpoints.set_kpoints_mesh([npoints] * 3)
        return kpoints

    return _generate_kpoints_mesh


@pytest.fixture(scope='session')
def generate_upf_data():
    """Return a ``UpfData`` instance for the given element."""
    from aiida_pseudo.data.pseudo import UpfData

    def _generate_upf_data(element: str, z_valence: float = 4.0) -> UpfData:
        content = f'<UPF version="2.0.1"><PP_HEADER\nelement="{element}"\nz_valence="{z_valence}"\n/></UPF>\n'
        stream = io.BytesIO(content.encode('utf-8'))
        return UpfData(stream, filename=f'{element}.upf')

    return _generate_upf_data


@pytest.fixture(scope='session')
def pseudo_family(generate_upf_data):
    """Create the SSSP pseudo potential family used by lordcapulet workflows."""
    from aiida.common.constants import elements
    from aiida_pseudo.data.pseudo.upf import UpfData
    from aiida_pseudo.groups.family import SsspFamily

    cutoffs = {}

    label = 'SSSP/1.3/PBEsol/efficiency'
    cutoff_values = (30.0, 240.0)

    with tempfile.TemporaryDirectory() as directory:
        for values in elements.values():
            element = values['symbol']

            actinides = (
                'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm',
                'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
            )
            if element in actinides:
                continue

            upf = generate_upf_data(element)
            dirpath = pathlib.Path(directory)
            filename = dirpath / f'{element}.upf'

            with open(filename, 'w+b') as handle, upf.open(mode='rb') as source:
                handle.write(source.read())
                handle.flush()

            cutoffs[element] = {
                'cutoff_wfc': cutoff_values[0],
                'cutoff_rho': cutoff_values[1],
            }

        family = SsspFamily.create_from_folder(dirpath, label)

    family.set_cutoffs(cutoffs, 'standard', unit='Ry')
    return family


@pytest.fixture
def generate_inputs_constrained_pw(fixture_code, generate_structure, generate_kpoints_mesh, pseudo_family):
    """Assemble valid inputs for ``ConstrainedPWCalculation``."""

    def _generate_inputs(structure_id='feo', with_jsonable=True):
        import copy
        from aiida.orm import Dict, JsonableData
        from lordcapulet.data_structures.occupation_matrix import OccupationMatrixData

        structure = generate_structure(structure_id)
        kpoints = generate_kpoints_mesh(4)
        code = fixture_code('lordcapulet.constrained_pw')

        pseudos = pseudo_family.get_pseudos(structure=structure)

        parameters = Dict({
            'CONTROL': {
                'calculation': 'scf',
            },
            'SYSTEM': {
                'ecutwfc': 30.0,
                'ecutrho': 240.0,
                'occupations': 'smearing',
                'smearing': 'gaussian',
                'degauss': 0.02,
                'nspin': 2,
                'starting_magnetization': {'Fe': 0.5},
            },
            'ELECTRONS': {
                'conv_thr': 1.0e-6,
            },
        })

        oscdft_card = Dict({
            'nconstr': 1,
            'debug_print': '.FALSE.',
        })

        # Build a simple target matrix
        dim = 5
        occ_data = {
            'Atom_1': {
                'specie': 'Fe',
                'shell': '3d',
                'occupation_matrix': {
                    'up': np.diag([1.0, 0.8, 0.5, 0.2, 0.0]).tolist(),
                    'down': np.diag([0.9, 0.7, 0.4, 0.1, 0.0]).tolist(),
                },
            }
        }

        if with_jsonable:
            # Deep-copy because as_dict() returns a reference and
            # JsonableData.__init__ mutates it by adding @class/@module keys
            occ_matrix_data = OccupationMatrixData(copy.deepcopy(occ_data))
            target_matrix = JsonableData(occ_matrix_data)
            # Clear cached obj so next .obj access reconstructs via from_dict,
            # which pops @class/@module — mimicking the DB round-trip behaviour
            del target_matrix._obj
        else:
            # Legacy Dict format
            constrained_format = OccupationMatrixData(copy.deepcopy(occ_data)).to_constrained_matrix_format()
            target_matrix = Dict(constrained_format)

        inputs = {
            'code': code,
            'structure': structure,
            'kpoints': kpoints,
            'pseudos': pseudos,
            'parameters': parameters,
            'oscdft_card': oscdft_card,
            'target_matrix': target_matrix,
            'metadata': {
                'options': {
                    'resources': {'num_machines': 1, 'num_mpiprocs_per_machine': 1},
                    'max_wallclock_seconds': 3600,
                    'withmpi': False,
                },
            },
        }

        return inputs

    return _generate_inputs


# ============================================================================
# Math/Geometry Fixtures
# ============================================================================

@pytest.fixture
def sample_rotation_matrix():
    """Fixture providing a sample rotation matrix for tests."""
    return np.array([
        [0, 0, 1, 0, 0],
        [0, -1j/np.sqrt(2), 0, 1j/np.sqrt(2), 0],
        [0, 1/np.sqrt(2), 0, 1/np.sqrt(2), 0],
        [-1j/np.sqrt(2), 0, 0, 0, 1j/np.sqrt(2)],
        [1/np.sqrt(2), 0, 0, 0, 1/np.sqrt(2)]
    ])


# ============================================================================
# DataBank Fixtures (for all tests)
# ============================================================================

@pytest.fixture
def mock_occupation_matrix_data():
    """
    Create simple occupation matrices for testing.
    
    Structure: 2 atoms with 5x5 diagonal matrices (d-orbitals).
    """
    from lordcapulet.data_structures.occupation_matrix import OccupationMatrixData
    
    data = {
        'atom1': {
            'specie': 'Fe1',
            'shell': '3d',
            'occupation_matrix': {
                'up': np.diag([1.0, 0.8, 0.5, 0.2, 0.0]),
                'down': np.diag([0.9, 0.7, 0.4, 0.1, 0.0])
            }
        },
        'atom2': {
            'specie': 'Fe2',
            'shell': '3d',
            'occupation_matrix': {
                'up': np.diag([0.8, 0.6, 0.4, 0.2, 0.1]),
                'down': np.diag([0.7, 0.5, 0.3, 0.1, 0.0])
            }
        }
    }
    return OccupationMatrixData(data)


@pytest.fixture
def simple_databank(mock_occupation_matrix_data):
    """
    Real DataBank with simple test data.
    
    Use this for ALL unit tests. It's a real DataBank object,
    just initialized with simple diagonal matrices instead of
    loading from JSON.
    
    Contains:
    - 2 atoms ('atom1', 'atom2')  
    - 2 spins (up, down)
    - 5x5 diagonal matrices (d-orbitals)
    - 2 calculation records
    - All DataBank methods work normally
    """
    from lordcapulet.data_structures.databank import DataBank
    
    records = [
        {
            'pk': 1,
            'energy': -100.5,
            'energy_uncertainty': 0.0,
            'converged': True,
            'occ_data': mock_occupation_matrix_data,
            'metadata': {}
        },
        {
            'pk': 2,
            'energy': -99.8,
            'energy_uncertainty': 0.0,
            'converged': True,
            'occ_data': mock_occupation_matrix_data,
            'metadata': {}
        }
    ]
    
    return DataBank(records=records)


# Backward compatibility aliases (tests still using old names)
mock_databank_minimal = simple_databank
mock_databank_with_data = simple_databank


# ============================================================================
# PyTorch Fixtures
# ============================================================================

@pytest.fixture
def torch_device():
    """Get available torch device (CPU or CUDA)."""
    import torch
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def sample_train_data(torch_device):
    """Create sample training data for GP models."""
    import torch
    
    # Simple 2D input, 1D output
    train_X = torch.tensor([
        [0.2, 0.3],
        [0.4, 0.5],
        [0.6, 0.7],
        [0.8, 0.9],
    ], dtype=torch.float64, device=torch_device)
    
    train_Y = torch.tensor([
        [-1.5],
        [-1.2],
        [-0.9],
        [-0.6],
    ], dtype=torch.float64, device=torch_device)
    
    return train_X, train_Y


# ============================================================================
# Real Data Fixtures (for integration tests)
# ============================================================================

@pytest.fixture
def real_databank_path():
    """Path to real test databank JSON file."""
    # Use one of the example files
    import os
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, 'examples', 'FeO_scan_data_extractor.json')


@pytest.fixture
def real_databank(real_databank_path):
    """Load a real DataBank from JSON (for integration tests)."""
    from lordcapulet.data_structures.databank import DataBank
    import os
    
    # Only load if file exists (skip if not available)
    if not os.path.exists(real_databank_path):
        pytest.skip(f"Real data file not found: {real_databank_path}")
    
    return DataBank.from_json(real_databank_path, only_converged=True)


@pytest.fixture
def gp_databank():
    """
    Real DataBank for GP integration testing.
    
    Loads FeO scan data from test fixtures directory.
    This is a curated dataset specifically for testing Bayesian optimization.
    
    Use this fixture for:
    - GP model training tests
    - GP inference tests
    - LOO validation tests
    - End-to-end BO pipeline tests
    """
    from lordcapulet.data_structures.databank import DataBank
    import os
    
    # Path to the FeO data in test fixtures
    base_path = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(
        base_path, 
        'fixtures',
        'FeO_scan_data_extractor_redone.json'
    )
    
    if not os.path.exists(json_path):
        pytest.skip(f"GP test data not found: {json_path}")
    
    return DataBank.from_json(json_path, only_converged=True)


@pytest.fixture
def gp_databank_small(gp_databank):
    """
    Small subset of GP databank for faster integration tests.
    
    Returns first 50 calculations for quick smoke tests.
    """
    return gp_databank[:50] if len(gp_databank) > 50 else gp_databank
