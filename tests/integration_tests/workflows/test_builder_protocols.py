"""Integration tests for the protocol-driven builder API.

Covers:
- ``make_kpoints``  - density mode and explicit-mesh mode
- ``AFMScanWorkChain.get_builder_from_protocol``
    * builder field values match the YAML protocol inputs
    * equivalent to an equivalent manually assembled builder
    * overrides propagate correctly (scalar and nested)
- ``ConstrainedScanWorkChain.get_builder_from_protocol``
    * ``n_oscdft`` is auto-computed from tm_atoms
    * oscdft_card overrides work
- ``GlobalConstrainedSearchWorkChain.get_builder_from_protocol``
    * AFM + constrained sub-workchain fields come from their respective YAML
    * 'afm' override routes only to the AFM namespace
    * 'constrained' override routes only to the constrained namespace
    * top-level overrides set Nmax, N, etc.

All tests require an active AiiDA profile (via ``aiida_localhost`` / ``fixture_code``).
"""
import pytest
import yaml
from importlib_resources import files

from aiida.orm import Dict, Float, List, Str, KpointsData, Int

import lordcapulet.workflows.protocols as protocols_pkg
from lordcapulet.workflows.afm_scan import AFMScanWorkChain
from lordcapulet.workflows.constrained_scan import ConstrainedScanWorkChain
from lordcapulet.workflows.global_constrained_search import GlobalConstrainedSearchWorkChain
from lordcapulet.workflows.protocols.utils import make_kpoints


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _global_yaml():
    """Return parsed global_search.yaml as a plain dict."""
    path = files(protocols_pkg) / 'global_search.yaml'
    with path.open() as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# make_kpoints
# ---------------------------------------------------------------------------

class TestMakeKpoints:
    """Tests for the ``make_kpoints`` helper (needs AiiDA StructureData)."""

    def test_density_mode_returns_kpoints_data(self, generate_structure):
        structure = generate_structure('feo')
        inputs = {'kpoints_distance': 0.4}
        kpoints = make_kpoints(inputs, structure)
        assert isinstance(kpoints, KpointsData)

    def test_density_mode_produces_positive_mesh(self, generate_structure):
        structure = generate_structure('feo')
        inputs = {'kpoints_distance': 0.4}
        kpoints = make_kpoints(inputs, structure)
        mesh, _offset = kpoints.get_kpoints_mesh()
        assert all(m > 0 for m in mesh)

    def test_explicit_mesh_honoured_exactly(self, generate_structure):
        structure = generate_structure('feo')
        inputs = {'kpoints_mesh': [3, 4, 5]}
        kpoints = make_kpoints(inputs, structure)
        mesh, _offset = kpoints.get_kpoints_mesh()
        assert list(mesh) == [3, 4, 5]

    def test_explicit_mesh_ignores_distance_key(self, generate_structure):
        """When kpoints_mesh is given, kpoints_distance must be ignored."""
        structure = generate_structure('feo')
        inputs = {'kpoints_mesh': [2, 2, 2], 'kpoints_distance': 0.001}
        kpoints = make_kpoints(inputs, structure)
        mesh, _offset = kpoints.get_kpoints_mesh()
        assert list(mesh) == [2, 2, 2]

    def test_density_finer_gives_more_kpoints(self, generate_structure):
        """A smaller kpoints_distance should produce at least as many k-points."""
        structure = generate_structure('feo')
        coarse = make_kpoints({'kpoints_distance': 0.8}, structure)
        fine = make_kpoints({'kpoints_distance': 0.2}, structure)
        mesh_c, _ = coarse.get_kpoints_mesh()
        mesh_f, _ = fine.get_kpoints_mesh()
        # Fine mesh should have more or equal k-points in every direction
        assert all(f >= c for f, c in zip(mesh_f, mesh_c))


# ---------------------------------------------------------------------------
# AFMScanWorkChain.get_builder_from_protocol
# ---------------------------------------------------------------------------

class TestAFMBuilderFromProtocol:
    """Builder values must match what ``get_protocol_inputs`` returns."""

    @pytest.fixture
    def afm_builder(self, fixture_code, generate_structure):
        code = fixture_code('quantumespresso.pw')
        structure = generate_structure('feo')
        return AFMScanWorkChain.get_builder_from_protocol(
            code, structure, ['Fe1']
        )

    @pytest.fixture
    def afm_inputs(self):
        return AFMScanWorkChain.get_protocol_inputs('default')

    # -- agreement with protocol inputs ---

    def test_parameters_match_protocol(self, afm_builder, afm_inputs):
        assert afm_builder.parameters.get_dict() == afm_inputs['parameters']

    def test_magnitude_matches_protocol(self, afm_builder, afm_inputs):
        assert float(afm_builder.magnitude) == pytest.approx(afm_inputs['magnitude'])

    def test_walltime_matches_protocol(self, afm_builder, afm_inputs):
        assert float(afm_builder.walltime_hours) == pytest.approx(afm_inputs['walltime_hours'])

    def test_pseudo_family_string_matches_protocol(self, afm_builder, afm_inputs):
        assert afm_builder.pseudo_family_string.value == afm_inputs['pseudo_family']

    def test_tm_atoms_preserved(self, afm_builder):
        assert afm_builder.tm_atoms.get_list() == ['Fe1']

    def test_kpoints_is_valid(self, afm_builder):
        mesh, _ = afm_builder.kpoints.get_kpoints_mesh()
        assert all(m > 0 for m in mesh)

    # -- equivalence with manually assembled builder ---

    def test_equivalent_to_manual_builder(self, fixture_code, generate_structure):
        """Protocol builder and manually assembled builder must agree on all DFT fields."""
        code = fixture_code('quantumespresso.pw')
        structure = generate_structure('feo')
        tm_atoms = ['Fe1']

        inputs = AFMScanWorkChain.get_protocol_inputs('default')
        kpoints = make_kpoints(inputs, structure)

        # Protocol-driven builder
        pb = AFMScanWorkChain.get_builder_from_protocol(code, structure, tm_atoms)

        # Manually assembled
        mb = AFMScanWorkChain.get_builder()
        mb.code = code
        mb.structure = structure
        mb.kpoints = kpoints
        mb.parameters = Dict(dict=inputs['parameters'])
        mb.tm_atoms = List(list=tm_atoms)
        mb.magnitude = Float(inputs['magnitude'])
        mb.walltime_hours = Float(inputs['walltime_hours'])
        mb.pseudo_family_string = Str(inputs['pseudo_family'])

        assert pb.parameters.get_dict() == mb.parameters.get_dict()
        assert float(pb.magnitude) == pytest.approx(float(mb.magnitude))
        assert float(pb.walltime_hours) == pytest.approx(float(mb.walltime_hours))
        assert pb.pseudo_family_string.value == mb.pseudo_family_string.value
        assert pb.tm_atoms.get_list() == mb.tm_atoms.get_list()

    # -- override propagation ---

    def test_walltime_override_propagates(self, fixture_code, generate_structure):
        code = fixture_code('quantumespresso.pw')
        structure = generate_structure('feo')
        builder = AFMScanWorkChain.get_builder_from_protocol(
            code, structure, ['Fe1'],
            overrides={'walltime_hours': 9.5},
        )
        assert float(builder.walltime_hours) == pytest.approx(9.5)

    def test_magnitude_override_propagates(self, fixture_code, generate_structure):
        code = fixture_code('quantumespresso.pw')
        structure = generate_structure('feo')
        builder = AFMScanWorkChain.get_builder_from_protocol(
            code, structure, ['Fe1'],
            overrides={'magnitude': 0.9},
        )
        assert float(builder.magnitude) == pytest.approx(0.9)

    def test_nested_parameter_override_propagates(self, fixture_code, generate_structure):
        """Nested SYSTEM override must only change the specified key."""
        code = fixture_code('quantumespresso.pw')
        structure = generate_structure('feo')
        builder = AFMScanWorkChain.get_builder_from_protocol(
            code, structure, ['Fe1'],
            overrides={'parameters': {'SYSTEM': {'ecutwfc': 120.0}}},
        )
        params = builder.parameters.get_dict()
        assert params['SYSTEM']['ecutwfc'] == pytest.approx(120.0)
        # Sibling keys preserved
        default = AFMScanWorkChain.get_protocol_inputs('default')
        assert params['SYSTEM']['ecutrho'] == pytest.approx(
            default['parameters']['SYSTEM']['ecutrho']
        )
        assert params['CONTROL']['calculation'] == \
            default['parameters']['CONTROL']['calculation']

    def test_kpoints_mesh_override_bypasses_density(self, fixture_code, generate_structure):
        code = fixture_code('quantumespresso.pw')
        structure = generate_structure('feo')
        builder = AFMScanWorkChain.get_builder_from_protocol(
            code, structure, ['Fe1'],
            overrides={'kpoints_mesh': [3, 3, 3]},
        )
        mesh, _ = builder.kpoints.get_kpoints_mesh()
        assert list(mesh) == [3, 3, 3]

    def test_pseudo_family_override_propagates(self, fixture_code, generate_structure):
        code = fixture_code('quantumespresso.pw')
        structure = generate_structure('feo')
        builder = AFMScanWorkChain.get_builder_from_protocol(
            code, structure, ['Fe1'],
            overrides={'pseudo_family': 'mygroup/v1/PBE/efficiency'},
        )
        assert builder.pseudo_family_string.value == 'mygroup/v1/PBE/efficiency'


# ---------------------------------------------------------------------------
# ConstrainedScanWorkChain.get_builder_from_protocol
# ---------------------------------------------------------------------------

class TestConstrainedBuilderFromProtocol:
    """Builder values for the constrained scan workchain."""

    @pytest.fixture
    def con_builder(self, fixture_code, generate_structure):
        code = fixture_code('quantumespresso.pw')
        structure = generate_structure('feo')
        return ConstrainedScanWorkChain.get_builder_from_protocol(
            code, structure, ['Fe1'], occupation_matrices_list=[]
        )

    @pytest.fixture
    def con_inputs(self):
        return ConstrainedScanWorkChain.get_protocol_inputs('default')

    # -- agreement with protocol inputs ---

    def test_parameters_match_protocol(self, con_builder, con_inputs):
        assert con_builder.parameters.get_dict() == con_inputs['parameters']

    def test_walltime_matches_protocol(self, con_builder, con_inputs):
        assert float(con_builder.walltime_hours) == pytest.approx(con_inputs['walltime_hours'])

    def test_pseudo_family_string_matches_protocol(self, con_builder, con_inputs):
        assert con_builder.pseudo_family_string.value == con_inputs['pseudo_family']

    def test_tm_atoms_preserved(self, con_builder):
        assert con_builder.tm_atoms.get_list() == ['Fe1']

    def test_n_oscdft_computed_for_one_fe(self, con_builder):
        """1 Fe with 3d (5x5x2=50): n_oscdft must be 50."""
        osc = con_builder.oscdft_card.get_dict()
        assert osc['n_oscdft'] == 50

    def test_n_oscdft_computed_for_two_fe(self, fixture_code, generate_structure):
        """2 Fe atoms: n_oscdft must be 100."""
        code = fixture_code('quantumespresso.pw')
        structure = generate_structure('feo')
        builder = ConstrainedScanWorkChain.get_builder_from_protocol(
            code, structure, ['Fe1', 'Fe2'], occupation_matrices_list=[]
        )
        assert builder.oscdft_card.get_dict()['n_oscdft'] == 100

    def test_oscdft_card_defaults_present(self, con_builder, con_inputs):
        """oscdft_card from YAML must appear (minus n_oscdft which is computed)."""
        osc = con_builder.oscdft_card.get_dict()
        proto_osc = con_inputs['oscdft_card']
        for key in proto_osc:
            assert key in osc
            assert osc[key] == pytest.approx(proto_osc[key])

    # -- equivalence with manually assembled builder ---

    def test_equivalent_to_manual_builder(self, fixture_code, generate_structure):
        """Protocol builder and manually assembled builder agree on all shared fields."""
        from lordcapulet.utils.preprocessing.submission import get_default_manifolds, get_dimensions
        from lordcapulet.workflows.protocols.utils import recursive_merge

        code = fixture_code('quantumespresso.pw')
        structure = generate_structure('feo')
        tm_atoms = ['Fe1']

        inputs = ConstrainedScanWorkChain.get_protocol_inputs('default')
        kpoints = make_kpoints(inputs, structure)

        manifolds = get_default_manifolds(tm_atoms)
        n_oscdft = sum(get_dimensions(manifolds))
        oscdft_dict = dict(inputs['oscdft_card'])
        oscdft_dict['n_oscdft'] = n_oscdft

        pb = ConstrainedScanWorkChain.get_builder_from_protocol(
            code, structure, tm_atoms, occupation_matrices_list=[]
        )

        # Manual builder
        mb = ConstrainedScanWorkChain.get_builder()
        mb.code = code
        mb.structure = structure
        mb.kpoints = kpoints
        mb.parameters = Dict(dict=inputs['parameters'])
        mb.tm_atoms = List(list=tm_atoms)
        mb.occupation_matrices_list = List(list=[])
        mb.oscdft_card = Dict(dict=oscdft_dict)
        mb.walltime_hours = Float(inputs['walltime_hours'])
        mb.pseudo_family_string = Str(inputs['pseudo_family'])

        assert pb.parameters.get_dict() == mb.parameters.get_dict()
        assert pb.oscdft_card.get_dict() == mb.oscdft_card.get_dict()
        assert float(pb.walltime_hours) == pytest.approx(float(mb.walltime_hours))
        assert pb.pseudo_family_string.value == mb.pseudo_family_string.value

    # -- override propagation ---

    def test_oscdft_nested_override(self, fixture_code, generate_structure):
        """Nested oscdft_card override changes only the specified key."""
        code = fixture_code('quantumespresso.pw')
        structure = generate_structure('feo')
        builder = ConstrainedScanWorkChain.get_builder_from_protocol(
            code, structure, ['Fe1'], occupation_matrices_list=[],
            overrides={'oscdft_card': {'constraint_strength': 5.0}},
        )
        osc = builder.oscdft_card.get_dict()
        assert osc['constraint_strength'] == pytest.approx(5.0)
        # Other keys unchanged
        proto = ConstrainedScanWorkChain.get_protocol_inputs('default')['oscdft_card']
        assert osc['constraint_conv_thr'] == pytest.approx(proto['constraint_conv_thr'])

    def test_walltime_override_propagates(self, fixture_code, generate_structure):
        code = fixture_code('quantumespresso.pw')
        structure = generate_structure('feo')
        builder = ConstrainedScanWorkChain.get_builder_from_protocol(
            code, structure, ['Fe1'], occupation_matrices_list=[],
            overrides={'walltime_hours': 4.0},
        )
        assert float(builder.walltime_hours) == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# GlobalConstrainedSearchWorkChain.get_builder_from_protocol
# ---------------------------------------------------------------------------

class TestGlobalBuilderFromProtocol:
    """Protocol-driven builder for the global orchestration workchain."""

    @pytest.fixture
    def builder(self, fixture_code, generate_structure):
        code = fixture_code('quantumespresso.pw')
        structure = generate_structure('feo')
        return GlobalConstrainedSearchWorkChain.get_builder_from_protocol(
            code, structure, ['Fe1']
        )

    # -- sub-workchain parameters come from their own YAML --

    def test_afm_parameters_match_afm_protocol(self, builder):
        expected = AFMScanWorkChain.get_protocol_inputs('default')['parameters']
        assert builder.afm.parameters.get_dict() == expected

    def test_constrained_parameters_match_constrained_protocol(self, builder):
        expected = ConstrainedScanWorkChain.get_protocol_inputs('default')['parameters']
        assert builder.constrained.parameters.get_dict() == expected

    def test_afm_magnitude_from_afm_yaml(self, builder):
        expected = AFMScanWorkChain.get_protocol_inputs('default')['magnitude']
        assert float(builder.afm.magnitude) == pytest.approx(expected)

    def test_constrained_oscdft_card_defaults(self, builder):
        expected_osc = ConstrainedScanWorkChain.get_protocol_inputs('default')['oscdft_card']
        osc = builder.constrained.oscdft_card.get_dict()
        for key, val in expected_osc.items():
            assert osc[key] == pytest.approx(val)

    def test_n_oscdft_computed_in_constrained(self, builder):
        """n_oscdft for 1 Fe (3d) must be 50."""
        assert builder.constrained.oscdft_card.get_dict()['n_oscdft'] == 50

    # -- global parameters come from global_search.yaml --

    def test_nmax_from_global_yaml(self, builder):
        expected = _global_yaml()['Nmax']
        assert int(builder.Nmax) == expected

    def test_n_from_global_yaml(self, builder):
        expected = _global_yaml()['N']
        assert int(builder.N) == expected

    def test_proposal_mode_from_global_yaml(self, builder):
        expected = _global_yaml()['proposal_mode']
        assert builder.proposal_mode.value == expected

    # -- same structure object used for both sub-workchains --

    def test_same_structure_in_both_namespaces(self, fixture_code, generate_structure):
        code = fixture_code('quantumespresso.pw')
        structure = generate_structure('feo')
        builder = GlobalConstrainedSearchWorkChain.get_builder_from_protocol(
            code, structure, ['Fe1']
        )
        assert builder.afm.structure.uuid == structure.uuid
        assert builder.constrained.structure.uuid == structure.uuid

    # -- override routing --

    def test_afm_override_only_affects_afm_namespace(self, fixture_code, generate_structure):
        """'afm' override key must route exclusively to the AFM sub-workchain."""
        code = fixture_code('quantumespresso.pw')
        structure = generate_structure('feo')
        default_con_wt = ConstrainedScanWorkChain.get_protocol_inputs('default')['walltime_hours']

        builder = GlobalConstrainedSearchWorkChain.get_builder_from_protocol(
            code, structure, ['Fe1'],
            overrides={'afm': {'walltime_hours': 9.9}},
        )
        assert float(builder.afm.walltime_hours) == pytest.approx(9.9)
        assert float(builder.constrained.walltime_hours) == pytest.approx(default_con_wt)

    def test_constrained_override_only_affects_constrained_namespace(
        self, fixture_code, generate_structure
    ):
        """'constrained' override key must route exclusively to constrained sub-chain."""
        code = fixture_code('quantumespresso.pw')
        structure = generate_structure('feo')
        default_afm_wt = AFMScanWorkChain.get_protocol_inputs('default')['walltime_hours']

        builder = GlobalConstrainedSearchWorkChain.get_builder_from_protocol(
            code, structure, ['Fe1'],
            overrides={'constrained': {'walltime_hours': 8.8}},
        )
        assert float(builder.constrained.walltime_hours) == pytest.approx(8.8)
        assert float(builder.afm.walltime_hours) == pytest.approx(default_afm_wt)

    def test_nmax_override_propagates(self, fixture_code, generate_structure):
        code = fixture_code('quantumespresso.pw')
        structure = generate_structure('feo')
        builder = GlobalConstrainedSearchWorkChain.get_builder_from_protocol(
            code, structure, ['Fe1'],
            overrides={'Nmax': 77},
        )
        assert int(builder.Nmax) == 77

    def test_n_override_propagates(self, fixture_code, generate_structure):
        code = fixture_code('quantumespresso.pw')
        structure = generate_structure('feo')
        builder = GlobalConstrainedSearchWorkChain.get_builder_from_protocol(
            code, structure, ['Fe1'],
            overrides={'N': 13},
        )
        assert int(builder.N) == 13

    def test_afm_nested_param_override(self, fixture_code, generate_structure):
        """Nested parameter override inside 'afm' must not affect constrained."""
        code = fixture_code('quantumespresso.pw')
        structure = generate_structure('feo')
        default_con_ecut = ConstrainedScanWorkChain.get_protocol_inputs('default')[
            'parameters']['SYSTEM']['ecutwfc']

        builder = GlobalConstrainedSearchWorkChain.get_builder_from_protocol(
            code, structure, ['Fe1'],
            overrides={'afm': {'parameters': {'SYSTEM': {'ecutwfc': 200.0}}}},
        )
        assert builder.afm.parameters.get_dict()['SYSTEM']['ecutwfc'] == pytest.approx(200.0)
        assert builder.constrained.parameters.get_dict()['SYSTEM']['ecutwfc'] == pytest.approx(
            default_con_ecut
        )

    def test_dict_code_assignment(self, fixture_code, generate_structure):
        """Passing code as dict routes codes to respective sub-workchains."""
        afm_code = fixture_code('quantumespresso.pw')
        con_code = fixture_code('lordcapulet.constrained_pw')
        structure = generate_structure('feo')
        builder = GlobalConstrainedSearchWorkChain.get_builder_from_protocol(
            {'afm': afm_code, 'constrained': con_code},
            structure,
            ['Fe1'],
        )
        assert builder.afm.code.uuid == afm_code.uuid
        assert builder.constrained.code.uuid == con_code.uuid

    # -- equivalence: sub-workchain fields == standalone get_builder_from_protocol results --

    def test_afm_fields_match_standalone_afm_builder(self, fixture_code, generate_structure):
        """Global builder's AFM namespace must be identical to a standalone AFM builder."""
        code = fixture_code('quantumespresso.pw')
        structure = generate_structure('feo')
        tm_atoms = ['Fe1']

        global_builder = GlobalConstrainedSearchWorkChain.get_builder_from_protocol(
            code, structure, tm_atoms
        )
        standalone = AFMScanWorkChain.get_builder_from_protocol(
            code, structure, tm_atoms
        )

        assert global_builder.afm.parameters.get_dict() == standalone.parameters.get_dict()
        assert float(global_builder.afm.magnitude) == pytest.approx(float(standalone.magnitude))
        assert float(global_builder.afm.walltime_hours) == pytest.approx(
            float(standalone.walltime_hours)
        )
        assert global_builder.afm.pseudo_family_string.value == standalone.pseudo_family_string.value

    def test_constrained_fields_match_standalone_constrained_builder(
        self, fixture_code, generate_structure
    ):
        """Global builder's constrained namespace must match a standalone constrained builder."""
        code = fixture_code('quantumespresso.pw')
        structure = generate_structure('feo')
        tm_atoms = ['Fe1']

        global_builder = GlobalConstrainedSearchWorkChain.get_builder_from_protocol(
            code, structure, tm_atoms
        )
        standalone = ConstrainedScanWorkChain.get_builder_from_protocol(
            code, structure, tm_atoms, occupation_matrices_list=[]
        )

        assert global_builder.constrained.parameters.get_dict() == \
            standalone.parameters.get_dict()
        assert (
            global_builder.constrained.oscdft_card.get_dict() ==
            standalone.oscdft_card.get_dict()
        )
        assert float(global_builder.constrained.walltime_hours) == pytest.approx(
            float(standalone.walltime_hours)
        )
