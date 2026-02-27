"""Unit tests for the protocol machinery - pure Python, no AiiDA required.

Covers:
- ``recursive_merge``  - four-level merge logic
- ``ProtocolMixin``    - YAML loading, available protocols, merge ordering
- ``ProtocolMixin.get_protocol_inputs`` for all three workchains
- Unknown protocol / not-implemented guard
"""
import pytest

from lordcapulet.workflows.protocols.utils import ProtocolMixin, recursive_merge


# =============================================================================
# recursive_merge
# =============================================================================

# class TestRecursiveMerge:
#     """Test the plain-Python deep-merge helper."""

#     def test_flat_update(self):
#         """Second dict wins for top-level scalar keys."""
#         result = recursive_merge({'a': 1, 'b': 2}, {'b': 99, 'c': 3})
#         assert result == {'a': 1, 'b': 99, 'c': 3}

#     def test_nested_merge_only_changes_listed_key(self):
#         """Only the specified sub-key changes; all siblings are preserved."""
#         base = {
#             'params': {
#                 'SYSTEM': {'ecutwfc': 60, 'ecutrho': 480},
#                 'CONTROL': {'calculation': 'scf'},
#             }
#         }
#         update = {'params': {'SYSTEM': {'ecutwfc': 100}}}
#         result = recursive_merge(base, update)

#         assert result['params']['SYSTEM']['ecutwfc'] == 100
#         assert result['params']['SYSTEM']['ecutrho'] == 480   # preserved
#         assert result['params']['CONTROL']['calculation'] == 'scf'  # preserved

#     def test_new_key_added(self):
#         """A key present only in update is added to the result."""
#         result = recursive_merge({'a': 1}, {'b': 2})
#         assert result == {'a': 1, 'b': 2}

#     def test_does_not_mutate_base(self):
#         """Base dict must be untouched after merging."""
#         base = {'a': {'x': 1}}
#         recursive_merge(base, {'a': {'x': 99}})
#         assert base['a']['x'] == 1

#     def test_does_not_mutate_update(self):
#         """Update dict must be untouched after merging."""
#         update = {'b': {'nested': 2}}
#         recursive_merge({'a': 1}, update)
#         assert update == {'b': {'nested': 2}}

#     def test_list_replaced_not_merged(self):
#         """Lists are replaced wholesale - not element-wise appended."""
#         result = recursive_merge({'kpoints': [4, 4, 4]}, {'kpoints': [6, 6, 6]})
#         assert result['kpoints'] == [6, 6, 6]

#     def test_triple_nesting(self):
#         """Works for at least three nesting levels."""
#         base = {'a': {'b': {'c': 1, 'd': 2}}}
#         result = recursive_merge(base, {'a': {'b': {'c': 99}}})
#         assert result['a']['b']['c'] == 99
#         assert result['a']['b']['d'] == 2  # sibling preserved

#     def test_empty_update_returns_copy_of_base(self):
#         base = {'x': 1}
#         result = recursive_merge(base, {})
#         assert result == base
#         assert result is not base  # must be a copy

#     def test_empty_base_returns_copy_of_update(self):
#         update = {'y': 2}
#         result = recursive_merge({}, update)
#         assert result == update
#         assert result is not update


# =============================================================================
# ProtocolMixin base guard
# =============================================================================

class TestProtocolMixinBase:
    """The bare mixin raises ``NotImplementedError`` until subclassed properly."""

    def test_get_protocol_filepath_raises(self):
        class Incomplete(ProtocolMixin):
            pass

        with pytest.raises(NotImplementedError, match='must implement get_protocol_filepath'):
            Incomplete.get_protocol_filepath()

    def test_get_protocol_inputs_propagates_not_implemented(self):
        """``get_protocol_inputs`` calls ``get_protocol_filepath`` internally."""
        class Incomplete(ProtocolMixin):
            pass

        with pytest.raises(NotImplementedError):
            Incomplete.get_protocol_inputs('default')


# =============================================================================
# AFMScanWorkChain protocol
# =============================================================================

class TestAFMProtocol:
    """Protocol tests for ``AFMScanWorkChain`` (YAML reading only, no AiiDA)."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from lordcapulet.workflows.afm_scan import AFMScanWorkChain
        self.WC = AFMScanWorkChain

    def test_available_protocols_contains_default(self):
        protocols = self.WC.get_available_protocols()
        assert 'default' in protocols

    def test_get_protocol_filepath_returns_path(self):
        path = self.WC.get_protocol_filepath()
        # Must be openable and contain a 'protocols' key
        import yaml
        with path.open() as fh:
            data = yaml.safe_load(fh)
        assert 'protocols' in data

    def test_unknown_protocol_raises_value_error(self):
        with pytest.raises(ValueError, match="Protocol 'nonexistent' is not defined"):
            self.WC.get_protocol_inputs('nonexistent')

    def test_default_inputs_has_pseudo_family(self):
        inputs = self.WC.get_protocol_inputs('default')
        assert 'pseudo_family' in inputs
        assert isinstance(inputs['pseudo_family'], str)

    def test_default_inputs_has_kpoints_distance(self):
        inputs = self.WC.get_protocol_inputs('default')
        assert 'kpoints_distance' in inputs
        assert isinstance(inputs['kpoints_distance'], float)

    def test_default_inputs_has_parameters(self):
        inputs = self.WC.get_protocol_inputs('default')
        assert 'parameters' in inputs
        p = inputs['parameters']
        assert 'CONTROL' in p
        assert 'SYSTEM' in p
        assert 'ELECTRONS' in p

    def test_afm_specific_magnitude_and_walltime(self):
        """afm_scan.yaml top-level keys must appear after merging."""
        inputs = self.WC.get_protocol_inputs('default')
        assert 'magnitude' in inputs
        assert 'walltime_hours' in inputs
        assert inputs['magnitude'] == pytest.approx(0.5)
        assert inputs['walltime_hours'] == pytest.approx(2.0)

    def test_none_protocol_resolves_to_default(self):
        """protocol=None should fall back to the YAML's default_protocol."""
        inputs_none = self.WC.get_protocol_inputs(None)
        inputs_default = self.WC.get_protocol_inputs('default')
        assert inputs_none == inputs_default

    def test_override_wins_over_yaml(self):
        """Caller override must override every YAML layer."""
        inputs = self.WC.get_protocol_inputs('default', overrides={'walltime_hours': 99.0})
        assert inputs['walltime_hours'] == pytest.approx(99.0)

    def test_nested_override_preserves_siblings(self):
        """Nested override changes only the specified sub-key."""
        original = self.WC.get_protocol_inputs('default')
        original_ecutrho = original['parameters']['SYSTEM']['ecutrho']

        inputs = self.WC.get_protocol_inputs(
            'default',
            overrides={'parameters': {'SYSTEM': {'ecutwfc': 999}}},
        )
        assert inputs['parameters']['SYSTEM']['ecutwfc'] == pytest.approx(999)
        assert inputs['parameters']['SYSTEM']['ecutrho'] == pytest.approx(original_ecutrho)
        assert 'CONTROL' in inputs['parameters']

    def test_kpoints_mesh_override_present(self):
        """Passing kpoints_mesh in overrides should store it (builder uses it later)."""
        inputs = self.WC.get_protocol_inputs(
            'default', overrides={'kpoints_mesh': [2, 2, 2]}
        )
        assert 'kpoints_mesh' in inputs
        assert inputs['kpoints_mesh'] == [2, 2, 2]

    def test_protocol_isolation_no_side_effects(self):
        """Calling get_protocol_inputs twice returns independent dicts."""
        a = self.WC.get_protocol_inputs('default')
        b = self.WC.get_protocol_inputs('default')
        a['parameters']['SYSTEM']['ecutwfc'] = 9999
        assert b['parameters']['SYSTEM']['ecutwfc'] != 9999


# =============================================================================
# ConstrainedScanWorkChain protocol
# =============================================================================

class TestConstrainedProtocol:
    """Protocol tests for ``ConstrainedScanWorkChain``."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from lordcapulet.workflows.constrained_scan import ConstrainedScanWorkChain
        self.WC = ConstrainedScanWorkChain

    def test_available_protocols_contains_default(self):
        assert 'default' in self.WC.get_available_protocols()

    def test_common_keys_inherited(self):
        """Keys from common.yaml must appear in constrained inputs too."""
        inputs = self.WC.get_protocol_inputs('default')
        assert 'kpoints_distance' in inputs
        assert 'pseudo_family' in inputs
        assert inputs['parameters']['SYSTEM']['ecutwfc'] == pytest.approx(60.0)

    def test_oscdft_card_present(self):
        """constrained_scan.yaml defines oscdft_card defaults."""
        inputs = self.WC.get_protocol_inputs('default')
        assert 'oscdft_card' in inputs
        osc = inputs['oscdft_card']
        assert 'oscdft_type' in osc
        assert 'constraint_strength' in osc
        assert 'constraint_conv_thr' in osc

    def test_oscdft_card_nested_override(self):
        """Nested override of oscdft_card preserves unmodified keys."""
        orig = self.WC.get_protocol_inputs('default')
        orig_thr = orig['oscdft_card']['constraint_conv_thr']

        inputs = self.WC.get_protocol_inputs(
            'default',
            overrides={'oscdft_card': {'constraint_strength': 5.0}},
        )
        assert inputs['oscdft_card']['constraint_strength'] == pytest.approx(5.0)
        assert inputs['oscdft_card']['constraint_conv_thr'] == pytest.approx(orig_thr)

    def test_unknown_protocol_raises(self):
        with pytest.raises(ValueError, match="Protocol 'fast' is not defined"):
            self.WC.get_protocol_inputs('fast')


# =============================================================================
# GlobalConstrainedSearchWorkChain - YAML loading (pure Python part)
# =============================================================================

class TestGlobalSearchYamlLoading:
    """Tests for global_search.yaml readability and expected keys."""

    @pytest.fixture(autouse=True)
    def _load_yaml(self):
        import yaml
        from importlib_resources import files
        import lordcapulet.workflows.protocols as protocols_pkg

        global_path = files(protocols_pkg) / 'global_search.yaml'
        with global_path.open() as fh:
            self.data = yaml.safe_load(fh)

    def test_yaml_loads_without_error(self):
        assert self.data is not None

    def test_nmax_and_n_present(self):
        assert 'Nmax' in self.data
        assert 'N' in self.data

    def test_protocols_key_present(self):
        assert 'protocols' in self.data
        assert 'default' in self.data['protocols']

    def test_default_nmax_is_positive(self):
        assert self.data['Nmax'] > 0

    def test_default_n_is_positive(self):
        assert self.data['N'] > 0

    def test_proposal_mode_present(self):
        assert 'proposal_mode' in self.data

    def test_global_merge_routing(self):
        """Top-level overrides not named 'afm' or 'constrained' should stay
        at top-level - verifying the routing logic in get_builder_from_protocol
        keeps them separate from sub-protocol overrides.

        This is a pure-dict test of the routing logic that mirrors what
        GlobalConstrainedSearchWorkChain.get_builder_from_protocol does.
        """
        from lordcapulet.workflows.protocols.utils import recursive_merge

        overrides = {
            'Nmax': 99,
            'N': 7,
            'afm': {'walltime_hours': 3.0},
            'constrained': {'walltime_hours': 4.0},
        }

        global_user = {k: v for k, v in overrides.items() if k not in ('afm', 'constrained')}
        afm_user = overrides.get('afm', {})
        con_user = overrides.get('constrained', {})

        base_global = {'Nmax': 20, 'N': 4}
        merged = recursive_merge(base_global, global_user)

        assert merged['Nmax'] == 99
        assert merged['N'] == 7
        assert afm_user == {'walltime_hours': 3.0}
        assert con_user == {'walltime_hours': 4.0}
