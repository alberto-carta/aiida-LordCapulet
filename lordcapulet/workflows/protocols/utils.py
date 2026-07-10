"""
Protocol support utilities for LordCapulet workchains.

What is a protocol?
-------------------
A *protocol* is a named preset of DFT input parameters stored in a YAML file.
Instead of writing out every parameter (k-points, cutoffs, pseudo family, …)
in every submission script, a workchain can call::

    inputs = MyWorkChain.get_protocol_inputs('default', overrides={...})

and get back a fully-populated dict of sane defaults, with only the values
specified in ``overrides`` changed.

What does ProtocolMixin do?
---------------------------
``ProtocolMixin`` is a "mixin" class - a small helper that is *not* meant to
be used on its own, but to be combined with ``WorkChain`` via multiple
inheritance::

    class StandardMagneticScanWorkChain(ProtocolMixin, WorkChain):
        ...

This gives ``StandardMagneticScanWorkChain`` all of AiiDA's workchain machinery (from
``WorkChain``) *plus* the YAML-loading methods (from ``ProtocolMixin``)
without duplicating any code.  Python resolves method names left-to-right
through the two parents, but there is no overlap between them.

``ProtocolMixin`` itself provides two things:

1. ``get_protocol_inputs(protocol, overrides)`` - reads the YAML files and
   applies a four-level merge to produce the final input dict.
2. ``get_available_protocols()`` - lists the named presets that exist for a
   given workchain.

It deliberately does *not* build the AiiDA ``ProcessBuilder``, because each
workchain has a different set of inputs.  That job belongs to each workchain's
own ``get_builder_from_protocol`` classmethod.

Merge order (later entries take priority)
-----------------------------------------
1. ``common.yaml`` ``default_inputs`` - shared DFT defaults used by every
   workchain (pseudo family, k-points mesh, ecutwfc, smearing, …).
2. Workchain-specific YAML (e.g. ``standard_magnetic_scan.yaml``) top-level keys - values
   that apply to this workchain regardless of which protocol is chosen
   (e.g. the default ``magnitude`` for the AFM scan).
3. Protocol section inside the workchain YAML (e.g. ``protocols.default``) -
   protocol-specific overrides; today we only ship one protocol (``default``),
   but new ones (e.g. ``fast``, ``accurate``) can be added by editing the YAML.
4. Caller-supplied ``overrides`` dict - highest priority; these come from the
   submission script and override everything else.

Example
-------
``common.yaml`` sets ``parameters.SYSTEM.ecutwfc = 60``.
``standard_magnetic_scan.yaml`` sets ``walltime_hours = 2.0``.
The caller passes ``overrides={'walltime_hours': 1.0}``.
Result: ecutwfc=60 (from common), walltime_hours=1.0 (caller wins over YAML).
"""
from __future__ import annotations

import copy
from typing import Optional


def recursive_merge(base: dict, update: dict) -> dict:
    """Return a new dict formed by recursively merging *update* into *base*.

    Works like ``{**base, **update}`` but handles nested dicts properly:
    instead of replacing a nested dict wholesale, it descends into it and
    merges key-by-key.  This means a caller can write::

        overrides = {'parameters': {'SYSTEM': {'ecutwfc': 50.0}}}

    and only ``ecutwfc`` will change; all other ``SYSTEM`` keys from the
    protocol YAML are preserved.

    Neither *base* nor *update* is modified in place.
    """
    result = copy.deepcopy(base)
    for key, val in update.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            # Both sides have a dict at this key → recurse instead of replacing
            result[key] = recursive_merge(result[key], val)
        else:
            # Scalar, list, or new key → take the update value verbatim
            result[key] = copy.deepcopy(val)
    return result


class ProtocolMixin:
    """Mixin that adds YAML-driven protocol support to a WorkChain.

    Usage
    -----
    Inherit from both ``ProtocolMixin`` and ``WorkChain``::

        class StandardMagneticScanWorkChain(ProtocolMixin, WorkChain):
            ...

    Then implement two classmethods in the workchain:

    1. ``get_protocol_filepath`` (required) - returns the path to the
       workchain's own YAML file so ``ProtocolMixin`` knows where to read it.
    2. ``get_builder_from_protocol`` (recommended) - uses
       ``get_protocol_inputs`` to build a fully populated
       ``ProcessBuilder``.

    The mixin itself never touches AiiDA types (``Dict``, ``KpointsData``,
    etc.).  It only reads YAML and returns plain Python dicts.  All
    AiiDA-specific wrapping is done in each workchain's own
    ``get_builder_from_protocol``.
    """

    @classmethod
    def get_protocol_filepath(cls):
        """Return the path to the workchain-specific YAML protocol file.

        This is the one method that every subclass **must** override.  It
        should return an ``importlib_resources`` ``Traversable`` (or any
        path-like object that supports ``.open()``) pointing to the YAML
        file that holds the defaults for this workchain.

        Example implementation::

            from importlib_resources import files
            import lordcapulet.workflows.protocols as pkg
            return files(pkg) / 'standard_magnetic_scan.yaml'
        """
        raise NotImplementedError(
            f"{cls.__name__} must implement get_protocol_filepath()"
        )

    @classmethod
    def _load_raw_yaml(cls):
        """Load and return ``(common_dict, workchain_dict)`` from disk.

        Reads two YAML files:
        - ``common.yaml``            - shared defaults for all workchains
        - the workchain-specific one - returned by ``get_protocol_filepath``

        Returns a tuple of two plain Python dicts.  This is kept as a
        separate private method so it can be called cheaply by both
        ``get_available_protocols`` and ``get_protocol_inputs`` without
        loading from disk multiple times in a hot path.
        """
        import yaml
        from importlib_resources import files
        import lordcapulet.workflows.protocols as protocols_pkg

        common_path = files(protocols_pkg) / 'common.yaml'
        with common_path.open() as fh:
            common = yaml.safe_load(fh) or {}

        wc_path = cls.get_protocol_filepath()
        with wc_path.open() as fh:
            wc_data = yaml.safe_load(fh) or {}

        return common, wc_data

    @classmethod
    def get_available_protocols(cls) -> dict:
        """Return ``{protocol_name: description_string}`` for this workchain.

        Useful for discovering what named presets are available before
        calling ``get_builder_from_protocol``::

            StandardMagneticScanWorkChain.get_available_protocols()
            # → {'default': 'Standard AFM scan with moderate cutoffs'}
        """
        _, wc_data = cls._load_raw_yaml()
        return {
            name: (data or {}).get('description', '')
            for name, data in wc_data.get('protocols', {}).items()
        }

    @classmethod
    def get_protocol_inputs(
        cls,
        protocol: Optional[str] = None,
        overrides: Optional[dict] = None,
    ) -> dict:
        """Return a fully-merged plain Python dict of inputs for *protocol*.

        This is the core method of ``ProtocolMixin``.  It applies the
        four-level merge described in the module docstring and returns a
        ``dict`` that ``get_builder_from_protocol`` can then use to
        populate an AiiDA ``ProcessBuilder``.

        :param protocol: Named preset to load (e.g. ``'default'``).  If
            ``None``, the ``default_protocol`` key in the workchain YAML is
            used as a fallback.
        :param overrides: Optional ``dict`` of values that take highest
            priority and are deep-merged on top of everything else.  Nested
            dicts are merged key-by-key (see ``recursive_merge``), so you
            can change a single sub-key like
            ``{'parameters': {'SYSTEM': {'ecutwfc': 50}}}`` without
            affecting the other ``SYSTEM`` keys.
        :returns: Flat-ish Python dict ready for use in
            ``get_builder_from_protocol``.
        :raises ValueError: If *protocol* is not defined in the YAML.
        """
        # These keys are YAML structure/metadata, not DFT inputs
        _META_KEYS = {'protocols', 'default_protocol'}

        common, wc_data = cls._load_raw_yaml()

        # Resolve protocol name from YAML default if not supplied by caller
        if protocol is None:
            protocol = wc_data.get('default_protocol', 'default')

        available = wc_data.get('protocols', {})
        if protocol not in available:
            raise ValueError(
                f"Protocol '{protocol}' is not defined for {cls.__name__}. "
                f"Available protocols: {list(available.keys())}"
            )

        # Step 1 - start from the shared common defaults
        inputs = copy.deepcopy(common.get('default_inputs', {}))

        # Step 2 - layer workchain-level defaults on top
        # (everything in the workchain YAML except the protocol definitions
        # themselves, e.g. `magnitude`, `walltime_hours`)
        wc_defaults = {k: v for k, v in wc_data.items() if k not in _META_KEYS}
        inputs = recursive_merge(inputs, wc_defaults)

        # Step 3 - layer the named protocol's overrides on top
        # Skip the human-readable 'description' key
        proto_data = {
            k: v
            for k, v in (available[protocol] or {}).items()
            if k != 'description'
        }
        if proto_data:
            inputs = recursive_merge(inputs, proto_data)

        # Step 4 - finally apply the caller's own overrides (highest priority)
        if overrides:
            inputs = recursive_merge(inputs, overrides)

        return inputs


def make_kpoints(inputs: dict, structure) -> 'KpointsData':
    """Build a :class:`~aiida.orm.KpointsData` from a merged protocol inputs dict.

    Two modes are supported, checked in this order:

    1. **Explicit mesh** - if ``'kpoints_mesh'`` is present in *inputs*
       (e.g. supplied via ``overrides={'kpoints_mesh': [4, 4, 4]}``),
       ``set_kpoints_mesh`` is called directly and the structure is ignored.

    2. **Density-based mesh** (default) - ``'kpoints_distance'`` is read
       from *inputs* and :meth:`KpointsData.set_kpoints_mesh_from_density`
       is called.  This mirrors the aiida-quantumespresso approach and
       correctly handles anisotropic cells: the number of k-points along
       each reciprocal lattice vector is chosen so that the spacing never
       exceeds the requested distance.  A cubic cell with distance 0.15 Å⁻¹
       and a typical ~4 Å lattice constant gives roughly a 4×4×4 mesh; a
       flattened or hexagonal cell will get a different mesh per direction.

    Args:
        inputs: merged protocol dict as returned by
            :meth:`ProtocolMixin.get_protocol_inputs`.
        structure: AiiDA :class:`~aiida.orm.StructureData` (only used in
            density mode, so that the reciprocal lattice vectors are known).

    Returns:
        A :class:`~aiida.orm.KpointsData` with the mesh set.
    """
    from aiida.orm import KpointsData

    kpoints = KpointsData()
    if 'kpoints_mesh' in inputs:
        # Caller explicitly requested a fixed mesh - honour it exactly.
        kpoints.set_kpoints_mesh(inputs['kpoints_mesh'])
    else:
        # Density-based: derive mesh from the structure's reciprocal lattice.
        kpoints.set_cell_from_structure(structure)
        kpoints.set_kpoints_mesh_from_density(inputs['kpoints_distance'])
    return kpoints
