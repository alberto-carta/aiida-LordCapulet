"""Utility functions for preprocessing before submission in LordCapulet.
"""

# default manifold for each atom

default_manifold = {
    # chalcogens use p orbitals
    'O': '2p',
    'S': '3p',
    'Se': '4p',
    'Te': '5p',
    # halogens use p orbitals
    'F': '2p',
    'Cl': '3p',
    'Br': '4p',
    'I': '5p',
    # transition metals use d orbitals
    'Sc': '3d',
    'Ti': '3d',
    'V': '3d',
    'Cr': '3d',
    'Mn': '3d',
    'Fe': '3d',
    'Co': '3d',
    'Ni': '3d',
    'Cu': '3d',
    'Zn': '3d',
    'Y': '4d',
    'Zr': '4d',
    'Nb': '4d',
    'Mo': '4d', 
    'Tc': '4d',
    'Ru': '4d',
    'Rh': '4d',
    'Pd': '4d',
    'Ag': '4d',
    'Cd': '4d',
    'La': '5d',
    'Hf': '5d',
    'Ta': '5d',
    'W': '5d',  
    'Re': '5d',
    'Os': '5d',
    'Ir': '5d',
    'Pt': '5d',
    'Au': '5d',
    'Hg': '5d',
    # the actinides use 5f orbitals
    'Ac': '5f',
    'Th': '5f',
    'Pa': '5f',
    'U': '5f',
    'Np': '5f',
    'Pu': '5f',
    'Am': '5f',
    # the lanthanides use 4f orbitals
    'Ce': '4f',
    'Pr': '4f',
    'Nd': '4f',
    'Pm': '4f',
    'Sm': '4f',
    'Eu': '4f',
    'Gd': '4f',
    'Tb': '4f',
    'Dy': '4f',
    'Ho': '4f',
    'Er': '4f', 
    'Tm': '4f',
    'Yb': '4f',
    'Lu': '4f',
}

# default dimensions for each manifold
default_dimensions = {
    's': 1,
    'p': 3,
    'd': 5,
    'f': 7,
}




#%%
def tag_and_list_atoms(atoms, table=None):
    """
    Tags atoms based on whether they are transition metals or other elements.
    Transition metals get a unique tag (e.g., Ni1, Mn2).
    Other elements get a tag based on their element symbol (e.g., O1, S1).
    These tags are stored in atom.info['custom_tag'].

    Args:
        atoms (list): A list of atom objects, assumed to be ASE Atom objects
                      or similar with 'symbol' and an 'info' dictionary attribute.
    """

    if table is None:
        table = {
            'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
            'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
            'La', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
            'Ac', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'U',
        }
    else:
        # assert that table is a set of strings
        assert isinstance(table, set), "table must be a set of element symbols as strings."

        for el in table:
            if not isinstance(el, str):
                raise ValueError("table must be a set of element symbols as strings.")

    tm_counts = {}
    other_counts = {}
    tm_atoms = []

    for atom in atoms:

        if atom.symbol in table:
            if atom.symbol not in tm_counts:
                tm_counts[atom.symbol] = 0
            
            tm_counts[atom.symbol] += 1
            # Store the custom string tag in atom.info
            atom.tag = tm_counts[atom.symbol]
            tm_atoms.append(f"{atom.symbol}{tm_counts[atom.symbol]}")
        else:
            if atom.symbol not in other_counts:
                other_counts[atom.symbol] = 1
            
            # Store the custom string tag in atom.info
            atom.tag = other_counts[atom.symbol]
    
    return tm_atoms

# function that gets a tm_atoms list and returns a list of default manifolds
def get_default_manifolds(tm_atoms):
    """
    Given a list of tagged transition metal atoms (e.g., ['Fe1', 'Ni1', 'Fe2']),
    returns a list of their default manifolds based on the predefined mapping.

    Args:
        tm_atoms (list): A list of tagged transition metal atom strings.

    Returns:
        list: A list of default manifolds corresponding to the input atoms.
    """
    manifolds = []
    for tm_atom in tm_atoms:
        element = ''.join(filter(str.isalpha, tm_atom))  # Extract element symbol
        manifold = default_manifold.get(element)
        if manifold is None:
            raise ValueError(f"No default manifold found for element: {element}")
        manifolds.append(manifold)
    
    return manifolds


# calculate dimensions for each manifold
# this is manifold_dim x manifold_dim x 2 to account for spin

def get_dimensions(manifolds):
    """
    Given a list of manifolds (e.g., ['3d', '4f', '3d']),
    returns a list of their corresponding dimensions based on the predefined mapping.

    Args:
        manifolds (list): A list of manifold strings.

    Returns:
        list: A list of dimensions corresponding to the input manifolds.
    """
    dimensions = []
    for manifold in manifolds:
        orbital_type = manifold[1]  # e.g., 'd' from '3d'
        dim = default_dimensions.get(orbital_type)
        if dim is None:
            raise ValueError(f"No default dimension found for manifold: {manifold}")
        dimensions.append(dim * dim * 2)  # account for spin
    
    return dimensions


def prepare_tm_info(atoms, table=None):
    """Condense the standard transition-metal preprocessing block into one call.

    Replaces the four-line boilerplate::

        tm_atoms      = tag_and_list_atoms(atoms, table=table)
        tm_manifolds  = get_default_manifolds(tm_atoms)
        tm_dimensions = get_dimensions(tm_manifolds)
        total_dimensions = sum(tm_dimensions)

    Args:
        atoms: ASE :class:`~ase.Atoms` object to inspect.
        table: optional set of element symbols to treat as transition metals;
            defaults to all TMs handled by :func:`tag_and_list_atoms`.

    Returns:
        tuple: ``(tm_atoms, tm_manifolds, tm_dimensions)`` where

        * ``tm_atoms``      - list of tagged species strings, e.g. ``['Fe1', 'Fe2']``
        * ``tm_manifolds``  - list of manifold strings, e.g. ``['3d', '3d']``
        * ``tm_dimensions`` - list of total orbital counts (``dim²×2`` per atom)
    """
    tm_atoms = tag_and_list_atoms(atoms, table=table)
    tm_manifolds = get_default_manifolds(tm_atoms)
    tm_dimensions = get_dimensions(tm_manifolds)
    return tm_atoms, tm_manifolds, tm_dimensions


def prepare_hubbard_structure(
    atoms,
    tm_atoms,
    tm_manifolds,
    U_values=5.0,
    neighbors=None,
    intersite_V_values=None,
):
    """Build a Hubbard-annotated AiiDA structure from an ASE ``Atoms`` object.

    Replaces the boilerplate that appears at the top of every submission
    script::

        structure = StructureData(ase=atoms)
        hubbard_structure = HubbardStructureData.from_structure(structure)
        for itm, tm_atom in enumerate(tm_atoms):
            hubbard_structure.initialize_onsites_hubbard(
                atom_name=tm_atom, atom_manifold=tm_manifolds[itm], value=Uval)
        hutils = HubbardUtils(hubbard_structure)
        hutils.reorder_atoms()
        hubbard_structure = hutils._hubbard_structure

    The Hubbard atoms are always reordered to sit before the ligands in the
    structure, as required by the Quantum ESPRESSO + HP workflow.

    Args:
        atoms: ASE :class:`~ase.Atoms` object (e.g. from ``ase.io.read``).
        tm_atoms: list of tagged TM species strings as returned by
            :func:`prepare_tm_info`, e.g. ``['Fe1', 'Fe2']``.
        tm_manifolds: list of manifold strings matching *tm_atoms*,
            e.g. ``['3d', '3d']``.
        U_values: Hubbard U value(s) in eV. Either:

            * a single ``float`` - the same U is applied to every TM site, or
            * a ``list`` of floats - one value per entry in *tm_atoms*
              (must have the same length).

        neighbors: Optional list of intersite neighbor specifications. Each
            entry may be a tuple ``(neighbour_name, neighbour_manifold)`` or a
            dict with keys ``'name'`` and ``'manifold'`` (e.g.
            ``[('O1', '2p'), ('O2', '2p')]`` or
            ``[{'name': 'O1', 'manifold': '2p'}]``). Every TM atom will get
            an intersite Hubbard V term with each listed neighbor. Defaults to
            ``None`` (no intersite terms).
        intersite_V_values: Hubbard V value(s) in eV for the intersite terms.
            Ignored when *neighbors* is ``None``. Either:

            * ``None`` - defaults to 1.0 for all neighbor pairs,
            * a single ``float`` - applied to every TM-neighbor pair, or
            * a ``list`` of floats - one value per entry in *neighbors*
              (must have the same length as *neighbors*).

    Returns:
        :class:`~aiida_quantumespresso.data.hubbard_structure.HubbardStructureData`
        with onsite (and optionally intersite) Hubbard parameters initialised
        and atoms reordered.

    Raises:
        ValueError: if *U_values* is a list whose length does not match
            *tm_atoms*, or if *intersite_V_values* is a list whose length
            does not match *neighbors*.
    """
    from aiida.orm import StructureData
    from aiida_quantumespresso.data.hubbard_structure import HubbardStructureData
    from aiida_quantumespresso.utils.hubbard import HubbardUtils

    # Normalise U_values to a per-atom list
    if isinstance(U_values, (int, float)):
        u_list = [float(U_values)] * len(tm_atoms)
    else:
        u_list = list(U_values)
        if len(u_list) != len(tm_atoms):
            raise ValueError(
                f"U_values has {len(u_list)} entries but tm_atoms has "
                f"{len(tm_atoms)} entries; lengths must match."
            )

    structure = StructureData(ase=atoms)
    hubbard_structure = HubbardStructureData.from_structure(structure)

    for tm_atom, manifold, u_val in zip(tm_atoms, tm_manifolds, u_list):
        hubbard_structure.initialize_onsites_hubbard(
            atom_name=tm_atom,
            atom_manifold=manifold,
            value=u_val,
        )

    # Optionally add intersite V terms
    if neighbors is not None:
        # Normalise each neighbor entry to a (name, manifold) tuple
        parsed_neighbors = []
        for nb in neighbors:
            if isinstance(nb, dict):
                parsed_neighbors.append((nb['name'], nb['manifold']))
            else:
                parsed_neighbors.append(tuple(nb))

        # Normalise intersite_V_values to a per-neighbor list
        if intersite_V_values is None:
            v_list = [1.0] * len(parsed_neighbors)
        elif isinstance(intersite_V_values, (int, float)):
            v_list = [float(intersite_V_values)] * len(parsed_neighbors)
        else:
            v_list = list(intersite_V_values)
            if len(v_list) != len(parsed_neighbors):
                raise ValueError(
                    f"intersite_V_values has {len(v_list)} entries but "
                    f"neighbors has {len(parsed_neighbors)} entries; "
                    f"lengths must match."
                )

        for tm_atom, tm_manifold in zip(tm_atoms, tm_manifolds):
            for (nb_name, nb_manifold), v_val in zip(parsed_neighbors, v_list):
                hubbard_structure.initialize_intersites_hubbard(
                    atom_name=tm_atom,
                    atom_manifold=tm_manifold,
                    neighbour_name=nb_name,
                    neighbour_manifold=nb_manifold,
                    value=v_val,
                )

    # Reorder so that Hubbard atoms precede ligands (required by QE/HP)
    hutils = HubbardUtils(hubbard_structure)
    hutils.reorder_atoms()
    return hutils._hubbard_structure