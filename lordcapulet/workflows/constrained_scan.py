from aiida.engine import WorkChain, ToContext, submit, append_
from aiida.orm import load_group, List, Dict, Code, KpointsData, StructureData, Float, Str, load_node, JsonableData
from aiida.plugins import CalculationFactory
import numpy as np
from aiida_quantumespresso.data.hubbard_structure import HubbardStructureData
from lordcapulet.utils import extract_occupations_from_calc
from lordcapulet.workflows.protocols.utils import ProtocolMixin

# Import the custom constrained calculation
from lordcapulet.calculations.constrained_pw import ConstrainedPWCalculation

class ConstrainedScanWorkChain(ProtocolMixin, WorkChain):
    """
    WorkChain that launches N ConstrainedPWCalculation with different target occupation matrices.
    
    This workchain takes a list of occupation matrices and runs a constrained DFT+U calculation
    for each one, gathering the results at the end.
    """
    
    @classmethod
    def define(cls, spec):
        super().define(spec)
        
        # Standard inputs for PW calculations
        spec.input('structure', valid_type=(StructureData, HubbardStructureData))
        spec.input('parameters', valid_type=Dict)
        spec.input('kpoints', valid_type=KpointsData)
        spec.input('code', valid_type=Code)
        spec.input('tm_atoms', valid_type=List)
        
        # OSCDFT specific inputs
        spec.input('oscdft_card', valid_type=Dict, help='OSCDFT parameters')
        spec.input('occupation_matrices_list', valid_type=List, 
                  help='List of target occupation matrices [iproposal][iatom][ispin][iorb][iorb]')
        spec.input('walltime_hours', valid_type=Float, default=lambda: Float(1.0),
                  help='Walltime in hours for each constrained calculation (default: 1 hour)')
        spec.input('pseudo_family_string', valid_type=Str,
                  default=lambda: Str('SSSP/1.3/PBEsol/efficiency'),
                  help='Pseudo-potential family to use (must be installed via aiida-pseudo)')
        
        spec.outline(
            cls.prepare_calculations,
            cls.run_all,
            cls.gather_results,
        )
        
        # Outputs
        spec.output('converged_matrix_pks', valid_type=List)
        spec.output('converged_calculation_pks', valid_type=List)
        spec.output('all_calculation_pks', valid_type=List)

    @classmethod
    def get_protocol_filepath(cls):
        """Return the path to the constrained scan protocol YAML file."""
        from importlib_resources import files
        import lordcapulet.workflows.protocols as protocols_pkg
        return files(protocols_pkg) / 'constrained_scan.yaml'

    @classmethod
    def get_builder_from_protocol(
        cls,
        code,
        structure,
        tm_atoms,
        occupation_matrices_list,
        protocol: str = 'default',
        overrides: dict | None = None,
        options: dict | None = None,
    ):
        """Return a pre-populated builder for the given protocol.

        Args:
            code: AiiDA :class:`~aiida.orm.Code` (or label string) to use.
            structure: :class:`~aiida.orm.StructureData` or
                :class:`~aiida_quantumespresso.data.hubbard_structure.HubbardStructureData`.
            tm_atoms: list of tagged TM species strings.
            occupation_matrices_list: list of occupation-matrix node PKs to use
                as constrained targets.
            protocol: protocol name (default: ``'default'``).
            overrides: optional dict of inputs to override protocol defaults.
            options: optional dict merged into ``metadata_options``.

        Returns:
            A populated :class:`~aiida.engine.ProcessBuilder`.
        """
        from aiida.orm import load_code
        from lordcapulet.workflows.protocols.utils import recursive_merge, make_kpoints
        from lordcapulet.utils.preprocessing.submission import (
            get_default_manifolds,
            get_dimensions,
        )

        if isinstance(code, str):
            code = load_code(code)

        inputs = cls.get_protocol_inputs(protocol, overrides)

        if options:
            inputs['metadata_options'] = recursive_merge(
                inputs.get('metadata_options', {}), options
            )

        # Compute n_oscdft from the TM atoms (dim*dim*2 per atom, matching get_dimensions)
        manifolds = get_default_manifolds(tm_atoms)
        n_oscdft = sum(get_dimensions(manifolds))

        oscdft_dict = dict(inputs.get('oscdft_card', {}))
        oscdft_dict['n_oscdft'] = n_oscdft

        kpoints = make_kpoints(inputs, structure)

        builder = cls.get_builder()
        builder.code = code
        builder.structure = structure
        builder.kpoints = kpoints
        builder.parameters = Dict(dict=inputs['parameters'])
        builder.tm_atoms = List(list=tm_atoms)
        builder.occupation_matrices_list = List(list=occupation_matrices_list)
        builder.oscdft_card = Dict(dict=oscdft_dict)
        builder.walltime_hours = Float(inputs.get('walltime_hours', 2.0))
        builder.pseudo_family_string = Str(
            inputs.get('pseudo_family', 'SSSP/1.3/PBEsol/efficiency')
        )
        return builder

    def prepare_calculations(self):
        """
        Prepare the list of calculations with different target occupation matrices.
        """
        # Get the list of occupation matrices
        occupation_matrices_list = self.inputs.occupation_matrices_list.get_list()
        
        self.ctx.n_calculations = len(occupation_matrices_list)
        self.ctx.target_matrices = occupation_matrices_list
        
        self.report(f"Preparing {self.ctx.n_calculations} constrained calculations")

    def run_all(self):
        """
        Submit all the constrained calculations with different target occupation matrices.
        """
        self.ctx.calc_futures = []
        
        for i, target_matrix_pk in enumerate(self.ctx.target_matrices):
            self.report(f"Submitting calculation {i+1}/{self.ctx.n_calculations}")
            
            # Load the target matrix node from PK
            target_matrix_node = load_node(target_matrix_pk)
            
            # Build the calculation
            builder = ConstrainedPWCalculation.get_builder()
            builder.code = self.inputs.code
            builder.structure = self.inputs.structure
            builder.parameters = self.inputs.parameters.clone()
            builder.kpoints = self.inputs.kpoints
            pseudo_family = load_group(self.inputs.pseudo_family_string.value)
            builder.pseudos = pseudo_family.get_pseudos(structure=builder.structure)
            
            # Set magnetization for all transition metal atoms to a small value
            tm_atoms = self.inputs.tm_atoms.get_list()
            magnetization_config = {}
            for tm_atom in tm_atoms:
                magnetization_config[tm_atom] = 1e-9
            
            # Add the magnetization to the parameters
            builder.parameters['SYSTEM']['starting_magnetization'] = magnetization_config

            # Set OSCDFT specific inputs
            builder.oscdft_card = self.inputs.oscdft_card

            # Pass the loaded target matrix node (JsonableData or Dict)
            builder.target_matrix = target_matrix_node
            
            # Set computational options with configurable walltime
            walltime_hours = self.inputs.walltime_hours.value
            walltime_str = f"{int(walltime_hours):02d}:{int((walltime_hours % 1) * 60):02d}:00"
            builder.metadata = {
                'options': {
                    'resources': {'num_machines': 1}, 
                    'withmpi': True,
                    'max_wallclock_seconds': int(walltime_hours * 3600)
                }
            }

            
            # Enable parsing of occupation matrices and add oscdft flag
            builder.settings = Dict(dict={
                'parser_options': {'parse_atomic_occupations': True},
                'CMDLINE': ['-oscdft'],
            })
            
            # Submit and store in context
            self.to_context(calcs=append_(self.submit(builder)))

    def gather_results(self):
        """
        Collect the PKs and occupation matrices from all calculations.
        """
        converged_calculation_pks = []
        calculation_pks = []
        occupation_matrices_pks = []
        
        self.report(f"DEBUG: gather_results called with {len(self.ctx.calcs)} calculations")
        
        for i, calc in enumerate(self.ctx.calcs):
            calculation_pks.append(calc.pk)
            
            # Reload the calculation node to get fresh state
            fresh_calc = load_node(calc.pk)
            
            # Debug information about calculation state (both cached and fresh)
            self.report(f"DEBUG: Calc {i+1} (PK {calc.pk}):")
            self.report(f"  Cached: is_finished={calc.is_finished}, exit_status={calc.exit_status}, exit_code={calc.exit_code}")
            self.report(f"  Fresh:  is_finished={fresh_calc.is_finished}, exit_status={fresh_calc.exit_status}, exit_code={fresh_calc.exit_code}")

            # Use the fresh node for checking status (use exit_status which is always available)
            if fresh_calc.is_finished and fresh_calc.exit_status == 0:
                converged_calculation_pks.append(calc.pk)
                self.report(f"Calculation {i+1} completed successfully, PK: {calc.pk}")
                
                # Extract and store occupation matrix using unified structure
                try:
                    occupation_data = extract_occupations_from_calc(fresh_calc)
                    # Store as AiiDA JsonableData node directly
                    occ_node = JsonableData(occupation_data)
                    occ_node.store()
                    occupation_matrices_pks.append(occ_node.pk)
                    
                    # Add occupation matrix pk to the extras of each calculation
                    fresh_calc.base.extras.set('occupation_matrix_pk', occ_node.pk)
                    
                    self.report(f"Occupation matrix extracted and stored with PK: {occ_node.pk}")
                except Exception as e:
                    self.report(f"Failed to extract occupation matrix from calculation {calc.pk}: {e}")
            elif fresh_calc.is_finished:
                self.report(f"Calculation {i+1} finished but failed, PK: {calc.pk}, exit status: {fresh_calc.exit_status}")
            else:
                self.report(f"Calculation {i+1} not yet finished, PK: {calc.pk}")
        
        # Store outputs
        self.out('converged_calculation_pks', List(list=converged_calculation_pks).store())
        self.out('all_calculation_pks', List(list=calculation_pks).store())
        self.out('converged_matrix_pks', List(list=occupation_matrices_pks).store())
        
        successful_extractions = len([pk for pk in occupation_matrices_pks if pk != -1])
        self.report(f"Constrained scan completed. {len(converged_calculation_pks)}/{len(calculation_pks)} calculations converged, {successful_extractions}/{len(calculation_pks)} occupation matrices extracted")