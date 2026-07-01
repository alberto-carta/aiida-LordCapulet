from aiida.engine import WorkChain, ToContext, submit, append_
from aiida.orm import load_group, List, Dict, Code, KpointsData, StructureData, Float, Str, Int, load_node, JsonableData
from aiida.plugins import CalculationFactory
# import UpfData
from aiida.orm import UpfData
import numpy as np
from aiida_quantumespresso.data.hubbard_structure import HubbardStructureData
from lordcapulet.utils import extract_occupations_from_calc
from lordcapulet.workflows.protocols.utils import ProtocolMixin
# load group

PwCalculation = CalculationFactory('quantumespresso.pw')

class StandardMagneticScanWorkChain(ProtocolMixin, WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)
        # Accept both StructureData and HubbardStructureData
        spec.input('structure', valid_type=(StructureData, HubbardStructureData))
        spec.input('parameters', valid_type=Dict)
        spec.input('kpoints', valid_type=KpointsData)
        # spec.input('pseudos', valid_type=Dict)
        spec.input('code', valid_type=Code)
        spec.input('hubbard_corr_atoms', valid_type=List)
        spec.input('magnitude', valid_type=Float, default=Float(0.5))
        spec.input('max_configurations', valid_type=Int, required=False,
                  help='Optional cap on the number of magnetic configurations to submit')
        spec.input('walltime_hours', valid_type=Float, default=lambda: Float(1.0),
                  help='Walltime in hours for each magnetic configuration calculation (default: 1 hour)')
        spec.input('pseudo_family_string', valid_type=Str,
                  default=lambda: Str('SSSP/1.3/PBEsol/efficiency'),
                  help='Pseudo-potential family to use (must be installed via aiida-pseudo)')
        spec.outline(
            cls.prepare_configs,
            cls.run_all,
            cls.gather_results,
        )
        spec.output('converged_matrix_pks', valid_type=List)
        spec.output('converged_calculation_pks', valid_type=List)
        spec.output('all_calculation_pks', valid_type=List)

    @classmethod
    def get_protocol_filepath(cls):
        """Return the path to the standard magnetic scan protocol YAML file."""
        from importlib_resources import files
        import lordcapulet.workflows.protocols as protocols_pkg
        return files(protocols_pkg) / 'standard_magnetic_scan.yaml'

    @classmethod
    def get_builder_from_protocol(
        cls,
        code,
        structure,
        hubbard_corr_atoms,
        protocol: str = 'default',
        overrides: dict | None = None,
        options: dict | None = None,
    ):
        """Return a pre-populated builder for the given protocol.

        Args:
            code: AiiDA :class:`~aiida.orm.Code` (or label string) to use.
            structure: :class:`~aiida.orm.StructureData` or
                :class:`~aiida_quantumespresso.data.hubbard_structure.HubbardStructureData`.
            hubbard_corr_atoms: list of tagged Hubbard-correlated species strings as
                returned by
                :func:`~lordcapulet.utils.preprocessing.submission.tag_and_list_atoms`.
            protocol: protocol name (default: ``'default'``).
            overrides: optional dict of inputs to override protocol defaults;
                deep-merged with highest priority.
            options: optional dict merged into ``metadata_options``.

        Returns:
            A populated :class:`~aiida.engine.ProcessBuilder`.
        """
        from aiida.orm import load_code
        from lordcapulet.workflows.protocols.utils import recursive_merge, make_kpoints

        if isinstance(code, str):
            code = load_code(code)

        inputs = cls.get_protocol_inputs(protocol, overrides)

        if options:
            inputs['metadata_options'] = recursive_merge(
                inputs.get('metadata_options', {}), options
            )

        kpoints = make_kpoints(inputs, structure)

        builder = cls.get_builder()
        builder.code = code
        builder.structure = structure
        builder.kpoints = kpoints
        builder.parameters = Dict(dict=inputs['parameters'])
        builder.hubbard_corr_atoms = List(list=hubbard_corr_atoms)
        builder.magnitude = Float(inputs.get('magnitude', 0.5))
        if 'max_configurations' in inputs and inputs['max_configurations'] is not None:
            builder.max_configurations = Int(inputs['max_configurations'])
        builder.walltime_hours = Float(inputs.get('walltime_hours', 2.0))
        builder.pseudo_family_string = Str(
            inputs.get('pseudo_family', 'SSSP/1.3/PBEsol/efficiency')
        )
        return builder

    def prepare_configs(self):
        hubbard_corr_atoms = self.inputs.hubbard_corr_atoms.get_list()
        N = len(hubbard_corr_atoms)
        self.ctx.magnetic_configs = []
        total_configurations = 2 ** N
        config_indices = range(total_configurations)

        if 'max_configurations' in self.inputs:
            max_configurations = self.inputs.max_configurations.value
            if max_configurations < 1:
                raise ValueError('max_configurations must be a positive integer')
            if max_configurations < total_configurations:
                config_indices = np.linspace(
                    0, total_configurations - 1, max_configurations, dtype=int
                ).tolist()
                self.report(
                    f"Limiting magnetic scan to {max_configurations}/"
                    f"{total_configurations} configurations"
                )

        for i in config_indices:
            config = {}
            binary_string = format(i, f'0{N}b')
            for j in range(N):
                config[hubbard_corr_atoms[j]] = self.inputs.magnitude * (1 if binary_string[j] == '1' else -1)
            self.ctx.magnetic_configs.append(config)
        self.ctx.results = []

    def run_all(self):
        self.ctx.calc_futures = []
        for starting_magnetization in self.ctx.magnetic_configs:
            builder = PwCalculation.get_builder()
            builder.code = self.inputs.code
            builder.structure = self.inputs.structure
            builder.parameters = self.inputs.parameters.clone()
            builder.kpoints = self.inputs.kpoints

            pseudo_family = load_group(self.inputs.pseudo_family_string.value)
            builder.pseudos = pseudo_family.get_pseudos(structure=builder.structure)

            builder.parameters['SYSTEM']['starting_magnetization'] = starting_magnetization
            
            # Set metadata for calculations with configurable walltime
            walltime_hours = self.inputs.walltime_hours.value
            walltime_str = f"{int(walltime_hours):02d}:{int((walltime_hours % 1) * 60):02d}:00"
            builder.metadata = {
                'options': {
                    'resources': {'num_machines': 1}, 
                    'withmpi': True,
                    'max_wallclock_seconds': int(walltime_hours * 3600)
                }
            }
            
            # <<< CORRECT KEY FOR OCCUPATION MATRICES >>>
            builder.settings = Dict(dict={'parser_options': {'parse_atomic_occupations': True}})
            # self.ctx.calc_futures.append(self.submit(builder))
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
        self.report(f"Magnetic scan completed. {len(converged_calculation_pks)}/{len(calculation_pks)} calculations converged, {successful_extractions}/{len(calculation_pks)} occupation matrices extracted")
