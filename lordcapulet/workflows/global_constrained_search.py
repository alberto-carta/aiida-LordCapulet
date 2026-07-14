from aiida.engine import WorkChain, ToContext, while_, if_, submit
from aiida.orm import Dict, List, Int, Float, Str, Bool, Code, KpointsData, StructureData, load_node
from aiida.plugins import CalculationFactory
from aiida_quantumespresso.data.hubbard_structure import HubbardStructureData

# Import the custom workchains
from lordcapulet.workflows.standard_magnetic_scan import StandardMagneticScanWorkChain
from lordcapulet.workflows.constrained_scan import ConstrainedScanWorkChain
from lordcapulet.functions import aiida_propose_occ_matrices_from_results

PROPOSAL_METADATA_KEYS = (
    'proposal_source',
    'proposal_mode',
    'proposal_generation',
)

STARTUP_MODES = ('from_scratch', 'seeded')


def validate_startup_mode(value, _port):
    """Validate the global-search startup mode."""
    mode = value.value
    if mode not in STARTUP_MODES:
        return f"startup_mode must be one of {STARTUP_MODES}, got {mode!r}"
    return None


class GlobalConstrainedSearchWorkChain(WorkChain):
    """
    Global WorkChain that orchestrates the entire constrained DFT+U search process.
    
    This workchain:
    1. Starts with an AFM search to get initial occupation matrices
    2. Iteratively runs constrained scans in batches of N proposals per generation
    3. After each batch, proposes new matrices using post-processing
    4. Continues until Nmax total proposals have been evaluated
    
    The workflow is:
    AFM Search -> Propose matrices -> Constrained Scan (N proposals) -> 
    Propose matrices -> Constrained Scan (N proposals) -> ... until Nmax reached
    """
    
    @classmethod
    def define(cls, spec):
        super().define(spec)
        
        # Expose inputs from StandardMagneticScanWorkChain
        spec.expose_inputs(StandardMagneticScanWorkChain, namespace='mag_scan')
        
        # Expose inputs from ConstrainedScanWorkChain (will be reused)
        spec.expose_inputs(ConstrainedScanWorkChain, namespace='constrained',
                          exclude=['occupation_matrices_list'])
        
        # Global search parameters
        spec.input('Nmax', valid_type=Int, 
                  help='Total maximum number of constrained DFT proposals to evaluate')
        spec.input('N', valid_type=Int,
                  help='Number of proposals per generation (batch size)')
        
        # Proposal function parameters
        spec.input('proposal_mode', valid_type=Str, default=lambda: Str('random'),
                  help='Mode for proposing new matrices: random, read, etc.')
        spec.input('proposal_debug', valid_type=Bool, default=lambda: Bool(False),
                  help='Enable debug output for proposal function')
        spec.input('proposal_holistic', valid_type=Bool, default=lambda: Bool(False),
                  help='Use holistic approach: analyze all previous result matrices, not just from last generation')
        spec.input('proposal_kwargs', valid_type=Dict, required=False,
                  help='Additional keyword arguments for proposal function')
        spec.input('startup_mode', valid_type=Str, default=lambda: Str('from_scratch'),
                  validator=validate_startup_mode,
                  help='Startup mode: from_scratch or seeded')
        spec.input('seed_global_workchain_pk', valid_type=Int, required=False,
                  help='PK of a previous GlobalConstrainedSearchWorkChain used for seeded startup')
        
        # Walltime control parameters (optional - will override sub-workchain defaults)
        spec.input('mag_scan_walltime_hours', valid_type=Float, required=False,
                  help='Walltime in hours for AFM calculations (overrides afm.walltime_hours if provided)')
        spec.input('constrained_walltime_hours', valid_type=Float, required=False,
                  help='Walltime in hours for constrained calculations (overrides constrained.walltime_hours if provided)')
        
        spec.outline(
            if_(cls.should_start_from_scratch)(
                cls.run_initial_mag_scan,
                cls.process_mag_scan_results,
            ),
            if_(cls.should_start_seeded)(
                cls.import_seed_results,
            ),
            while_(cls.should_continue_search)(
                cls.run_constrained_batch,
                cls.process_constrained_results,
                cls.update_counters,
            ),
            cls.gather_final_results,
        )
        
        # Outputs
        # spec.output('converged_mag_scan_matrix_pks', valid_type=List,
        #            help='Occupation matrices from initial AFM search')
        # spec.output('converged_constrained_matrix_pks', valid_type=List,
        #            help='All occupation matrices from constrained calculations')
        # spec.output('converged_mag_scan_calculation_pks', valid_type=List,
        #            help='PKs of all converged AFM calculations')
        # spec.output('converged_constrained_calculation_pks', valid_type=List,
        #            help='PKs of all converged constrained calculations')
        spec.output('converged_matrix_pks', valid_type=List,
                   help='All occupation matrices from AFM and constrained calculations')
        spec.output('converged_calculation_pks', valid_type=List,
                   help='PKs of all converged calculations (AFM + constrained)')
        spec.output('all_calculation_pks', valid_type=List,
                   help='PKs of all calculations performed')
        spec.output('generation_summary', valid_type=Dict,
                   help='Summary of results per generation')
        
        # Exit codes
        spec.exit_code(400, 'ERROR_MAG_SCAN_FAILED',
                      message='Initial magnetic scan failed')
        spec.exit_code(401, 'ERROR_CONSTRAINED_SCAN_FAILED',
                      message='Constrained scan failed')
        spec.exit_code(402, 'ERROR_PROPOSAL_FAILED',
                      message='Matrix proposal step failed')
        spec.exit_code(403, 'ERROR_SEED_IMPORT_FAILED',
                      message='Seed global workchain data could not be imported')

    @classmethod
    def get_builder_from_protocol(
        cls,
        code,
        structure,
        hubbard_corr_atoms,
        protocol: str = 'default',
        overrides: dict | None = None,
    ):
        """Return a pre-populated builder for the global constrained search.

        Protocol defaults are loaded from all three YAML files:
        ``common.yaml`` + ``standard_magnetic_scan.yaml`` for the magnetic scan sub-workchain,
        ``common.yaml`` + ``constrained_scan.yaml`` for the constrained scan,
        and ``global_search.yaml`` for global parameters.

        Args:
            code: AiiDA :class:`~aiida.orm.Code` (or label string).  Pass a
                dict ``{'mag_scan': code1, 'constrained': code2}`` to use
                different codes for each sub-workchain.
            structure: input structure.
            hubbard_corr_atoms: list of tagged Hubbard-corrected species strings returned by
                :func:`~lordcapulet.utils.preprocessing.submission.tag_and_list_atoms`.
            protocol: protocol name (default: ``'default'``).
            overrides: optional dict.  Keys ``'mag_scan'`` and ``'constrained'``
                route overrides to the respective sub-protocol; all other
                keys (``'Nmax'``, ``'N'``, ``'proposal_mode'``, …) override
                the global search parameters.

        Returns:
            A populated :class:`~aiida.engine.ProcessBuilder`.
        """
        import yaml
        from importlib_resources import files
        import lordcapulet.workflows.protocols as protocols_pkg
        from aiida.orm import load_code, KpointsData, Dict, List, Float, Str, Int, Bool
        from lordcapulet.utils.preprocessing.submission import (
            get_default_manifolds,
            get_dimensions,
        )
        from lordcapulet.workflows.protocols.utils import recursive_merge

        overrides = overrides or {}

        # ── resolve code(s) ─────────────────────────────────────────────────
        if isinstance(code, dict):
            afm_code = load_code(code['mag_scan']) if isinstance(code['mag_scan'], str) else code['mag_scan']
            con_code = load_code(code['constrained']) if isinstance(code['constrained'], str) else code['constrained']
        else:
            if isinstance(code, str):
                code = load_code(code)
            afm_code = con_code = code

        # ── load sub-protocol inputs ─────────────────────────────────────────
        afm_inputs = StandardMagneticScanWorkChain.get_protocol_inputs(
            protocol, overrides.get('mag_scan')
        )
        con_inputs = ConstrainedScanWorkChain.get_protocol_inputs(
            protocol, overrides.get('constrained')
        )

        # ── load global_search.yaml defaults ────────────────────────────────
        global_path = files(protocols_pkg) / 'global_search.yaml'
        with global_path.open() as fh:
            global_data = yaml.safe_load(fh) or {}

        proto_data = (global_data.get('protocols', {}).get(protocol) or {})
        global_inputs = {k: v for k, v in global_data.items()
                         if k not in {'protocols', 'default_protocol'}}
        global_inputs = recursive_merge(
            global_inputs,
            {k: v for k, v in proto_data.items() if k != 'description'},
        )
        # User top-level overrides (not 'mag_scan' / 'constrained' keys)
        user_global = {k: v for k, v in overrides.items()
                       if k not in ('mag_scan', 'constrained')}
        global_inputs = recursive_merge(global_inputs, user_global)

        # ── compute n_oscdft from hubbard_corr_atoms ───────────────────────────────────
        manifolds = get_default_manifolds(hubbard_corr_atoms)
        n_oscdft = sum(get_dimensions(manifolds))

        oscdft_dict = dict(con_inputs.get('oscdft_card', {}))
        oscdft_dict['n_oscdft'] = n_oscdft

        # ── build kpoints ────────────────────────────────────────────────────
        # make_kpoints uses density-based auto-mesh by default, or a fixed mesh
        # if the caller passed overrides={'mag_scan': {'kpoints_mesh': [...]}}.
        from lordcapulet.workflows.protocols.utils import make_kpoints
        afm_kpoints = make_kpoints(afm_inputs, structure)
        con_kpoints = make_kpoints(con_inputs, structure)

        # ── populate builder ─────────────────────────────────────────────────
        builder = cls.get_builder()

        # AFM sub-workchain
        builder.mag_scan.code = afm_code
        builder.mag_scan.structure = structure
        builder.mag_scan.kpoints = afm_kpoints
        builder.mag_scan.parameters = Dict(dict=afm_inputs['parameters'])
        builder.mag_scan.hubbard_corr_atoms = List(list=hubbard_corr_atoms)
        builder.mag_scan.magnitude = Float(afm_inputs.get('magnitude', 0.5))
        if 'max_configurations' in afm_inputs and afm_inputs['max_configurations'] is not None:
            builder.mag_scan.max_configurations = Int(afm_inputs['max_configurations'])
        builder.mag_scan.walltime_hours = Float(afm_inputs.get('walltime_hours', 2.0))
        builder.mag_scan.pseudo_family_string = Str(
            afm_inputs.get('pseudo_family', 'SSSP/1.3/PBEsol/efficiency')
        )

        # Constrained sub-workchain (occupation_matrices_list excluded via expose_inputs)
        builder.constrained.code = con_code
        builder.constrained.structure = structure
        builder.constrained.kpoints = con_kpoints
        builder.constrained.parameters = Dict(dict=con_inputs['parameters'])
        builder.constrained.hubbard_corr_atoms = List(list=hubbard_corr_atoms)
        builder.constrained.oscdft_card = Dict(dict=oscdft_dict)
        builder.constrained.walltime_hours = Float(con_inputs.get('walltime_hours', 2.0))
        builder.constrained.pseudo_family_string = Str(
            con_inputs.get('pseudo_family', 'SSSP/1.3/PBEsol/efficiency')
        )

        # Global search parameters
        builder.Nmax = Int(global_inputs.get('Nmax', 20))
        builder.N = Int(global_inputs.get('N', 4))
        builder.proposal_mode = Str(global_inputs.get('proposal_mode', 'random_so_n'))
        builder.proposal_debug = Bool(global_inputs.get('proposal_debug', False))
        builder.proposal_holistic = Bool(global_inputs.get('proposal_holistic', False))
        if 'proposal_kwargs' in global_inputs:
            builder.proposal_kwargs = Dict(dict=global_inputs['proposal_kwargs'])
        if 'startup_mode' in global_inputs:
            builder.startup_mode = Str(global_inputs['startup_mode'])
        if 'seed_global_workchain_pk' in global_inputs:
            builder.seed_global_workchain_pk = Int(global_inputs['seed_global_workchain_pk'])

        return builder

    @staticmethod
    def _metadata_from_proposal_node(proposal_node):
        """Extract provenance extras from a proposal-list node."""
        metadata = {}
        for key in PROPOSAL_METADATA_KEYS:
            try:
                if key in proposal_node.base.extras:
                    metadata[key] = proposal_node.base.extras.get(key)
            except AttributeError:
                continue
        return metadata

    @staticmethod
    def _set_proposal_extras(node, metadata):
        """Persist proposal provenance on a submitted child process node."""
        for key, value in metadata.items():
            node.base.extras.set(key, value)

    @staticmethod
    def _outputs_have(outputs, *keys):
        """Return True if an output namespace has all requested attributes."""
        return all(key in outputs for key in keys)

    @staticmethod
    def _parse_generation_key(key):
        """Return the integer generation index from an output-summary key."""
        if isinstance(key, int):
            return key
        text = str(key).strip()
        if text.lower().startswith('generation '):
            text = text.split()[-1]
        return int(text)

    @staticmethod
    def _is_constrained_process(node):
        process_type = getattr(node, 'process_type', '').lower()
        return 'constrainedscan' in process_type or 'constrained_scan' in process_type

    @staticmethod
    def _is_mag_scan_process(node):
        process_type = getattr(node, 'process_type', '').lower()
        return 'standardmagneticscan' in process_type or 'standard_magnetic_scan' in process_type

    @staticmethod
    def _get_extra(node, key):
        """Read an AiiDA extra, returning None when absent."""
        try:
            if key in node.base.extras:
                return node.base.extras.get(key)
        except AttributeError:
            return None
        return None

    def should_start_from_scratch(self):
        """Return True when this workchain should run a fresh magnetic scan."""
        return self.inputs.startup_mode.value == 'from_scratch'

    def should_start_seeded(self):
        """Return True when this workchain should import a previous global run."""
        return self.inputs.startup_mode.value == 'seeded'

    def _build_proposal_kwargs(self, current_generation):
        """Convert user proposal kwargs to AiiDA nodes and add generation index."""
        proposal_kwargs = {'current_generation': Int(current_generation)}

        if 'proposal_kwargs' not in self.inputs:
            return proposal_kwargs

        for key, value in self.inputs.proposal_kwargs.get_dict().items():
            if isinstance(value, bool):
                proposal_kwargs[key] = Bool(value)
            elif isinstance(value, str):
                proposal_kwargs[key] = Str(value)
            elif isinstance(value, (int, float)):
                proposal_kwargs[key] = Float(value) if isinstance(value, float) else Int(value)
            elif isinstance(value, list):
                proposal_kwargs[key] = List(list=value)
            elif isinstance(value, dict):
                proposal_kwargs[key] = Dict(dict=value)
            else:
                proposal_kwargs[key] = value

        return proposal_kwargs

    def _infer_imported_proposal_metadata(self, constrained_generation, entry):
        """Fill missing proposal provenance for imported constrained generations."""
        metadata = {
            'proposal_mode': entry.get('proposal_mode', self.inputs.proposal_mode.value),
            'proposal_generation': entry.get('proposal_generation', constrained_generation - 1),
            'constrained_generation': entry.get('constrained_generation', constrained_generation),
        }
        if 'proposal_source' in entry:
            metadata['proposal_source'] = entry['proposal_source']
            return metadata

        mode = metadata['proposal_mode']
        if mode in ('gp', 'gaussian_process'):
            metadata['proposal_source'] = 'random_warmup' if constrained_generation == 1 else 'gp'
        elif mode in ('random', 'random_so_n'):
            metadata['proposal_source'] = 'random'
        elif mode in ('template_product', 'primitive_template_product'):
            metadata['proposal_source'] = 'template_product'
        else:
            metadata['proposal_source'] = mode
        return metadata

    def _imported_generation_result(self, generation, entry):
        """Return a normalized imported generation-summary entry."""
        normalized = dict(entry)
        normalized['imported_seed'] = True
        if normalized.get('type') == 'constrained':
            normalized.update(self._infer_imported_proposal_metadata(generation, normalized))
        return normalized

    def _seed_data_from_parent_outputs(self, seed_wc):
        """Extract seed data from final outputs of a previous global workchain."""
        outputs = getattr(seed_wc, 'outputs', None)
        required = (
            'converged_matrix_pks',
            'converged_calculation_pks',
            'all_calculation_pks',
            'generation_summary',
        )
        if outputs is None or not self._outputs_have(outputs, *required):
            return None

        summary = outputs.generation_summary.get_dict()
        generation_results = {}
        for key, entry in summary.items():
            generation = self._parse_generation_key(key)
            generation_results[generation] = self._imported_generation_result(
                generation, entry
            )

        constrained_generations = [
            generation for generation, entry in generation_results.items()
            if entry.get('type') == 'constrained'
        ]
        if not constrained_generations:
            return None

        latest_generation = max(constrained_generations)
        return {
            'generation_results': generation_results,
            'generation': latest_generation,
            'imported_constrained_attempts': sum(
                entry.get('n_calculations', 0)
                for entry in generation_results.values()
                if entry.get('type') == 'constrained'
            ),
            'converged_matrix_pks': outputs.converged_matrix_pks.get_list().copy(),
            'converged_calculation_pks': outputs.converged_calculation_pks.get_list().copy(),
            'all_calculation_pks': outputs.all_calculation_pks.get_list().copy(),
        }

    def _seed_data_from_called_children(self, seed_wc):
        """Reconstruct seed data from called magnetic and constrained scan children."""
        generation_results = {}
        converged_matrix_pks = []
        converged_calculation_pks = []
        all_calculation_pks = []
        constrained_generation = 1

        for child in getattr(seed_wc, 'called', []):
            outputs = getattr(child, 'outputs', None)
            if outputs is None:
                continue

            if self._is_mag_scan_process(child):
                if not self._outputs_have(
                    outputs,
                    'converged_matrix_pks',
                    'converged_calculation_pks',
                    'all_calculation_pks',
                ):
                    continue
                matrices = outputs.converged_matrix_pks.get_list()
                converged_calcs = outputs.converged_calculation_pks.get_list()
                all_calcs = outputs.all_calculation_pks.get_list()
                generation_results[0] = {
                    'type': 'mag_scan',
                    'n_calculations': len(all_calcs),
                    'converged_matrix_pks': matrices,
                    'converged_calculation_pks': converged_calcs,
                    'imported_seed': True,
                }
                converged_matrix_pks.extend(matrices)
                converged_calculation_pks.extend(converged_calcs)
                all_calculation_pks.extend(all_calcs)

            elif self._is_constrained_process(child):
                if not self._outputs_have(
                    outputs,
                    'converged_matrix_pks',
                    'converged_calculation_pks',
                    'all_calculation_pks',
                ):
                    continue
                matrices = outputs.converged_matrix_pks.get_list()
                converged_calcs = outputs.converged_calculation_pks.get_list()
                all_calcs = outputs.all_calculation_pks.get_list()
                entry = {
                    'type': 'constrained',
                    'n_calculations': len(all_calcs),
                    'n_successful': len(matrices),
                    'n_failed': len(all_calcs) - len(matrices),
                    'matrix_pks': matrices,
                    'calculation_pks': all_calcs,
                    'converged_calculation_pks': converged_calcs,
                }
                for key in PROPOSAL_METADATA_KEYS + ('constrained_generation', 'n_proposals_submitted'):
                    value = self._get_extra(child, key)
                    if value is not None:
                        entry[key] = value
                generation_results[constrained_generation] = (
                    self._imported_generation_result(constrained_generation, entry)
                )
                converged_matrix_pks.extend(matrices)
                converged_calculation_pks.extend(converged_calcs)
                all_calculation_pks.extend(all_calcs)
                constrained_generation += 1

        constrained_generations = [
            generation for generation, entry in generation_results.items()
            if entry.get('type') == 'constrained'
        ]
        if not constrained_generations:
            return None

        return {
            'generation_results': generation_results,
            'generation': max(constrained_generations),
            'imported_constrained_attempts': sum(
                entry.get('n_calculations', 0)
                for entry in generation_results.values()
                if entry.get('type') == 'constrained'
            ),
            'converged_matrix_pks': converged_matrix_pks,
            'converged_calculation_pks': converged_calculation_pks,
            'all_calculation_pks': all_calculation_pks,
        }

    def _seed_training_pks_for_initial_gp(self):
        """Return imported mag-scan and random-warmup PKs for seeded GP startup."""
        matrix_pks = []
        calculation_pks = []
        for generation in sorted(self.ctx.generation_results):
            entry = self.ctx.generation_results[generation]
            if entry.get('type') == 'mag_scan':
                matrix_pks.extend(entry.get('converged_matrix_pks', []))
                calculation_pks.extend(entry.get('converged_calculation_pks', []))
            elif (
                entry.get('type') == 'constrained'
                and entry.get('proposal_source') == 'random_warmup'
            ):
                matrix_pks.extend(entry.get('matrix_pks', []))
                calculation_pks.extend(entry.get('converged_calculation_pks', []))
        return matrix_pks, calculation_pks

    def _propose_after_seed_import(self):
        """Generate the first proposal batch after importing seed data."""
        if self.ctx.N_cumulative >= self.inputs.Nmax.value:
            self.ctx.current_proposals = []
            self.ctx.current_proposal_metadata = {}
            return None

        matrix_pks, calculation_pks = self._seed_training_pks_for_initial_gp()
        if not matrix_pks:
            self.report("Seed workchain has no mag-scan or random-warmup matrices for proposal")
            return self.exit_codes.ERROR_SEED_IMPORT_FAILED

        matrices_for_proposal = List(list=matrix_pks)
        calculations_for_proposal = List(list=calculation_pks)
        self.report(
            f"Using seeded proposal from {len(matrix_pks)} imported mag-scan/random-warmup matrices"
        )

        proposed_matrices_pks = aiida_propose_occ_matrices_from_results(
            occ_matr_pks=matrices_for_proposal,
            calc_pks=calculations_for_proposal,
            N=self.inputs.N,
            debug=self.inputs.proposal_debug,
            mode=self.inputs.proposal_mode,
            hubbard_corr_atoms=self.inputs.constrained.hubbard_corr_atoms,
            **self._build_proposal_kwargs(self.ctx.generation)
        )
        self.ctx.current_proposals = proposed_matrices_pks.get_list()
        self.ctx.current_proposal_metadata = self._metadata_from_proposal_node(
            proposed_matrices_pks
        )
        return None

    def import_seed_results(self):
        """Import previous global-search results for seeded startup."""
        if 'seed_global_workchain_pk' not in self.inputs:
            self.report("seed_global_workchain_pk is required for seeded startup")
            return self.exit_codes.ERROR_SEED_IMPORT_FAILED

        seed_wc = load_node(self.inputs.seed_global_workchain_pk.value)
        seed_data = (
            self._seed_data_from_parent_outputs(seed_wc)
            or self._seed_data_from_called_children(seed_wc)
        )
        if seed_data is None:
            self.report("Could not recover usable seed data from the seed workchain")
            return self.exit_codes.ERROR_SEED_IMPORT_FAILED

        self.ctx.imported_constrained_attempts = seed_data['imported_constrained_attempts']
        self.ctx.N_cumulative = 0
        self.ctx.generation = seed_data['generation']
        self.ctx.generation_results = seed_data['generation_results']
        self.ctx.converged_matrix_pks = seed_data['converged_matrix_pks']
        self.ctx.converged_calculation_pks = seed_data['converged_calculation_pks']
        self.ctx.all_calculation_pks = seed_data['all_calculation_pks']

        self.report(
            f"Imported seed workchain {self.inputs.seed_global_workchain_pk.value}: "
            f"{self.ctx.imported_constrained_attempts} constrained attempts; "
            f"new-run budget is {self.inputs.Nmax.value}"
        )
        return self._propose_after_seed_import()

    def run_initial_mag_scan(self):
        """
        Run the initial AFM search to get starting occupation matrices.
        """
        self.report("Starting initial magnetic scan")
        
        # Submit AFM scan with exposed inputs
        mag_scan_builder = StandardMagneticScanWorkChain.get_builder()
        mag_scan_builder.update(self.inputs.mag_scan)
        
        # Override walltime if provided at global level
        if 'mag_scan_walltime_hours' in self.inputs:
            mag_scan_builder.walltime_hours = self.inputs.mag_scan_walltime_hours
            self.report(f"Using global mag_scan walltime: {self.inputs.mag_scan_walltime_hours.value} hours")
        
        future = self.submit(mag_scan_builder)
        return ToContext(mag_scan_wc=future)

    def process_mag_scan_results(self):
        """
        Process AFM results and propose initial matrices for constrained calculations.
        """
        if not self.ctx.mag_scan_wc.is_finished_ok:
            self.report(f"Magnetic scan workchain failed with exit status: {self.ctx.mag_scan_wc.exit_status}")
            return self.exit_codes.ERROR_MAG_SCAN_FAILED
            
        # Check if we have any occupation matrices at all
        if 'converged_matrix_pks' not in self.ctx.mag_scan_wc.outputs:
            self.report("Magnetic scan workchain completed but no converged occupation matrices found")
            return self.exit_codes.ERROR_MAG_SCAN_FAILED
            
        self.report("Magnetic scan completed successfully, processing results")
        
        # Get AFM occupation matrices
        mag_scan_matrices = self.ctx.mag_scan_wc.outputs.converged_matrix_pks
        mag_scan_calculation_pks = self.ctx.mag_scan_wc.outputs.converged_calculation_pks
        self.ctx.converged_mag_scan_matrices = mag_scan_matrices


        self.ctx.converged_calculation_pks = mag_scan_calculation_pks.get_list().copy()
        
        # Check if we have any successful AFM results
        if len(mag_scan_matrices.get_list()) == 0:
            self.report("No successful magnetic scan calculations found")
            return self.exit_codes.ERROR_MAG_SCAN_FAILED
        
        self.report(f"Found {len(mag_scan_matrices.get_list())} successful magnetic scan occupation matrices")
        
        # Initialize counters and storage
        self.ctx.N_cumulative = 0
        self.ctx.generation = 0
        # self.ctx.all_matrix_pks = mag_scan_matrices.get_list().copy()
        self.ctx.converged_matrix_pks = mag_scan_matrices.get_list().copy()  # Only successful result matrices
        self.ctx.all_calculation_pks = self.ctx.mag_scan_wc.outputs.all_calculation_pks.get_list().copy()
        self.ctx.generation_results = {}
        
        # Store AFM results
        self.ctx.generation_results[0] = {
            'type': 'mag_scan',
            'n_calculations': len(mag_scan_matrices.get_list()),
            'converged_matrix_pks': mag_scan_matrices.get_list(),
            'converged_calculation_pks': mag_scan_calculation_pks.get_list()
        }
        
        # For initial proposal, use AFM results (holistic mode doesn't apply here)
        proposed_matrices_pks = aiida_propose_occ_matrices_from_results(
            occ_matr_pks=mag_scan_matrices,
            calc_pks=mag_scan_calculation_pks,
            N=self.inputs.N,
            debug=self.inputs.proposal_debug,
            mode=self.inputs.proposal_mode,
            hubbard_corr_atoms=self.inputs.mag_scan.hubbard_corr_atoms,
            **self._build_proposal_kwargs(self.ctx.generation)
        )
        
        # Store PKs of the proposal nodes for the next constrained scan
        self.ctx.current_proposals = proposed_matrices_pks.get_list()
        self.ctx.current_proposal_metadata = self._metadata_from_proposal_node(
            proposed_matrices_pks
        )

    def should_continue_search(self):
        """
        Check if we should continue the iterative search.
        """
        return self.ctx.N_cumulative < self.inputs.Nmax.value

    def run_constrained_batch(self):
        """
        Run a batch of constrained calculations with the current proposed matrices.
        """
        self.ctx.generation += 1
        remaining_budget = self.inputs.Nmax.value - self.ctx.N_cumulative
        proposal_metadata = dict(getattr(self.ctx, 'current_proposal_metadata', {}))
        requested_batch_size = (
            len(self.ctx.current_proposals)
            if proposal_metadata.get('proposal_source') == 'random_warmup'
            else self.inputs.N.value
        )
        n_proposals = min(
            requested_batch_size,
            remaining_budget,
            len(self.ctx.current_proposals),
        )
        
        self.report(f"Starting generation {self.ctx.generation} with {n_proposals} proposals")
        
        current_proposals = self.ctx.current_proposals[:n_proposals]
        proposal_metadata['constrained_generation'] = self.ctx.generation
        proposal_metadata['n_proposals_submitted'] = len(current_proposals)
        self.ctx.current_batch_proposal_metadata = proposal_metadata
        
        # Build constrained scan
        constrained_builder = ConstrainedScanWorkChain.get_builder()
        constrained_builder.update(self.inputs.constrained)
        constrained_builder.occupation_matrices_list = List(list=current_proposals)
        
        # Override walltime if provided at global level
        if 'constrained_walltime_hours' in self.inputs:
            constrained_builder.walltime_hours = self.inputs.constrained_walltime_hours
            self.report(f"Using global constrained walltime: {self.inputs.constrained_walltime_hours.value} hours")
        
        future = self.submit(constrained_builder)
        self._set_proposal_extras(future, proposal_metadata)
        return ToContext(constrained_wc=future)

    def process_constrained_results(self):
        """
        Process results from constrained scan and propose new matrices if needed.
        """
        if not self.ctx.constrained_wc.is_finished_ok:
            self.report("Constrained scan workchain failed, but checking individual calculations")
        
        # Get results
        constrained_matrices = self.ctx.constrained_wc.outputs.converged_matrix_pks
        calculation_pks = self.ctx.constrained_wc.outputs.all_calculation_pks
        converged_calculations = self.ctx.constrained_wc.outputs.converged_calculation_pks
        
        # Count successful calculations using converged_calculations list
        current_converged_matrices = constrained_matrices.get_list()
        n_successful = len(current_converged_matrices)
        n_total = len(calculation_pks.get_list())
        failed_count = n_total - n_successful
        
        if n_successful == 0:
            self.report(f"All {n_total} calculations in generation {self.ctx.generation} failed")
            return self.exit_codes.ERROR_CONSTRAINED_SCAN_FAILED
        elif failed_count > 0:
            self.report(f"Generation {self.ctx.generation}: {n_successful} successful, {failed_count} failed calculations")
        
        # Store results
        generation_result = {
            'type': 'constrained',
            'n_calculations': n_total,
            'n_successful': n_successful,
            'n_failed': failed_count,
            'matrix_pks': constrained_matrices.get_list(),
            'calculation_pks': calculation_pks.get_list(),
            'converged_calculation_pks': converged_calculations.get_list()
        }
        generation_result.update(
            getattr(self.ctx, 'current_batch_proposal_metadata', {})
        )
        self.ctx.generation_results[self.ctx.generation] = generation_result
        
        # Update cumulative storage
        # self.ctx.all_matrix_pks.extend(constrained_matrices.get_list()) # this one is probably redundant

        self.ctx.converged_calculation_pks.extend(converged_calculations.get_list())   
        self.ctx.converged_matrix_pks.extend(current_converged_matrices)  # Only successful results
        self.ctx.all_calculation_pks.extend(calculation_pks.get_list())
        
        self.report(f"Generation {self.ctx.generation} completed: "
                   f"{n_successful}/{n_total} successful calculations")
        
        # If we haven't reached Nmax, propose new matrices for next iteration
        if self.ctx.N_cumulative + len(calculation_pks.get_list()) < self.inputs.Nmax.value:
            # Choose which matrices to use for proposal based on holistic mode
            if self.inputs.proposal_holistic.value:
                # Use all successful result matrices from all generations
                matrices_for_proposal = List(list=self.ctx.converged_matrix_pks)
                calculations_for_proposal = List(list=self.ctx.converged_calculation_pks)
                self.report(f"Using holistic approach: analyzing {len(self.ctx.converged_matrix_pks)} total result matrices")
            else:
                # Use only successful matrices from current generation (Markovian)
                matrices_for_proposal = List(list=current_converged_matrices)
                calculations_for_proposal = converged_calculations
                self.report(f"Using Markovian approach: analyzing {n_successful} matrices from current generation")
            
            proposed_matrices_pks = aiida_propose_occ_matrices_from_results(
                occ_matr_pks=matrices_for_proposal,
                calc_pks=calculations_for_proposal,
                N=self.inputs.N,
                debug=self.inputs.proposal_debug,
                mode=self.inputs.proposal_mode,
                hubbard_corr_atoms=self.inputs.constrained.hubbard_corr_atoms,
                **self._build_proposal_kwargs(self.ctx.generation)
            )
            
            # Store PKs of the proposal nodes for the next constrained scan
            self.ctx.current_proposals = proposed_matrices_pks.get_list()
            self.ctx.current_proposal_metadata = self._metadata_from_proposal_node(
                proposed_matrices_pks
            )

    def update_counters(self):
        """
        Update the cumulative counter.
        """
        last_generation = self.ctx.generation_results[self.ctx.generation]
        self.ctx.N_cumulative += last_generation['n_calculations']
        
        self.report(f"Cumulative calculations: {self.ctx.N_cumulative}/{self.inputs.Nmax.value}")


    def gather_final_results(self):

        """
        Gather and output final results.
        """
        self.report(f"Global search completed. Total calculations: {self.ctx.N_cumulative}")
        
        # The class is gathering data as it is being proced, in this final step put everything in
        # the database with the store() method.


        # the workflow cannot create aiida types
        # here you need to make sure that the instances you pass
        # to out are already stored as aiida nodes, otherwise this will return a Data exception


        converged_matrix_pks = List(list=self.ctx.converged_matrix_pks)
        converged_matrix_pks.store()
        self.out('converged_matrix_pks', converged_matrix_pks)
        
        converged_calculation_pks = List(list=self.ctx.converged_calculation_pks)
        converged_calculation_pks.store()
        self.out('converged_calculation_pks', converged_calculation_pks)

        all_calculation_pks = List(list=self.ctx.all_calculation_pks)
        all_calculation_pks.store()
        self.out('all_calculation_pks', all_calculation_pks)

        generation_results_str_keys = Dict(dict={f"Generation {k}": v for k, v in self.ctx.generation_results.items()})
        generation_results_str_keys.store()
        self.out('generation_summary', generation_results_str_keys)
