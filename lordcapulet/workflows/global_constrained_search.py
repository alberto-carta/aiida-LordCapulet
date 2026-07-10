from aiida.engine import WorkChain, ToContext, while_, if_, submit
from aiida.orm import Dict, List, Int, Float, Str, Bool, Code, KpointsData, StructureData, load_node
from aiida.plugins import CalculationFactory
from aiida_quantumespresso.data.hubbard_structure import HubbardStructureData

# Import the custom workchains
from lordcapulet.workflows.standard_magnetic_scan import StandardMagneticScanWorkChain
from lordcapulet.workflows.constrained_scan import ConstrainedScanWorkChain
from lordcapulet.functions import aiida_propose_occ_matrices_from_results

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
        
        # Walltime control parameters (optional - will override sub-workchain defaults)
        spec.input('mag_scan_walltime_hours', valid_type=Float, required=False,
                  help='Walltime in hours for AFM calculations (overrides afm.walltime_hours if provided)')
        spec.input('constrained_walltime_hours', valid_type=Float, required=False,
                  help='Walltime in hours for constrained calculations (overrides constrained.walltime_hours if provided)')
        
        spec.outline(
            cls.run_initial_mag_scan,
            cls.process_mag_scan_results,
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
            hubbard_corr_atoms: list of tagged hubbard species strings returned by
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

        return builder

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
        
        # Propose initial matrices for first constrained batch
        proposal_kwargs = {}

        # always include generation index (required by GP mode)
        proposal_kwargs['current_generation'] = Int(self.ctx.generation)

        if 'proposal_kwargs' in self.inputs:
            # Convert proposal_kwargs to AiiDA types if needed
            for key, value in self.inputs.proposal_kwargs.get_dict().items():
                if isinstance(value, str):
                    proposal_kwargs[key] = Str(value)
                elif isinstance(value, (int, float)):
                    proposal_kwargs[key] = Float(value) if isinstance(value, float) else Int(value)
                elif isinstance(value, bool):
                    proposal_kwargs[key] = Bool(value)
                elif isinstance(value, list):
                    proposal_kwargs[key] = List(list=value)
                elif isinstance(value, dict):
                    proposal_kwargs[key] = Dict(dict=value)
                else:
                    proposal_kwargs[key] = value
        
        # For initial proposal, use AFM results (holistic mode doesn't apply here)
        proposed_matrices_pks = aiida_propose_occ_matrices_from_results(
            occ_matr_pks=mag_scan_matrices,
            calc_pks=mag_scan_calculation_pks,
            N=self.inputs.N,
            debug=self.inputs.proposal_debug,
            mode=self.inputs.proposal_mode,
            hubbard_corr_atoms=self.inputs.mag_scan.hubbard_corr_atoms,
            **proposal_kwargs
        )
        
        # Store PKs of the proposal nodes for the next constrained scan
        self.ctx.current_proposals = proposed_matrices_pks.get_list()

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
        n_proposals = min(self.inputs.N.value, 
                         self.inputs.Nmax.value - self.ctx.N_cumulative)
        
        self.report(f"Starting generation {self.ctx.generation} with {n_proposals} proposals")
        
        # Take only the number of proposals we need
        # THIS MIGHT NEED A CHANGE IF ONE NEEDS TO CHANGET THE NUMBER
        # OF PROPOSALS PER GENERATION, FOR INSTANCE FOR INITIAL GENERATION
        # OF GAUSSIAN PROCESS PROPOSALS
        
        current_proposals = self.ctx.current_proposals[:n_proposals]
        
        # Build constrained scan
        constrained_builder = ConstrainedScanWorkChain.get_builder()
        constrained_builder.update(self.inputs.constrained)
        constrained_builder.occupation_matrices_list = List(list=current_proposals)
        
        # Override walltime if provided at global level
        if 'constrained_walltime_hours' in self.inputs:
            constrained_builder.walltime_hours = self.inputs.constrained_walltime_hours
            self.report(f"Using global constrained walltime: {self.inputs.constrained_walltime_hours.value} hours")
        
        future = self.submit(constrained_builder)
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
        self.ctx.generation_results[self.ctx.generation] = {
            'type': 'constrained',
            'n_calculations': n_total,
            'n_successful': n_successful,
            'n_failed': failed_count,
            'matrix_pks': constrained_matrices.get_list(),
            'calculation_pks': calculation_pks.get_list(),
            'converged_calculation_pks': converged_calculations.get_list()
        }
        
        # Update cumulative storage
        # self.ctx.all_matrix_pks.extend(constrained_matrices.get_list()) # this one is probably redundant

        self.ctx.converged_calculation_pks.extend(converged_calculations.get_list())   
        self.ctx.converged_matrix_pks.extend(current_converged_matrices)  # Only successful results
        self.ctx.all_calculation_pks.extend(calculation_pks.get_list())
        
        self.report(f"Generation {self.ctx.generation} completed: "
                   f"{n_successful}/{n_total} successful calculations")
        
        # If we haven't reached Nmax, propose new matrices for next iteration
        if self.ctx.N_cumulative + len(calculation_pks.get_list()) < self.inputs.Nmax.value:
            proposal_kwargs = {}

            # add generation number to proposal kwargs
            proposal_kwargs['current_generation'] = Int(self.ctx.generation)

            if 'proposal_kwargs' in self.inputs:
                # Convert proposal_kwargs to AiiDA types if needed
                for key, value in self.inputs.proposal_kwargs.get_dict().items():
                    if isinstance(value, str):
                        proposal_kwargs[key] = Str(value)
                    elif isinstance(value, (int, float)):
                        proposal_kwargs[key] = Float(value) if isinstance(value, float) else Int(value)
                    elif isinstance(value, bool):
                        proposal_kwargs[key] = Bool(value)
                    elif isinstance(value, list):
                        proposal_kwargs[key] = List(list=value)
                    elif isinstance(value, dict):
                        proposal_kwargs[key] = Dict(dict=value)
                    else:
                        proposal_kwargs[key] = value
                
            
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
                **proposal_kwargs
            )
            
            # Store PKs of the proposal nodes for the next constrained scan
            self.ctx.current_proposals = proposed_matrices_pks.get_list()

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


