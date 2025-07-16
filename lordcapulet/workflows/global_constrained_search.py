from aiida.engine import WorkChain, ToContext, while_, if_, submit
from aiida.orm import Dict, List, Int, Float, Str, Bool, Code, KpointsData, StructureData, load_node
from aiida.plugins import CalculationFactory
from aiida_quantumespresso.data.hubbard_structure import HubbardStructureData

# Import the custom workchains
from lordcapulet.workflows.afm_scan import AFMScanWorkChain
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
        
        # Expose inputs from AFMScanWorkChain
        spec.expose_inputs(AFMScanWorkChain, namespace='afm')
        
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
        
        spec.outline(
            cls.run_initial_afm_search,
            cls.process_afm_results,
            while_(cls.should_continue_search)(
                cls.run_constrained_batch,
                cls.process_constrained_results,
                cls.update_counters,
            ),
            cls.gather_final_results,
        )
        
        # Outputs
        spec.output('all_afm_matrices', valid_type=List,
                   help='Occupation matrices from initial AFM search')
        spec.output('all_constrained_matrices', valid_type=List,
                   help='All occupation matrices from constrained calculations')
        spec.output('all_calculation_pks', valid_type=List,
                   help='PKs of all calculations performed')
        spec.output('generation_summary', valid_type=Dict,
                   help='Summary of results per generation')
        
        # Exit codes
        spec.exit_code(400, 'ERROR_AFM_SEARCH_FAILED',
                      message='Initial AFM search failed')
        spec.exit_code(401, 'ERROR_CONSTRAINED_SCAN_FAILED',
                      message='Constrained scan failed')
        spec.exit_code(402, 'ERROR_PROPOSAL_FAILED',
                      message='Matrix proposal step failed')

    def run_initial_afm_search(self):
        """
        Run the initial AFM search to get starting occupation matrices.
        """
        self.report("Starting initial AFM search")
        
        # Submit AFM scan with exposed inputs
        afm_builder = AFMScanWorkChain.get_builder()
        afm_builder.update(self.inputs.afm)
        
        future = self.submit(afm_builder)
        return ToContext(afm_wc=future)

    def process_afm_results(self):
        """
        Process AFM results and propose initial matrices for constrained calculations.
        """
        if not self.ctx.afm_wc.is_finished_ok:
            return self.exit_codes.ERROR_AFM_SEARCH_FAILED
            
        self.report("AFM search completed successfully, processing results")
        
        # Get AFM occupation matrices
        afm_matrices = self.ctx.afm_wc.outputs.all_occupation_matrices
        self.ctx.all_afm_matrices = afm_matrices
        
        # Initialize counters and storage
        self.ctx.N_cumulative = 0
        self.ctx.generation = 0
        self.ctx.all_matrices_pks = afm_matrices.get_list().copy()
        self.ctx.result_matrices_pks = afm_matrices.get_list().copy()  # Only successful result matrices
        self.ctx.all_calculation_pks = []
        self.ctx.generation_results = {}
        
        # Store AFM results
        self.ctx.generation_results[0] = {
            'type': 'afm',
            'n_calculations': len(afm_matrices.get_list()),
            'matrix_pks': afm_matrices.get_list()
        }
        
        # Propose initial matrices for first constrained batch
        proposal_kwargs = {}
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
            pk_list=afm_matrices,
            N=self.inputs.N,
            debug=self.inputs.proposal_debug,
            mode=self.inputs.proposal_mode,
            **proposal_kwargs
        )
        
        # Load the actual dictionaries for the next constrained scan
        self.ctx.current_proposals = [
            load_node(pk).get_dict() 
            for pk in proposed_matrices_pks.get_list()
        ]

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
        current_proposals = self.ctx.current_proposals[:n_proposals]
        
        # Build constrained scan
        constrained_builder = ConstrainedScanWorkChain.get_builder()
        constrained_builder.update(self.inputs.constrained)
        constrained_builder.occupation_matrices_list = List(list=current_proposals)
        
        future = self.submit(constrained_builder)
        return ToContext(constrained_wc=future)

    def process_constrained_results(self):
        """
        Process results from constrained scan and propose new matrices if needed.
        """
        if not self.ctx.constrained_wc.is_finished_ok:
            self.report("Constrained scan workchain failed, but checking individual calculations")
        
        # Get results
        constrained_matrices = self.ctx.constrained_wc.outputs.all_occupation_matrices
        calculation_pks = self.ctx.constrained_wc.outputs.calculation_pks
        
        # Count successful calculations (matrices with PK != -1)
        successful_matrices = [pk for pk in constrained_matrices.get_list() if pk != -1]
        failed_count = len([pk for pk in constrained_matrices.get_list() if pk == -1])
        
        if len(successful_matrices) == 0:
            self.report(f"All {len(calculation_pks.get_list())} calculations in generation {self.ctx.generation} failed")
            return self.exit_codes.ERROR_CONSTRAINED_SCAN_FAILED
        elif failed_count > 0:
            self.report(f"Generation {self.ctx.generation}: {len(successful_matrices)} successful, {failed_count} failed calculations")
        
        # Store results
        self.ctx.generation_results[self.ctx.generation] = {
            'type': 'constrained',
            'n_calculations': len(calculation_pks.get_list()),
            'n_successful': len(successful_matrices),
            'n_failed': failed_count,
            'matrix_pks': constrained_matrices.get_list(),
            'calculation_pks': calculation_pks.get_list()
        }
        
        # Update cumulative storage
        self.ctx.all_matrices_pks.extend(constrained_matrices.get_list())
        self.ctx.result_matrices_pks.extend(successful_matrices)  # Only successful results
        self.ctx.all_calculation_pks.extend(calculation_pks.get_list())
        
        self.report(f"Generation {self.ctx.generation} completed: "
                   f"{len(successful_matrices)}/{len(calculation_pks.get_list())} successful calculations")
        
        # If we haven't reached Nmax, propose new matrices for next iteration
        if self.ctx.N_cumulative + len(calculation_pks.get_list()) < self.inputs.Nmax.value:
            proposal_kwargs = {}
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
                matrices_for_proposal = List(list=self.ctx.result_matrices_pks)
                self.report(f"Using holistic approach: analyzing {len(self.ctx.result_matrices_pks)} total result matrices")
            else:
                # Use only successful matrices from current generation (Markovian)
                matrices_for_proposal = List(list=successful_matrices)
                self.report(f"Using Markovian approach: analyzing {len(successful_matrices)} matrices from current generation")
            
            proposed_matrices_pks = aiida_propose_occ_matrices_from_results(
                pk_list=matrices_for_proposal,
                N=self.inputs.N,
                debug=self.inputs.proposal_debug,
                mode=self.inputs.proposal_mode,
                **proposal_kwargs
            )
            
            # Load the actual dictionaries for the next constrained scan
            self.ctx.current_proposals = [
                load_node(pk).get_dict() 
                for pk in proposed_matrices_pks.get_list()
            ]

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
        
        self.out('all_afm_matrices', self.ctx.all_afm_matrices)

        # The class is gathering data as it is being proced, in this final step put everything in
        # the database with the store() method.


        # the workflow cannot create aiida types
        # here you need to make sure that the instances you pass
        # to out are already stored as aiida nodes, otherwise this will return a Data exception


        all_matrices_pks = List(list=self.ctx.all_matrices_pks)
        all_matrices_pks.store()
        self.out('all_constrained_matrices', all_matrices_pks)
        # the workflow cannot create aiida types

        all_calculation_pks = List(list=self.ctx.all_calculation_pks)
        all_calculation_pks.store()
        self.out('all_calculation_pks', all_calculation_pks)

        generation_results_str_keys = Dict(dict={f"Generation {k}": v for k, v in self.ctx.generation_results.items()})
        generation_results_str_keys.store()
        self.out('generation_summary', generation_results_str_keys)

        # self.out('all_constrained_matrices', List(list=self.ctx.all_matrices_pks))
        # self.out('all_calculation_pks', List(list=self.ctx.all_calculation_pks))
        # self.out('generation_summary', Dict(dict=self.ctx.generation_results))
        # generation_results_str_keys = {f"Generation {k}": v for k, v in self.ctx.generation_results.items()}
        # self.out('generation_summary', Dict(dict=generation_results_str_keys))

