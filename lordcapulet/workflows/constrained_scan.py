from aiida.engine import WorkChain, ToContext, submit, append_
from aiida.orm import load_group, List, Dict, Code, KpointsData, StructureData, Float, Str 
from aiida.plugins import CalculationFactory
import numpy as np
from aiida_quantumespresso.data.hubbard_structure import HubbardStructureData

# Import the custom constrained calculation
from lordcapulet.calculations.constrained_pw import ConstrainedPWCalculation

class ConstrainedScanWorkChain(WorkChain):
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
        
        # Optional pseudo family string (fallback to hardcoded if not provided)
        # spec.input('pseudo_family_string', valid_type=Str, 
        #           default=lambda: Str('SSSP/1.3/PBEsol/efficiency'))
        
        spec.outline(
            cls.prepare_calculations,
            cls.run_all,
            cls.gather_results,
        )
        
        # Outputs
        spec.output('all_occupation_matrices', valid_type=List)
        spec.output('calculation_pks', valid_type=List)

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
        
        # Load pseudo family
        # pseudo_family_string = self.inputs.pseudo_family_string.value
        # pseudo_family = load_group(pseudo_family_string)
        # pseudos = pseudo_family.get_pseudos(structure=self.inputs.structure)
        
        for i, target_matrix_dict in enumerate(self.ctx.target_matrices):
            self.report(f"Submitting calculation {i+1}/{self.ctx.n_calculations}")
            
            # Build the calculation
            builder = ConstrainedPWCalculation.get_builder()
            builder.code = self.inputs.code
            builder.structure = self.inputs.structure
            builder.parameters = self.inputs.parameters.clone()
            builder.kpoints = self.inputs.kpoints
            # this is hardcoded for now, needs to be improved 
            pseudo_family = load_group('SSSP/1.3/PBEsol/efficiency')
            pseudos = pseudo_family.get_pseudos(structure=builder.structure)
            builder.pseudos = pseudos
            
            # Set magnetization for all transition metal atoms to a small value
            tm_atoms = self.inputs.tm_atoms.get_list()
            magnetization_config = {}
            for tm_atom in tm_atoms:
                magnetization_config[tm_atom] = 1e-9
            
            # Add the magnetization to the parameters
            builder.parameters['SYSTEM']['starting_magnetization'] = magnetization_config

            # Set OSCDFT specific inputs
            builder.oscdft_card = self.inputs.oscdft_card

            
            # Convert the target matrix to the format expected by ConstrainedPWCalculation
            # target_matrix = target_matrix_dict.get_dict().get('matrix', [])
            # natoms = len(target_matrix)
            # target_matrix = [np.array(target_matrix[iatom]) for iatom in range(natoms)]
            builder.target_matrix = target_matrix_dict
            
            # Set computational options
            # builder.metadata.options = {
            #     'resources': {'num_machines': 1}, 
            #     'withmpi': True
            # }

            
            # Enable parsing of occupation matrices and add oscdft flag
            builder.settings = Dict(dict={
                'parser_options': {'parse_atomic_occupations': True},
                'CMDLINE': ['-oscdft'],
            })
            
            # Submit and store in context
            self.to_context(calcs=append_(self.submit(builder)))

    def gather_results(self):
        """
        Collect the PKs results from all calculations.
        """
        matrices = []
        calculation_pks = []
        
        for i, calc in enumerate(self.ctx.calcs):
            calculation_pks.append(calc.pk)
            
            if 'output_atomic_occupations' in calc.outputs:
                # Store the PK of the output occupation matrix
                pk = calc.outputs.output_atomic_occupations.pk
                matrices.append(pk)
                self.report(f"Calculation {i+1} completed successfully with occupation matrix PK: {pk}")
            else:
                # Use -1 as sentinel value for failed calculations
                matrices.append(-1)
                self.report(f"Calculation {i+1} completed but no occupation matrix found")
        
        # Store outputs
        self.out('all_occupation_matrices', List(list=matrices).store())
        self.out('calculation_pks', List(list=calculation_pks).store())
        
        self.report(f"Constrained scan completed. {len([m for m in matrices if m != -1])}/{len(matrices)} calculations successful")
