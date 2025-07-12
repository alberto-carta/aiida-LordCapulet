from aiida.engine import WorkChain, ToContext, submit, append_
from aiida.orm import load_group,List, Dict, Code, KpointsData, StructureData, Float, Str 
from aiida.plugins import CalculationFactory
# import UpfData
from aiida.orm import UpfData
import numpy as np
from aiida_quantumespresso.data.hubbard_structure import HubbardStructureData
# load group

PwCalculation = CalculationFactory('quantumespresso.pw')

class AFMScanWorkChain(WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)
        # Accept both StructureData and HubbardStructureData
        spec.input('structure', valid_type=(StructureData, HubbardStructureData))
        spec.input('parameters', valid_type=Dict)
        spec.input('kpoints', valid_type=KpointsData)
        # spec.input('pseudos', valid_type=Dict)
        spec.input('code', valid_type=Code)
        spec.input('tm_atoms', valid_type=List)
        spec.input('magnitude', valid_type=Float, default=Float(0.5))  
        spec.outline(
            cls.prepare_configs,
            cls.run_all,
            cls.gather_results,
        )
        spec.output('all_occupation_matrices', valid_type=List)

    def prepare_configs(self):
        tm_atoms = self.inputs.tm_atoms.get_list()
        N = len(tm_atoms)
        self.ctx.magnetic_configs = []
        for i in range(2 ** N):
            config = {}
            binary_string = format(i, f'0{N}b')
            for j in range(N):
                config[tm_atoms[j]] = self.inputs.magnitude * (1 if binary_string[j] == '1' else -1)
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
            
            pseudo_family = load_group('SSSP/1.3/PBEsol/efficiency')
            pseudos = pseudo_family.get_pseudos(structure=builder.structure)
            # builder.pseudos = self.inputs.pseudos 
            builder.pseudos = pseudos

            builder.parameters['SYSTEM']['starting_magnetization'] = starting_magnetization
            builder.metadata.options = {'resources': {'num_machines': 1}, 'withmpi': True}
            # <<< CORRECT KEY FOR OCCUPATION MATRICES >>>
            builder.settings = Dict(dict={'parser_options': {'parse_atomic_occupations': True}})
            # self.ctx.calc_futures.append(self.submit(builder))
            self.to_context(calcs=append_(self.submit(builder)))
        # return ToContext(calcs=self.ctx.calc_futures)

    # def gather_results(self):
    #     matrices = []
    #     for calc in self.ctx.calcs:
    #         if 'output_atomic_occupations' in calc.outputs:
    #             uuid = str(calc.outputs.output_atomic_occupations.uuid)
    #         else:
    #             uuid = 'no matrix'
    #         matrices.append(uuid)
    #     self.out('all_occupation_matrices', List(list=matrices).store())
    def gather_results(self):
        matrices = []
        for calc in self.ctx.calcs:
            if 'output_atomic_occupations' in calc.outputs:
                pk = calc.outputs.output_atomic_occupations.pk
            else:
                pk = -1  # Or any sentinel value you prefer for "no matrix"
            matrices.append(pk)
        self.out('all_occupation_matrices', List(list=matrices).store())