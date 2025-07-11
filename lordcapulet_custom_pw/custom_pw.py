from aiida_quantumespresso.calculations.pw import PwCalculation

class CustomPwCalculation(PwCalculation):
    """Custom PwCalculation that writes an additional hello_world.in file."""

    def prepare_for_submission(self, folder):
        calcinfo = super().prepare_for_submission(folder)
        with folder.open('hello_world.in', 'w') as handle:
            handle.write('hello world\n')
        calcinfo.retrieve_list.append('hello_world.in')
        return calcinfo
