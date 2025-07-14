from aiida_quantumespresso.calculations.pw import PwCalculation
from aiida.orm import Dict
import numpy as np
from typing import Any, Dict as TypedDict

class CustomPwCalculation(PwCalculation):
    """Custom PwCalculation that writes an additional hello_world.in file."""

    def prepare_for_submission(self, folder):
        calcinfo = super().prepare_for_submission(folder)
        with folder.open('hello_world.in', 'w') as handle:
            handle.write('hello world\n')
        calcinfo.retrieve_list.append('hello_world.in')
        return calcinfo



class ConstrainedPWCalculation(PwCalculation):
    """
    PwCalculation subclass that takes two dictionaries as extra inputs
    and writes a file based on a user-defined function.
    """

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('oscdft_card', valid_type=Dict, help='Constraint parameters for PW calculation')
        spec.input('target_matrix', valid_type=Dict, help='Target occupation matrix')
        # Optionally, mark as not required for backwards compatibility:
        # spec.input('input1', valid_type=Dict, required=False)
        # spec.input('input2', valid_type=Dict, required=False)

    def write_oscdft_data(self, parameters: TypedDict[str, Any], occupation_numbers: np.ndarray) -> str:
        """
        Writes the structured OSCDFT data to a string in the correct format.
        
        Args:
            parameters: A dictionary containing the OSCDFT parameters.
            occupation_numbers: A 4D numpy array with the occupation numbers.
            
        Returns:
            str: The formatted OSCDFT input string.
        """
        output_lines = []
        
        # Write &OSCDFT section
        output_lines.append(" &OSCDFT")
        for key, value in parameters.items():
            # Format values back to string
            value_str = str(value)
            output_lines.append(f" {key} = {value_str},")
        output_lines.append("/")

        # Write TARGET_OCCUPATION_NUMBERS section
        output_lines.append("TARGET_OCCUPATION_NUMBERS")
        # Iterate through the 4D numpy array using ndindex to get all indices
        for idx_tuple in np.ndindex(occupation_numbers.shape):
            value = occupation_numbers[idx_tuple]
            
            # Convert 0-based Python indices back to 1-based for the output file
            idx1, idx2, idx3, idx4 = (i + 1 for i in idx_tuple)

            # Format each row with appropriate spacing
            # str(idx).rjust(2) is equivalent to Julia's lpad(string(idx), 2)
            # f"{value:8.3f}" is equivalent to Julia's @sprintf("%8.3f", value)
            formatted_row = (
                f"{str(idx1).rjust(2)} "
                f"{str(idx2).rjust(2)} "
                f"{str(idx3).rjust(2)} "
                f"{str(idx4).rjust(2)} "
                f"{value:8.3f}"
            )
            output_lines.append(f" {formatted_row}")
        
        return "\n".join(output_lines)

    def prepare_for_submission(self, folder):
        # Call super to get standard QE input
        calcinfo = super().prepare_for_submission(folder)

        # Get the input dicts
        oscdft_parameters = self.inputs.oscdft_card.get_dict()
        target_matrix_dict = self.inputs.target_matrix.get_dict()
        
        # Convert target matrix dict to numpy array (assuming it's stored as nested lists or similar)
        # You may need to adjust this conversion based on how the data is structured
        target_matrix = np.array(target_matrix_dict.get('matrix', []))

        # Generate OSCDFT input content
        oscdft_content = self.write_oscdft_data(oscdft_parameters, target_matrix)

        # Write the OSCDFT input to a file in the calculation folder
        with folder.open('oscdft.in', 'w') as handle:
            handle.write(oscdft_content)

        # Add the new file to the retrieve list
        if hasattr(calcinfo, "retrieve_list"):
            calcinfo.retrieve_list.append('oscdft.in')
        else:
            # Fallback for legacy calcinfo
            calcinfo['retrieve_list'].append('oscdft.in')

        return calcinfo