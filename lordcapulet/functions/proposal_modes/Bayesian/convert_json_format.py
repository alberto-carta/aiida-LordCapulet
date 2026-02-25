#!/usr/bin/env python3
"""
Convert old JSON format to new format.

Old format has 'inputs' and 'outputs' fields in each calculation.
New format has direct fields like 'output_parameters', 'occupation_matrices', etc.

Usage:
    python convert_json_format.py input.json output.json
"""

import json
import sys
from pathlib import Path


def convert_calculation(old_calc):
    """Convert a single calculation from old format to new format."""
    new_calc = {
        "pk": old_calc["pk"],
        "exit_status": old_calc["exit_status"],
        "converged": old_calc.get("converged", old_calc["exit_status"] == 0),
        "process_type": old_calc["process_type"],
        "calculation_source": old_calc["calculation_source"],
    }
    
    # Handle output_parameters - either nested in outputs or at top level
    if "output_parameters" in old_calc:
        # Already at top level (UO2 format)
        output_params = old_calc["output_parameters"]
        new_calc["output_parameters"] = output_params
    elif "outputs" in old_calc and "output_parameters" in old_calc["outputs"]:
        # Nested in outputs (old format)
        output_params = old_calc["outputs"]["output_parameters"]
        new_calc["output_parameters"] = output_params
    else:
        output_params = {}
    
    # Extract energy directly for easier access
    if output_params:
        if "energy" in output_params:
            new_calc["energy"] = output_params["energy"]
        elif "total_energy" in output_params:
            new_calc["energy"] = output_params["total_energy"]
    
    # Handle occupation_matrices - multiple possible sources
    if "occupation_matrices" in old_calc:
        # Already in correct format (FeO format)
        new_calc["occupation_matrices"] = old_calc["occupation_matrices"]
    elif "output_atomic_occupations" in old_calc:
        # UO2 format - needs transformation
        # Transform from: atom_id -> {spin_data: {up: {occupation_matrix: ...}, down: {...}}}
        # To: atom_id -> {occupation_matrix: {up: ..., down: ...}}
        old_occ = old_calc["output_atomic_occupations"]
        
        # Extract specie information from pseudopotential keys in inputs
        # Build ordered list of species from pseudo keys
        specie_list = []
        if "inputs" in old_calc:
            pseudo_keys = sorted([k for k in old_calc["inputs"].keys() if k.startswith("pseudos__")])
            for key in pseudo_keys:
                # Extract just the specie name (everything before the digits)
                parts = key.replace("pseudos__", "")
                for i in range(len(parts)-1, -1, -1):
                    if not parts[i].isdigit():
                        specie = parts[:i+1]
                        specie_list.append(specie)
                        break
        
        # Create specie map by matching atom_id to position in specie_list
        specie_map = {}
        specie_counter = {}
        for i, atom_id in enumerate(sorted(old_occ.keys(), key=lambda x: int(x))):
            if i < len(specie_list):
                specie = specie_list[i]
                # Count occurrences of this specie
                if specie not in specie_counter:
                    specie_counter[specie] = 0
                specie_counter[specie] += 1
                # Create label like "U_1", "U_2", "O_1", etc.
                specie_map[atom_id] = f"{specie}_{specie_counter[specie]}"
        
        new_occ = {}
        for atom_id, atom_data in old_occ.items():
            new_atom = {
                "occupation_matrix": {}
            }
            # Try to extract specie/shell info if available
            if "specie" in atom_data:
                new_atom["specie"] = atom_data["specie"]
            elif atom_id in specie_map:
                # Use specie from pseudopotential mapping
                new_atom["specie"] = specie_map[atom_id]
            
            if "shell" in atom_data:
                new_atom["shell"] = atom_data["shell"]
            
            # Extract occupation matrices from spin_data
            if "spin_data" in atom_data:
                for spin, spin_data in atom_data["spin_data"].items():
                    if "occupation_matrix" in spin_data:
                        new_atom["occupation_matrix"][spin] = spin_data["occupation_matrix"]
            
            new_occ[atom_id] = new_atom
        new_calc["occupation_matrices"] = new_occ
    elif "outputs" in old_calc and "occupation_matrices" in old_calc["outputs"]:
        # Nested in outputs (old format)
        new_calc["occupation_matrices"] = old_calc["outputs"]["occupation_matrices"]
    
    # Extract occupation_matrix_pk if present
    if "occupation_matrix_pk" in old_calc:
        new_calc["occupation_matrix_pk"] = old_calc["occupation_matrix_pk"]
    elif "outputs" in old_calc and "occupation_matrix_pk" in old_calc["outputs"]:
        new_calc["occupation_matrix_pk"] = old_calc["outputs"]["occupation_matrix_pk"]
    
    # Extract so_n_decomposition if present (decomposed matrices)
    if "so_n_decomposition" in old_calc:
        new_calc["so_n_decomposition"] = old_calc["so_n_decomposition"]
    
    return new_calc


def convert_json_format(input_file, output_file):
    """Convert entire JSON file from old format to new format."""
    print(f"Reading {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Keep metadata and statistics as-is
    new_data = {
        "metadata": data.get("metadata", {}),
        "statistics": data.get("statistics", {}),
        "calculations": {}
    }
    
    # Convert each calculation
    print(f"Converting {len(data['calculations'])} calculations...")
    for pk, calc in data["calculations"].items():
        new_data["calculations"][pk] = convert_calculation(calc)
    
    # Write output
    print(f"Writing to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(new_data, f, indent=2)
    
    print(f"Done! Converted {len(new_data['calculations'])} calculations.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_json_format.py input.json output.json")
        sys.exit(1)
    
    input_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2])
    
    if not input_file.exists():
        print(f"Error: Input file {input_file} does not exist.")
        sys.exit(1)
    
    convert_json_format(input_file, output_file)
