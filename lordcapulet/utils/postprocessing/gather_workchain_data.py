#!/usr/bin/env python3
"""
Script to recursively traverse a workchain and extract data from converged PW and ConstrainedPW calculations.

This module provides functionality to:
1. Recursively traverse a workchain's called links
2. Identify PW and ConstrainedPW calculations
3. Extract output_atomic_occupations, output_parameters, and inputs for converged calculations
4. Store the data in a structured dictionary and save as JSON

Usage:
    from lordcapulet.utils.postprocessing.gather_workchain_data import gather_workchain_data
    
    # Gather data from workchain with PK 12345 and save to file
    gather_workchain_data(12345, "workchain_data.json")
"""

import json
import warnings
from typing import Dict, Any, List, Union, Tuple
from pathlib import Path
from datetime import datetime
from aiida.orm import load_node, CalcJobNode, WorkChainNode
from aiida.common.exceptions import NotExistent
from aiida.plugins import CalculationFactory

# Try to import alive_progress for progress bar
try:
    from alive_progress import alive_bar
    HAS_ALIVE_BAR = True
except ImportError:
    HAS_ALIVE_BAR = False
    warnings.warn("alive_progress not available. Install with: pip install alive-progress")

# Try to import custom calculation types
try:
    from lordcapulet.calculations.constrained_pw import ConstrainedPWCalculation
    HAS_CONSTRAINED_PW = True
except ImportError:
    HAS_CONSTRAINED_PW = False
    warnings.warn("Could not import ConstrainedPWCalculation. Will try to identify by process type.")

def is_pw_calculation(node: CalcJobNode) -> bool:
    """
    Check if a calculation node is a PW or ConstrainedPW calculation.
    
    Args:
        node: AiiDA calculation node to check
        
    Returns:
        bool: True if node is a PW or ConstrainedPW calculation
    """
    if not isinstance(node, CalcJobNode):
        return False
    
    # Check process type string
    process_type = getattr(node, 'process_type', '')
    
    # Check for standard PW calculation
    if 'quantumespresso.pw' in process_type:
        return True
    
    # Check for ConstrainedPW calculation
    if 'lordcapulet.constrained_pw' in process_type or 'ConstrainedPW' in process_type:
        return True
    
    # Check by class if available
    if HAS_CONSTRAINED_PW:
        try:
            PwCalculation = CalculationFactory('quantumespresso.pw')
            if isinstance(node, (PwCalculation, ConstrainedPWCalculation)):
                return True
        except Exception:
            pass
    
    # Additional check by node attributes or labels
    node_type = getattr(node, 'node_type', '').lower()
    if 'pw' in node_type and 'calculation' in node_type:
        return True
        
    return False


def extract_calculation_data(calc_node: CalcJobNode) -> Dict[str, Any]:
    """
    Extract relevant data from a converged PW/ConstrainedPW calculation.
    
    Args:
        calc_node: AiiDA calculation node
        
    Returns:
        dict: Dictionary containing extracted data with keys:
              - pk: Primary key of the calculation
              - exit_status: Exit status of the calculation
              - inputs: Dictionary of input parameters
              - output_parameters: Output parameters if available
              - output_atomic_occupations: Atomic occupations if available
              - process_type: Type of the calculation process
    """
    data = {
        'pk': calc_node.pk,
        'exit_status': calc_node.exit_status,
        'process_type': getattr(calc_node, 'process_type', 'unknown'),
        'inputs': {},
        'output_parameters': None,
        'output_atomic_occupations': None
    }
    
    # Extract inputs
    try:
        inputs = calc_node.inputs
        for key, input_node in inputs.items():
            try:
                # Try to get dictionary representation for Dict nodes
                if hasattr(input_node, 'get_dict'):
                    data['inputs'][key] = input_node.get_dict()
                # For other node types, store basic info
                else:
                    data['inputs'][key] = {
                        'node_type': input_node.node_type,
                        'pk': input_node.pk,
                        'uuid': str(input_node.uuid)
                    }
            except Exception as e:
                data['inputs'][key] = f"Error extracting input: {str(e)}"
                
    except Exception as e:
        data['inputs'] = f"Error accessing inputs: {str(e)}"
    
    # Extract output_parameters
    try:
        if 'output_parameters' in calc_node.outputs:
            output_params = calc_node.outputs.output_parameters
            if hasattr(output_params, 'get_dict'):
                data['output_parameters'] = output_params.get_dict()
    except Exception as e:
        data['output_parameters'] = f"Error extracting output_parameters: {str(e)}"
    
    # Extract output_atomic_occupations
    try:
        if 'output_atomic_occupations' in calc_node.outputs:
            occupations = calc_node.outputs.output_atomic_occupations
            if hasattr(occupations, 'get_dict'):
                data['output_atomic_occupations'] = occupations.get_dict()
            elif hasattr(occupations, 'get_array'):
                # For ArrayData nodes
                data['output_atomic_occupations'] = {
                    name: array.tolist() for name, array in occupations.get_arraydict().items()
                }
    except Exception as e:
        data['output_atomic_occupations'] = f"Error extracting output_atomic_occupations: {str(e)}"
    
    return data


def discover_pw_calculations(node, visited: set = None, depth: int = 0, max_depth: int = 50, debug: bool = False) -> List[Tuple[int, int, str]]:
    """
    Recursively discover all converged PW/ConstrainedPW calculations in a workchain.
    
    Args:
        node: Starting AiiDA node (workchain or calculation)
        visited: Set of already visited node PKs to avoid infinite loops
        depth: Current recursion depth
        max_depth: Maximum recursion depth to prevent infinite recursion
        debug: If True, print detailed traversal information
        
    Returns:
        list: List of tuples (pk, exit_status, process_type) for converged PW/ConstrainedPW calculations only
    """
    if visited is None:
        visited = set()
    
    if depth > max_depth:
        if debug:
            warnings.warn(f"Maximum recursion depth ({max_depth}) reached. Stopping traversal.")
        return []
    
    if node.pk in visited:
        return []
    
    visited.add(node.pk)
    calculations = []
    
    # If current node is a PW/ConstrainedPW calculation, check if converged before adding
    if isinstance(node, CalcJobNode) and is_pw_calculation(node):
        process_type = getattr(node, 'process_type', 'unknown')
        exit_status = getattr(node, 'exit_status', None)
        
        if exit_status == 0:  # Only collect converged calculations
            calculations.append((node.pk, exit_status, process_type))
            if debug:
                print(f"{'  ' * depth}Found converged {process_type} calculation: PK {node.pk}")
        else:
            if debug:
                print(f"{'  ' * depth}Skipping non-converged {process_type} calculation: PK {node.pk} (exit_status={exit_status})")
        
        return calculations
    
    # If it's a workchain, traverse its called links
    if hasattr(node, 'called'):
        try:
            called_nodes = node.called
            if debug:
                print(f"{'  ' * depth}Traversing workchain: PK {node.pk}, found {len(called_nodes)} called nodes")
            
            for called_node in called_nodes:
                child_calculations = discover_pw_calculations(called_node, visited, depth + 1, max_depth, debug)
                calculations.extend(child_calculations)
                
        except Exception as e:
            if debug:
                print(f"{'  ' * depth}Error accessing called nodes for PK {node.pk}: {str(e)}")
    
    return calculations


def discover_all_pw_calculations_for_stats(node, visited: set = None, depth: int = 0, max_depth: int = 50) -> Tuple[int, int, Dict[str, int], Dict[str, int], List[Dict[str, Any]]]:
    """
    Recursively discover ALL PW/ConstrainedPW calculations for statistics purposes.
    
    Args:
        node: Starting AiiDA node (workchain or calculation)
        visited: Set of already visited node PKs to avoid infinite loops
        depth: Current recursion depth
        max_depth: Maximum recursion depth to prevent infinite recursion
        
    Returns:
        tuple: (total_calcs, converged_calcs, exit_status_counts, calc_type_counts, non_converged_details)
    """
    if visited is None:
        visited = set()
    
    if depth > max_depth:
        return 0, 0, {}, {}, []
    
    if node.pk in visited:
        return 0, 0, {}, {}, []
    
    visited.add(node.pk)
    
    total_calcs = 0
    converged_calcs = 0
    exit_status_counts = {}
    calc_type_counts = {}
    non_converged_details = []
    
    # If current node is a PW/ConstrainedPW calculation, count it
    if isinstance(node, CalcJobNode) and is_pw_calculation(node):
        process_type = getattr(node, 'process_type', 'unknown')
        exit_status = getattr(node, 'exit_status', None)
        
        total_calcs = 1
        calc_type_counts[process_type] = 1
        exit_status_counts[str(exit_status)] = 1
        
        if exit_status == 0:
            converged_calcs = 1
        else:
            non_converged_details.append({
                'pk': node.pk,
                'exit_status': exit_status,
                'process_type': process_type
            })
        
        return total_calcs, converged_calcs, exit_status_counts, calc_type_counts, non_converged_details
    
    # If it's a workchain, traverse its called links
    if hasattr(node, 'called'):
        try:
            called_nodes = node.called
            
            for called_node in called_nodes:
                child_total, child_conv, child_exit, child_types, child_non_conv = discover_all_pw_calculations_for_stats(
                    called_node, visited, depth + 1, max_depth)
                
                total_calcs += child_total
                converged_calcs += child_conv
                non_converged_details.extend(child_non_conv)
                
                # Merge dictionaries
                for status, count in child_exit.items():
                    exit_status_counts[status] = exit_status_counts.get(status, 0) + count
                
                for calc_type, count in child_types.items():
                    calc_type_counts[calc_type] = calc_type_counts.get(calc_type, 0) + count
                
        except Exception:
            pass  # Silently ignore errors in stats collection
    
    return total_calcs, converged_calcs, exit_status_counts, calc_type_counts, non_converged_details


def process_calculations(calculation_list: List[Tuple[int, int, str]], debug: bool = False) -> List[Dict[str, Any]]:
    """
    Process a list of converged calculations and extract data.
    
    Args:
        calculation_list: List of tuples (pk, exit_status, process_type) - all should be converged
        debug: If True, print detailed processing information
        
    Returns:
        list: List of calculation data dictionaries
    """
    results = []
    
    if HAS_ALIVE_BAR and len(calculation_list) > 0:
        with alive_bar(len(calculation_list), title="Processing calculations") as bar:
            for pk, exit_status, process_type in calculation_list:
                try:
                    calc_node = load_node(pk)
                    calc_data = extract_calculation_data(calc_node)
                    results.append(calc_data)
                    
                    if debug:
                        print(f"Processed converged {process_type} calculation: PK {pk}")
                        
                except Exception as e:
                    if debug:
                        print(f"Error processing calculation PK {pk}: {str(e)}")
                
                bar()
    else:
        # Fallback without progress bar
        print(f"Processing {len(calculation_list)} converged calculations...")
        for i, (pk, exit_status, process_type) in enumerate(calculation_list):
            try:
                calc_node = load_node(pk)
                calc_data = extract_calculation_data(calc_node)
                results.append(calc_data)
                
                if debug:
                    print(f"Processed converged {process_type} calculation: PK {pk}")
                elif i % max(1, len(calculation_list) // 10) == 0:  # Print progress every 10%
                    progress = (i + 1) / len(calculation_list) * 100
                    print(f"Progress: {progress:.1f}% ({i + 1}/{len(calculation_list)})")
                    
            except Exception as e:
                if debug:
                    print(f"Error processing calculation PK {pk}: {str(e)}")
    
    return results
def gather_workchain_data(workchain_pk: int, output_filename: str, debug: bool = False) -> Dict[str, Any]:
    """
    Main function to gather data from a workchain and save to JSON file.
    
    Args:
        workchain_pk: Primary key of the root workchain
        output_filename: Name of the output JSON file
        debug: If True, print detailed traversal information (default: False)
        
    Returns:
        dict: Complete data dictionary containing all extracted information
        
    Raises:
        NotExistent: If the workchain node cannot be loaded
        Exception: For other errors during data extraction or file writing
    """
    try:
        # Load the root workchain node
        root_node = load_node(workchain_pk)
        print(f"Loading workchain: PK {workchain_pk}, type: {type(root_node).__name__}")
        
        # Phase 1: Discover all PW/ConstrainedPW calculations for statistics
        print("Phase 1: Discovering all PW/ConstrainedPW calculations for statistics...")
        total_calcs, converged_count, exit_status_counts, calc_type_counts, non_converged_details = discover_all_pw_calculations_for_stats(root_node)
        
        print(f"Found {total_calcs} total PW/ConstrainedPW calculations ({converged_count} converged, {total_calcs - converged_count} non-converged)")
        
        # Phase 2: Discover only converged calculations for processing
        print("Phase 2: Collecting converged calculations for processing...")
        converged_calculation_list = discover_pw_calculations(root_node, debug=debug)
        
        # Phase 3: Process the converged calculations and extract data
        print("Phase 3: Processing converged calculations and extracting data...")
        calculations_data = process_calculations(converged_calculation_list, debug=debug)
        
        # Build statistics dictionary
        statistics = {
            'total_pw_calculations': total_calcs,
            'converged_calculations': converged_count,
            'non_converged_calculations': total_calcs - converged_count,
            'exit_status_counts': exit_status_counts,
            'calculation_types': calc_type_counts,
            'non_converged_details': non_converged_details
        }
        
        # Calculate additional statistics
        convergence_rate = (statistics['converged_calculations'] / statistics['total_pw_calculations'] * 100) if statistics['total_pw_calculations'] > 0 else 0
        
        # Print statistics
        print("\n" + "="*50)
        print("WORKCHAIN STATISTICS:")
        print("="*50)
        print(f"Total PW/ConstrainedPW calculations found: {statistics['total_pw_calculations']}")
        print(f"Converged calculations: {statistics['converged_calculations']}")
        print(f"Non-converged calculations: {statistics['non_converged_calculations']}")
        print(f"Convergence rate: {convergence_rate:.1f}%")
        
        print("\nCalculation types:")
        for calc_type, count in statistics['calculation_types'].items():
            print(f"  {calc_type}: {count}")
        
        print("\nExit status distribution:")
        for exit_status, count in statistics['exit_status_counts'].items():
            status_name = "SUCCESS" if exit_status == "0" else f"ERROR_{exit_status}"
            print(f"  {status_name}: {count}")
        
        if statistics['non_converged_calculations'] > 0:
            print(f"\nNon-converged calculations details:")
            for detail in statistics['non_converged_details'][:5]:  # Show first 5
                print(f"  PK {detail['pk']}: exit_status={detail['exit_status']}, type={detail['process_type']}")
            if len(statistics['non_converged_details']) > 5:
                print(f"  ... and {len(statistics['non_converged_details']) - 5} more")
        
        print("="*50)
        
        # Add convergence rate to statistics
        statistics['convergence_rate_percent'] = round(convergence_rate, 2)
        
        # Prepare final data structure
        output_data = {
            'metadata': {
                'root_workchain_pk': workchain_pk,
                'root_workchain_type': getattr(root_node, 'process_type', type(root_node).__name__),
                'total_calculations_found': len(calculations_data),
                'extraction_timestamp': datetime.now().isoformat()
            },
            'statistics': statistics,
            'calculations': {}
        }
        
        # Store calculations data with PK as key
        for calc_data in calculations_data:
            pk = calc_data['pk']
            output_data['calculations'][str(pk)] = calc_data
        
        # Save to JSON file
        output_path = Path(output_filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nData extraction completed successfully!")
        print(f"Total calculations found: {statistics['total_pw_calculations']} ({statistics['converged_calculations']} converged, {statistics['non_converged_calculations']} non-converged)")
        print(f"Convergence rate: {convergence_rate:.1f}%")
        print(f"Data saved to: {output_path.absolute()}")
        
        return output_data
        
    except NotExistent:
        raise NotExistent(f"Could not load node with PK {workchain_pk}. Please check that the PK is correct.")
    
    except Exception as e:
        print(f"Error during data extraction: {str(e)}")
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract data from converged PW/ConstrainedPW calculations in a workchain"
    )
    parser.add_argument("pk", type=int, help="Primary key of the root workchain")
    parser.add_argument("output", help="Output JSON filename")
    parser.add_argument("--max-depth", type=int, default=50, 
                       help="Maximum recursion depth (default: 50)")
    parser.add_argument("--debug", action="store_true", 
                       help="Print detailed traversal information")
    
    args = parser.parse_args()
    
    try:
        gather_workchain_data(args.pk, args.output, debug=args.debug)
    except Exception as e:
        print(f"Failed to extract data: {e}")
        exit(1)
