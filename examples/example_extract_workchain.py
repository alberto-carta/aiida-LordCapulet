#!/usr/bin/env python3
"""
Example script demonstrating how to use the gather_workchain_data function.

This script shows different ways to call the data gathering function and 
provides example usage patterns.
"""
#%%
from lordcapulet.utils.postprocessing.gather_workchain_data import gather_workchain_data

def analyze_statistics_example():
    """Example showing how to analyze the extracted statistics."""
    print("Example 3: Analyzing extracted statistics")
    
    # This would use real data from a previous run
    # workchain_pk = 19232 # FeO
    # workchain_pk = 24930 # CoO
    # workchain_pk = 25730 # CuO
    # workchain_pk = 36461 # VO
    workchain_pk = 36496 # CrO

    output_file = "analysis_data.json"
    
    try:
        data = gather_workchain_data(workchain_pk, output_file)
        stats = data['statistics']
        
        print(f"\nDetailed Analysis for Workchain {workchain_pk}:")
        print(f"Total calculations: {stats['total_pw_calculations']}")
        print(f"Success rate: {stats['convergence_rate_percent']:.1f}%")
        
        # Identify most problematic exit status
        non_zero_exits = {k: v for k, v in stats['exit_status_counts'].items() if k != "0"}
        if non_zero_exits:
            most_common_error = max(non_zero_exits, key=non_zero_exits.get)
            print(f"Most common error: exit_status {most_common_error} ({non_zero_exits[most_common_error]} occurrences)")
        
        # Show calculation type efficiency
        print("\nCalculation type analysis:")
        for calc_type, total_count in stats['calculation_types'].items():
            # This is a simplified analysis - in practice you'd need to track
            # convergence per calculation type
            print(f"  {calc_type}: {total_count} calculations")
        
    except Exception as e:
        print(f"Error in analysis: {e}")
#%%
import aiida
aiida.load_profile()
analyze_statistics_example()

#%%
if __name__ == "__main__":
    # You can run this script directly or import the function
    print("LordCapulet Workchain Data Gatherer")
    print("="*40)
    
    # Uncomment the line below to run the examples
    # example_usage()
    # analyze_statistics_example()
    
    # Or use command line arguments
    import sys
    if len(sys.argv) == 3:
        try:
            pk = int(sys.argv[1])
            output_file = sys.argv[2]
            data = gather_workchain_data(pk, output_file)
            
            # Print quick summary
            stats = data['statistics']
            print(f"\nQuick Summary:")
            print(f"Total calculations: {stats['total_pw_calculations']}")
            print(f"Converged: {stats['converged_calculations']} ({stats['convergence_rate_percent']}%)")
            print(f"Failed: {stats['non_converged_calculations']}")
            
        except ValueError:
            print("Usage: python example_usage.py <workchain_pk> <output_file>")
            print("       where workchain_pk is an integer")
    elif len(sys.argv) != 1:
        print("Usage: python example_usage.py <workchain_pk> <output_file>")
        print("Example: python example_usage.py 12345 my_data.json")
    else:
        print("Edit this script to use your workchain PKs, or run with arguments:")
        print("python example_usage.py <workchain_pk> <output_file>")
        print("\nAvailable examples:")
        print("- example_usage(): Basic usage and multiple workchain processing") 
        print("- analyze_statistics_example(): Statistical analysis of extracted data")
