# Workchain Data Gatherer

This utility recursively traverses a workchain and extracts data from all converged (exit_status==0) PW and ConstrainedPW calculations.

## Usage

### As a Python Function

```python
from lordcapulet.utils.postprocessing.gather_workchain_data import gather_workchain_data

# Extract data from workchain with PK 12345
data = gather_workchain_data(12345, "my_results.json")

# Access the extracted data
print(f"Found {data['metadata']['total_calculations_found']} calculations")
for calc_pk, calc_data in data['calculations'].items():
    print(f"Calculation {calc_pk}: {calc_data['exit_status']}")
```

### Command Line Usage

```bash
# Run directly
python lordcapulet/utils/postprocessing/gather_workchain_data.py <workchain_pk> <output_file.json>

# Example
python lordcapulet/utils/postprocessing/gather_workchain_data.py 12345 workchain_data.json

# With custom max depth
python lordcapulet/utils/postprocessing/gather_workchain_data.py 12345 data.json --max-depth 100
```

### Using the Example Script

```bash
python example_usage.py 12345 my_data.json
```

## Output Format

The script generates a JSON file with the following structure:

```json
{
  "metadata": {
    "root_workchain_pk": 12345,
    "root_workchain_type": "GlobalConstrainedSearchWorkChain",
    "total_calculations_found": 25,
    "extraction_timestamp": "2024-01-01T12:00:00.123456"
  },
  "statistics": {
    "total_pw_calculations": 42,
    "converged_calculations": 25,
    "non_converged_calculations": 17,
    "convergence_rate_percent": 59.5,
    "exit_status_counts": {
      "0": 25,
      "410": 15,
      "500": 2
    },
    "calculation_types": {
      "lordcapulet.constrained_pw": 35,
      "aiida_quantumespresso.calculations.pw.PwCalculation": 7
    },
    "non_converged_details": [
      {
        "pk": 7480,
        "exit_status": 410,
        "process_type": "lordcapulet.constrained_pw"
      },
      ...
    ]
  },
  "calculations": {
    "7356": {
      "pk": 7356,
      "exit_status": 0,
      "process_type": "lordcapulet.constrained_pw",
      "inputs": {
        "parameters": { ... },
        "structure": { ... },
        "oscdft_card": { ... },
        "target_matrix": { ... }
      },
      "output_parameters": {
        "energy": -1234.567,
        "forces": [...],
        ...
      },
      "output_atomic_occupations": {
        "occupations": [...],
        ...
      }
    },
    ...
  }
}
```

## Statistics Provided

The script now provides comprehensive statistics about the workchain:

### Summary Statistics
- **total_pw_calculations**: Total number of PW/ConstrainedPW calculations found
- **converged_calculations**: Number of calculations with exit_status == 0
- **non_converged_calculations**: Number of calculations with exit_status != 0
- **convergence_rate_percent**: Percentage of calculations that converged successfully

### Detailed Breakdowns
- **exit_status_counts**: Count of calculations by exit status (0=success, others=various errors)
- **calculation_types**: Count of calculations by process type
- **non_converged_details**: List of non-converged calculations with PK, exit status, and type

### Console Output

During execution, the script prints detailed statistics:

```
==================================================
WORKCHAIN STATISTICS:
==================================================
Total PW/ConstrainedPW calculations found: 42
Converged calculations: 25
Non-converged calculations: 17
Convergence rate: 59.5%

Calculation types:
  lordcapulet.constrained_pw: 35
  aiida_quantumespresso.calculations.pw.PwCalculation: 7

Exit status distribution:
  SUCCESS: 25
  ERROR_410: 15
  ERROR_500: 2

Non-converged calculations details:
  PK 7480: exit_status=410, type=lordcapulet.constrained_pw
  PK 8726: exit_status=410, type=lordcapulet.constrained_pw
  ...
==================================================
```

## Data Extracted

For each converged calculation, the following data is extracted:

### Inputs
- **parameters**: QE input parameters (CONTROL, SYSTEM, ELECTRONS, etc.)
- **structure**: Atomic structure information
- **kpoints**: K-point sampling
- **pseudos**: Pseudopotential information
- **oscdft_card**: OSCDFT constraint parameters (for ConstrainedPW)
- **target_matrix**: Target occupation matrix (for ConstrainedPW)

### Outputs
- **output_parameters**: Final energy, forces, stress, convergence info
- **output_atomic_occupations**: Atomic occupation matrices

## Advanced Usage

### Processing Multiple Workchains

```python
workchain_pks = [12345, 12346, 12347]
all_data = {}

for pk in workchain_pks:
    try:
        data = gather_workchain_data(pk, f"workchain_{pk}.json")
        all_data[pk] = data
        print(f"Workchain {pk}: {data['metadata']['total_calculations_found']} calculations")
    except Exception as e:
        print(f"Failed to process workchain {pk}: {e}")
```

### Custom Analysis

```python
data = gather_workchain_data(12345, "data.json")

# Access statistics
stats = data['statistics']
print(f"Convergence rate: {stats['convergence_rate_percent']}%")
print(f"Most common exit status: {max(stats['exit_status_counts'], key=stats['exit_status_counts'].get)}")

# Filter by calculation type
constrained_calcs = {
    pk: calc for pk, calc in data['calculations'].items() 
    if 'constrained' in calc['process_type'].lower()
}

# Extract energies
energies = []
for calc in data['calculations'].values():
    if calc['output_parameters'] and 'energy' in calc['output_parameters']:
        energies.append(calc['output_parameters']['energy'])

print(f"Found {len(energies)} calculations with energy data")
if energies:
    print(f"Energy range: {min(energies):.3f} to {max(energies):.3f} Ry")

# Analyze convergence issues
if stats['non_converged_calculations'] > 0:
    print(f"\nConvergence issues found:")
    print(f"  {stats['non_converged_calculations']} out of {stats['total_pw_calculations']} calculations failed")
    print(f"  Most common error: exit_status={max(stats['exit_status_counts'], key=stats['exit_status_counts'].get)}")
    
    # Print details of failed calculations
    for detail in stats['non_converged_details'][:3]:
        print(f"  Failed calc PK {detail['pk']}: {detail['process_type']} (exit_status={detail['exit_status']})")
```

## Error Handling

The script includes comprehensive error handling:

- **Missing Nodes**: Reports if workchain PK doesn't exist
- **Access Errors**: Handles cases where outputs aren't available
- **Infinite Loops**: Prevents infinite recursion with visited set
- **Depth Limits**: Configurable maximum recursion depth
- **Data Extraction**: Graceful handling of missing or malformed data

## Requirements

- AiiDA framework properly configured
- Access to the AiiDA database containing the workchains
- Python 3.6+ with json, pathlib, warnings modules

## Notes

- Only calculations with `exit_status == 0` are considered converged
- The script automatically detects both standard PW and ConstrainedPW calculations
- Large workchains may take time to process - progress is printed during traversal
- Output files can be large for workchains with many calculations
