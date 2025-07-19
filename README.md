# LordCapulet

<img src="LordCapulet.png" alt="LordCapulet" width="400">

**LordCapulet** is an AiiDA plugin that provides automated workflows for constrained DFT+U calculations using OSCDFT (Oxidation-State Constrained DFT). The main feature is the **GlobalConstrainedSearchWorkChain**, which performs iterative searches through occupation matrix space to find optimal electronic configurations.

## Key Features

### GlobalConstrainedSearchWorkChain
The flagship workflow that orchestrates an automated search process:

1. **Initial AFM Search**: Performs antiferromagnetic calculations to generate initial occupation matrices
2. **Iterative Constrained Search**: Intelligently proposes new occupation matrices based on previous results
3. **Batch Processing**: Runs N proposals per generation until Nmax total calculations are completed
4. **Adaptive Sampling**: Uses both Markovian (generation-to-generation) and holistic (all-history) proposal modes

This workflow enables systematic exploration of electronic ground states, automatically discovering multiple magnetic local minima.

### Additional Components

- **ConstrainedPWCalculation**: A custom PW calculation that handles OSCDFT constraints
- **AFMScanWorkChain**: Workflow for scanning different ferro-antiferromagnetic configurations
- **ConstrainedScanWorkChain**: Workflow for running multiple constrained calculations with different target occupation matrices

## Installation

```bash
pip install -e .
```

## Usage

### Direct Import
```python
from lordcapulet import ConstrainedPWCalculation, AFMScanWorkChain, ConstrainedScanWorkChain, GlobalConstrainedSearchWorkChain
```

### Using AiiDA Entry Points
```python
from aiida.plugins import WorkflowFactory, CalculationFactory

GlobalConstrainedSearchWorkChain = WorkflowFactory('lordcapulet.global_constrained_search')
ConstrainedPWCalculation = CalculationFactory('lordcapulet.constrained_pw')
```

### Running a Global Constrained Search
```python
from aiida.engine import submit
from lordcapulet.workflows import GlobalConstrainedSearchWorkChain

inputs = {
    'afm': {
        'structure': your_structure,
        'parameters': pw_parameters,
        'kpoints': kpoints,
        'code': code,
        'tm_atoms': List(list=tm_atoms),
    },
    'constrained': {
        'structure': your_structure,
        'parameters': pw_parameters,
        'kpoints': kpoints,
        'code': code,
        'tm_atoms': List(list=tm_atoms),
        'oscdft_card': oscdft_parameters,
    },
    'Nmax': Int(100),  # Total number of proposals to evaluate
    'N': Int(10),      # Number of proposals per generation
}

workchain = submit(GlobalConstrainedSearchWorkChain, **inputs)
```

## Verification

Check that the plugins are properly registered:
```bash
verdi plugin list aiida.calculations | grep lordcapulet
verdi plugin list aiida.workflows | grep lordcapulet
```

After installation, restart the AiiDA daemon:
```bash
verdi daemon restart
```

## Testing

Run the test suite to verify functionality:

```bash
# Simple test runner (no dependencies)
python run_tests.py

# With pytest (if installed)
pytest tests/
```

See `tests/README.md` for detailed testing information.

## Requirements

- Python >= 3.8
- AiiDA >= 2.0.0
- aiida-quantumespresso >= 4.0.0
- numpy

## Package Structure

```
lordcapulet/                         # Main package directory
├── calculations/                    # AiiDA calculation plugins
│   ├── constrained_pw.py           # Custom PW calculation with OSCDFT constraints
│   └── __init__.py                 # Module initialization
├── functions/                       # AiiDA calcfunction plugins
│   ├── __init__.py                 # Module initialization
│   └── propose.py                  # Functions for proposing occupation matrices
├── __init__.py                     # Package initialization with convenient imports
├── utils/                          # Utility functions and helpers
│   └── __init__.py                 # Module initialization
└── workflows/                       # AiiDA workflow plugins
    ├── afm_scan.py                 # Antiferromagnetic configuration scanner
    ├── constrained_scan.py         # Multiple constrained calculations workflow
    ├── global_constrained_search.py # Global automated search workflow (main feature)
    └── __init__.py                 # Module initialization
```
