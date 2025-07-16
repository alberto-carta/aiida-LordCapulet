# LordCapulet


```Provides AiiDA calculation and workflow plugins for running constrained DFT+U calculations, including:

- **ConstrainedPWCalculation**: A custom PW calculation that handles OSCDFT constraints
- **AFMScanWorkChain**: Workflow for scanning different antiferromagnetic configurations
- **ConstrainedScanWorkChain**: Workflow for running multiple constrained calculations with different target occupation matrices

![LordCapulet](LordCapulet.png)



#### Package Structure

```
lorcapulet/                          # Main package directory
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
    └── __init__.py                 # Module initialization


## Installation

From the plugin root directory:
```bash
pip install -e .
```

## Package Structure

```
lorcapulet
├── calculations
│   ├── constrained_pw.py
│   └── __init__.py
├── functions
│   ├── __init__.py
│   └── propose.py
├── __init__.py
├── utils
│   └── __init__.py
└── workflows
    ├── afm_scan.py
    ├── constrained_scan.py
    └── __init__.py

```

## Usage

### Direct Import
```python
from lordcapulet import ConstrainedPWCalculation, AFMScanWorkChain, ConstrainedScanWorkChain

# Or specific module imports
from lordcapulet.workflows import AFMScanWorkChain, ConstrainedScanWorkChain
from lordcapulet.calculations import ConstrainedPWCalculation
from lordcapulet.functions import aiida_propose_occ_matrices_from_results
```

### Using AiiDA Entry Points
```python
from aiida.plugins import WorkflowFactory, CalculationFactory

AFMScanWorkChain = WorkflowFactory('lordcapulet.afm_scan')
ConstrainedScanWorkChain = WorkflowFactory('lordcapulet.constrained_scan')
ConstrainedPWCalculation = CalculationFactory('lordcapulet.constrained_pw')
```

### Running a Constrained Scan
```python
from aiida.engine import submit
from lordcapulet.workflows import ConstrainedScanWorkChain

inputs = {
    'structure': your_structure,
    'parameters': pw_parameters,
    'kpoints': kpoints,
    'code': code,
    'tm_atoms': List(list=tm_atoms),
    'oscdft_card': oscdft_parameters,
    'occupation_matrices_list': List(list=target_matrices),
}

workchain = submit(ConstrainedScanWorkChain, **inputs)
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

## Requirements

- Python >= 3.8
- AiiDA >= 2.0.0
- aiida-quantumespresso >= 4.0.0
- numpy
