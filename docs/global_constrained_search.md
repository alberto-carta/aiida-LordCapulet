# GlobalConstrainedSearchWorkChain

The `GlobalConstrainedSearchWorkChain` is a high-level workflow that orchestrates the entire constrained DFT+U search process. It combines AFM search, constrained calculations, and iterative matrix proposals into a single automated workflow.

## Overview

This workchain implements the following algorithm:

1. **Initial AFM Search**: Performs an antiferromagnetic search to get initial occupation matrices
2. **Iterative Constrained Search**: 
   - Proposes N new matrices from existing results
   - Runs constrained DFT calculations on these matrices
   - Repeats until Nmax total calculations are performed

## Workflow Diagram

```
Start → AFM Search → Propose Matrices → Constrained Scan (N proposals) 
          ↑                              ↓
          └─── Propose Matrices ←── [N_cumulative < Nmax?] 
                    ↓                     ↓ (No)
               Constrained Scan     Final Results
```

## Input Parameters

### Required Inputs

The workchain uses `expose_inputs` to inherit all inputs from both `AFMScanWorkChain` and `ConstrainedScanWorkChain`:

#### AFM Namespace (`afm`)
- `structure`: Structure to analyze (StructureData or HubbardStructureData)
- `parameters`: DFT parameters (Dict)
- `kpoints`: K-points sampling (KpointsData)
- `code`: Code for pw.x calculations (Code)
- `tm_atoms`: List of transition metal atom labels (List)
- `magnitude`: Magnetization magnitude for AFM (Float, default=0.5)

#### Constrained Namespace (`constrained`)
- `structure`: Structure (same as AFM)
- `parameters`: DFT parameters (same as AFM)
- `kpoints`: K-points (same as AFM)
- `code`: Code (same as AFM)
- `tm_atoms`: TM atoms (same as AFM)
- `oscdft_card`: OSCDFT parameters (Dict)

#### Global Search Parameters
- `Nmax`: Total maximum number of constrained proposals (Int)
- `N`: Number of proposals per generation/batch (Int)

### Optional Inputs

- `proposal_mode`: Mode for matrix proposal ('random', 'read', etc.) (Str, default='random')
- `proposal_debug`: Enable debug output for proposals (Bool, default=False)
- `proposal_holistic`: Use holistic approach for proposals - analyze all previous result matrices instead of just the last generation (Bool, default=False)
- `proposal_kwargs`: Additional keyword arguments for proposal function (Dict)

## Outputs

- `all_afm_matrices`: List of PKs from initial AFM search occupation matrices
- `all_constrained_matrices`: List of all occupation matrix PKs from constrained calculations
- `all_calculation_pks`: List of all calculation PKs performed
- `generation_summary`: Dict with summary of each generation's results

## Generation Summary Format

The `generation_summary` output contains a dictionary with the following structure:

```python
{
    0: {  # Generation 0 = AFM search
        'type': 'afm',
        'n_calculations': 4,
        'matrix_pks': [123, 124, 125, 126]
    },
    1: {  # Generation 1 = First constrained batch
        'type': 'constrained', 
        'n_calculations': 8,
        'n_successful': 6,
        'n_failed': 2,
        'matrix_pks': [127, 128, ..., -1, -1],  # -1 indicates failed calculations
        'calculation_pks': [135, 136, ...]
    },
    # ... more generations
}
```

## Advanced Usage

### Using Holistic Proposal Mode

By default, the workchain uses a Markovian approach where each generation proposes new matrices based only on the results from the previous generation. You can enable holistic mode to use all successful results from all previous generations:

```python
inputs['proposal_holistic'] = Bool(True)
```

**Markovian mode** (default): Generation N+1 proposals are based only on successful results from generation N.

**Holistic mode**: Generation N+1 proposals are based on all successful results from generations 0 through N.

### Using Read Mode for Proposals

If you have existing occupation matrix data, you can use 'read' mode:

```python
inputs['proposal_mode'] = Str('read')
inputs['proposal_kwargs'] = Dict(dict={
    'readfile': 'your_data_file.json'
})
```

### Monitoring Progress

```bash
# Check workchain status
verdi process status <PK>

# Follow logs in real time
verdi process report <PK> --watch

# Check individual calculation status
verdi process list -a  # Show all processes
```

## Error Handling

The workchain defines several exit codes:

- **400**: `ERROR_AFM_SEARCH_FAILED` - Initial AFM search failed completely
- **401**: `ERROR_CONSTRAINED_SCAN_FAILED` - All calculations in a constrained generation failed
- **402**: `ERROR_PROPOSAL_FAILED` - Matrix proposal step failed

**Note**: The workchain only fails if ALL calculations in a generation fail. Partial failures (some calculations succeed, some fail) are handled gracefully and logged in the generation summary.

## Performance Considerations

- **Batch Size (N)**: Larger batches are more efficient but use more resources simultaneously
- **Total Size (Nmax)**: Consider computational cost vs. search thoroughness
- **Proposal Mode**: 'random' mode is more exploratory, 'read' mode uses existing data
- **Holistic vs Markovian**: Holistic mode may find better proposals by using all data, but may also get stuck in local minima

## Implementation Details

### Workflow Control

The workchain uses AiiDA's `while_` construct to iterate until `N_cumulative >= Nmax`. Each iteration:

1. Proposes N new matrices from all existing results
2. Runs constrained calculations on these matrices
3. Updates counters and storage
4. Checks if more iterations are needed

### Data Management

- All occupation matrices are stored as PKs for efficient data handling
- Results from each generation are tracked separately with success/failure counts
- Cumulative lists maintain complete search history
- Failed calculations are marked with PK = -1 and excluded from future proposals
- Holistic mode maintains a separate list of only successful result matrices for proposals

### Integration with Existing Functions

The workchain integrates seamlessly with:
- `aiida_propose_occ_matrices_from_results()` for matrix proposals
- Existing AFM and constrained scan workchains
- Standard AiiDA data provenance and caching
