#!/usr/bin/env python3
"""
Extract calculations from workchains with the new OccupationMatrixData system.

This module works with the refactored lordcapulet package that uses:
- JsonableData nodes to store OccupationMatrixData objects  
- Direct occupation_matrix output links from calculations
- Unified occupation matrix handling across all workchains

Usage:
    from lordcapulet.utils.postprocessing import gather_workchain_data
    
    # Extract from a specific workchain
    data = gather_workchain_data(workchain_pk=12345)
    
    # Extract with SO(N) decomposition
    data = gather_workchain_data(workchain_pk=12345, perform_so_n=True)
    
    # Save to JSON
    data = gather_workchain_data(workchain_pk=12345, output_filename='results.json')
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import aiida

from aiida.orm import load_node, CalcJobNode, WorkChainNode, JsonableData
from aiida.common.exceptions import NotExistent

from lordcapulet.data_structures.occupation_matrix import OccupationMatrixData
from lordcapulet.utils.so_n_decomposition import (
    get_so_n_lie_basis,
    decompose_rho_and_fix_gauge,
    euler_angles_to_rotation
)

# Optional progress bar
try:
    from alive_progress import alive_bar
    HAS_ALIVE_BAR = True
except ImportError:
    HAS_ALIVE_BAR = False


_VALID_PROPOSAL_MODES = {'random', 'random_so_n', 'gaussian_process', 'gp', 'read'}


def _proposal_mode_to_kind(raw: Optional[str]) -> str:
    """Normalize a raw proposal_mode string to an internal kind label."""
    if raw in ('gp', 'gaussian_process'):
        return 'gp'
    if raw in ('random', 'random_so_n'):
        return 'random'
    return 'plain'


def _constrained_label(kind: str, generation_number: Optional[int]) -> str:
    """Label a constrained_scan calc based on proposal kind + generation."""
    if kind == 'gp':
        if generation_number == 1:
            return 'random proposal'
        return 'GP proposal'
    if kind == 'random':
        return 'random proposal'
    return 'constrained scan'


def _proposal_source_label(proposal_source: Optional[str]) -> Optional[str]:
    """Return a label from explicit proposal metadata, if available."""
    if proposal_source in ('random', 'random_warmup'):
        return 'random proposal'
    if proposal_source == 'random_fallback':
        return 'random fallback'
    if proposal_source == 'gp':
        return 'GP proposal'
    if proposal_source == 'template_product':
        return 'template-product proposal'
    return None


class WorkchainDataExtractor:
    """
    Extract and analyze calculations from AiiDA workchains using the new occupation matrix system.
    
    This extractor works with:
    - GlobalConstrainedSearchWorkChain
    - AFMScanWorkChain  
    - ConstrainedScanWorkChain
    - Individual calculations
    
    Features:
    - Automatic extraction of occupation matrices via output links
    - SO(N) decomposition with gauge fixing
    - Progress tracking and statistics
    - JSON export functionality
    """
    
    def __init__(self,
                 perform_so_n: bool = False,
                 sanity_check_reconstruct: bool = False,
                 debug: bool = False,
                 include_non_converged: bool = False,
                 proposal_mode_override: Optional[str] = None):
        """
        Initialize the extractor.

        Args:
            perform_so_n: Perform SO(N) decomposition on occupation matrices
            sanity_check_reconstruct: Include reconstruction validation in SO(N) results
            debug: Print detailed debug information
            include_non_converged: Include calculations with exit status 410 (non-converged SCF)
            proposal_mode_override: Force the proposal_mode used for convergence-rate
                labeling. If None, the mode is auto-detected from
                workchain.inputs.proposal_mode (only GlobalConstrainedSearchWorkChain
                declares this input). Valid values: 'random', 'random_so_n',
                'gaussian_process', 'gp', 'read'.
        """
        if proposal_mode_override is not None and proposal_mode_override not in _VALID_PROPOSAL_MODES:
            raise ValueError(
                f"proposal_mode_override must be one of {sorted(_VALID_PROPOSAL_MODES)} "
                f"or None, got {proposal_mode_override!r}"
            )

        self.perform_so_n = perform_so_n
        self.sanity_check_reconstruct = sanity_check_reconstruct
        self.debug = debug
        self.include_non_converged = include_non_converged
        self.proposal_mode_override = proposal_mode_override
        self._active_proposal_mode: Optional[str] = None
        self.convergence_stats: Dict[str, Dict[str, float]] = {}

        # Track regularization statistics
        self.regularization_stats = {
            'total_decompositions': 0,
            'regularizations_needed': 0,
            'regularizations_successful': 0,
            'regularizations_failed': 0,
            'details': []  # List of (pk, atom, success) tuples
        }
    
    def extract_from_workchain(self, workchain_pk: int) -> Dict[str, Any]:
        """
        Extract all calculations from a workchain.
        
        Args:
            workchain_pk: Primary key of the workchain
            
        Returns:
            Complete data dictionary with metadata, statistics, and calculations
        """
        # Load and validate workchain
        workchain = load_node(workchain_pk)
        if not isinstance(workchain, WorkChainNode):
            raise ValueError(f"PK {workchain_pk} is not a workchain")
        
        process_type = getattr(workchain, 'process_type', '').lower()
        print(f"Extracting from workchain PK {workchain_pk}")
        print(f"Workchain type: {getattr(workchain, 'process_type', 'unknown')}")

        # Decide which proposal_mode to use for convergence-rate labeling.
        # Override beats auto-detect; auto-detect only finds the input on
        # GlobalConstrainedSearchWorkChain.
        detected = self._detect_proposal_mode(workchain)
        self._active_proposal_mode = self.proposal_mode_override or detected

        # Extract calculations based on workchain type
        if 'globalconstrained' in process_type or 'global_constrained' in process_type:
            calculations = self._extract_from_global_search(workchain)
        elif 'afmscan' in process_type or 'afm_scan' in process_type:
            # AFM scan: generation = None for all calculations
            calculations = self._extract_from_afm_scan(workchain)
            # Create context for generation mapping
            class SimpleContext:
                pass
            self.ctx = SimpleContext()
            self.ctx.generation_map = {calc.pk: None for calc in calculations}
            self.ctx.proposal_source_map = {}
        elif 'constrainedscan' in process_type or 'constrained_scan' in process_type:
            # Single constrained scan: generation = 0 for all calculations
            calculations = self._extract_from_constrained_scan(workchain)
            # Create context for generation mapping
            class SimpleContext:
                pass
            self.ctx = SimpleContext()
            self.ctx.generation_map = {calc.pk: 0 for calc in calculations}
            proposal_source = self._get_extra(workchain, 'proposal_source')
            self.ctx.proposal_source_map = {
                calc.pk: proposal_source for calc in calculations
                if proposal_source is not None
            }
        else:
            raise ValueError(f"Unsupported workchain type: {process_type}")
        
        print(f"Found {len(calculations)} calculations")
        
        # Process the calculations
        calc_data_list = self._process_calculations(calculations, self.ctx.generation_map if hasattr(self, 'ctx') and hasattr(self.ctx, 'generation_map') else {})
        
        # Build output data structure
        return self._build_output_data(
            calc_data_list=calc_data_list,
            workchain_info={
                'pk': workchain_pk,
                'process_type': getattr(workchain, 'process_type', 'unknown'),
                'node_type': type(workchain).__name__
            }
        )
    
    def extract_single_calculation(self, calc_pk: int) -> Dict[str, Any]:
        """
        Extract and analyze a single calculation.
        
        Args:
            calc_pk: Primary key of the calculation
            
        Returns:
            Calculation data dictionary
        """
        calc_node = load_node(calc_pk)
        if not isinstance(calc_node, CalcJobNode):
            raise ValueError(f"PK {calc_pk} is not a calculation")
        
        print(f"Extracting calculation PK {calc_pk}")
        print(f"Calculation type: {getattr(calc_node, 'process_type', 'unknown')}")
        
        # Single calculation extraction: determine generation based on source
        source = self._determine_source(calc_node)
        generation_number = None if source == 'afm_workchain' else 0

        # No parent workchain to read; only the override can supply a mode.
        self._active_proposal_mode = self.proposal_mode_override

        calc_data = self._extract_calc_data(calc_node, generation_number)
        
        if self.perform_so_n:
            self._print_regularization_summary()
        
        return calc_data
    
    def _extract_from_global_search(self, workchain: WorkChainNode) -> List[CalcJobNode]:
        """Extract calculations from GlobalConstrainedSearchWorkChain."""
        calculations = []
        
        # Create a simple context object to store generation mapping
        class SimpleContext:
            pass
        self.ctx = SimpleContext()
        self.ctx.generation_map = {}  # Maps calc_pk -> generation_number
        self.ctx.proposal_source_map = {}
        
        # Try to get generation information from workchain extras or by traversing called workchains
        generation = 1
        for called_wc in workchain.called:
            process_type = getattr(called_wc, 'process_type', '').lower()
            
            if 'afmscan' in process_type or 'afm_scan' in process_type:
                # AFM scans have no generation (generation 0, but we'll use None to distinguish)
                calcs = self._extract_from_afm_scan(called_wc)
                for calc in calcs:
                    self.ctx.generation_map[calc.pk] = None
                calculations.extend(calcs)
            elif 'constrainedscan' in process_type or 'constrained_scan' in process_type:
                # Constrained scans: increment generation for each scan workchain
                calcs = self._extract_from_constrained_scan(called_wc)
                proposal_source = self._get_extra(called_wc, 'proposal_source')
                for calc in calcs:
                    self.ctx.generation_map[calc.pk] = generation
                    if proposal_source is not None:
                        self.ctx.proposal_source_map[calc.pk] = proposal_source
                calculations.extend(calcs)
                generation += 1
        
        return calculations
    
    def _extract_from_afm_scan(self, workchain: WorkChainNode) -> List[CalcJobNode]:
        """Extract calculations from AFMScanWorkChain."""
        calculations = []
        
        try:
            # Get calculation PKs from output
            if 'all_calculation_pks' in workchain.outputs:
                calc_pks = workchain.outputs.all_calculation_pks.get_list()
                for pk in calc_pks:
                    calc = load_node(pk)
                    if self._should_include_calculation(calc):
                        calculations.append(calc)
            else:
                # Fallback: traverse called nodes
                for called in workchain.called:
                    if isinstance(called, CalcJobNode) and self._should_include_calculation(called):
                        calculations.append(called)
        except Exception as e:
            if self.debug:
                print(f"Error extracting AFM calculations: {e}")
        
        return calculations
    
    def _extract_from_constrained_scan(self, workchain: WorkChainNode) -> List[CalcJobNode]:
        """Extract calculations from ConstrainedScanWorkChain."""
        calculations = []
        
        try:
            # Get calculation PKs from output
            if 'all_calculation_pks' in workchain.outputs:
                calc_pks = workchain.outputs.all_calculation_pks.get_list()
                for pk in calc_pks:
                    calc = load_node(pk)
                    if self._should_include_calculation(calc):
                        calculations.append(calc)
            else:
                # Fallback: traverse called nodes
                for called in workchain.called:
                    if isinstance(called, CalcJobNode) and self._should_include_calculation(called):
                        calculations.append(called)
        except Exception as e:
            if self.debug:
                print(f"Error extracting constrained calculations: {e}")
        
        return calculations
    
    def _should_include_calculation(self, calc: CalcJobNode) -> bool:
        """Determine if a calculation should be included in extraction."""
        if calc.exit_status == 0:
            return True
        if self.include_non_converged and calc.exit_status == 410:
            return True
        return False
    
    def _process_calculations(self, calculations: List[CalcJobNode], generation_map: Dict[int, int] = None) -> List[Dict[str, Any]]:
        """Process a list of calculations and extract data.
        
        Args:
            calculations: List of calculation nodes
            generation_map: Optional mapping of calc_pk -> generation_number
        """
        # Reset regularization stats
        self.regularization_stats = {
            'total_decompositions': 0,
            'regularizations_needed': 0,
            'regularizations_successful': 0,
            'regularizations_failed': 0,
            'details': []
        }
        
        if generation_map is None:
            generation_map = {}
        proposal_source_map = (
            self.ctx.proposal_source_map
            if hasattr(self, 'ctx') and hasattr(self.ctx, 'proposal_source_map')
            else {}
        )
        
        results = []
        
        if HAS_ALIVE_BAR and len(calculations) > 0:
            with alive_bar(len(calculations), title="Processing calculations") as bar:
                for calc in calculations:
                    try:
                        generation_number = generation_map.get(calc.pk, 0)
                        proposal_source = proposal_source_map.get(calc.pk)
                        calc_data = self._extract_calc_data(
                            calc, generation_number, proposal_source
                        )
                        if calc_data:
                            results.append(calc_data)
                    except Exception as e:
                        if self.debug:
                            print(f"Error processing calculation {calc.pk}: {e}")
                    bar()
        else:
            print(f"Processing {len(calculations)} calculations...")
            for i, calc in enumerate(calculations):
                try:
                    generation_number = generation_map.get(calc.pk, 0)
                    proposal_source = proposal_source_map.get(calc.pk)
                    calc_data = self._extract_calc_data(
                        calc, generation_number, proposal_source
                    )
                    if calc_data:
                        results.append(calc_data)
                except Exception as e:
                    if self.debug:
                        print(f"Error processing calculation {calc.pk}: {e}")
                
                if not self.debug and i % max(1, len(calculations) // 10) == 0:
                    progress = (i + 1) / len(calculations) * 100
                    print(f"Progress: {progress:.1f}% ({i + 1}/{len(calculations)})")
        
        if self.perform_so_n:
            self._print_regularization_summary()

        self.convergence_stats = self._convergence_rates_by_source(
            results, self._active_proposal_mode
        )
        self._print_convergence_summary()

        return results

    def _extract_calc_data(
        self,
        calc: CalcJobNode,
        generation_number: int = None,
        proposal_source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Extract data from a single calculation.
        
        Args:
            calc: Calculation node
            generation_number: Generation number for constrained calculations (None for AFM, 0 for single scan)
        """
        calc_data = {
            'pk': calc.pk,
            'exit_status': calc.exit_status,
            'generation_number': generation_number,
            'proposal_source': proposal_source,
            'converged': calc.exit_status == 0,
            'process_type': getattr(calc, 'process_type', 'unknown'),
            'calculation_source': self._determine_source(calc),
            'output_parameters': None,
            'occupation_matrices': None
        }
        
        # Extract output parameters
        try:
            if 'output_parameters' in calc.outputs:
                params = calc.outputs.output_parameters
                if hasattr(params, 'get_dict'):
                    calc_data['output_parameters'] = params.get_dict()
        except Exception as e:
            if self.debug:
                print(f"Error extracting output_parameters from {calc.pk}: {e}")
        
        # Extract occupation matrices using new system
        # The new workchains store occupation_matrix_pk in the calculation extras
        try:
            # Check if occupation matrix PK is stored in extras
            if 'occupation_matrix_pk' in calc.base.extras:
                occ_matrix_pk = calc.base.extras.get('occupation_matrix_pk')
                try:
                    occ_node = load_node(occ_matrix_pk)
                    if isinstance(occ_node, JsonableData):
                        occ_data = occ_node.obj
                        if isinstance(occ_data, OccupationMatrixData):
                            calc_data['occupation_matrices'] = occ_data.as_dict()
                            calc_data['occupation_matrix_pk'] = occ_matrix_pk
                            
                            if self.debug:
                                source_desc = "AFM" if calc_data['calculation_source'] == 'afm_workchain' else "constrained DFT"
                                print(f"Analyzed {source_desc} calculation (PK: {calc.pk}), obtained matrix (PK: {occ_matrix_pk})")
                            
                            # Perform SO(N) decomposition if requested
                            if self.perform_so_n:
                                so_n_results = self._perform_so_n_decomposition(occ_data, calc.pk)
                                calc_data['so_n_decomposition'] = so_n_results
                        else:
                            if self.debug:
                                print(f"Warning: occupation_matrix_pk points to non-OccupationMatrixData for calculation {calc.pk}")
                            calc_data['occupation_matrices'] = None
                    else:
                        if self.debug:
                            print(f"Warning: occupation_matrix_pk points to non-JsonableData for calculation {calc.pk}")
                        calc_data['occupation_matrices'] = None
                except Exception as e:
                    if self.debug:
                        print(f"Error loading occupation matrix (PK: {occ_matrix_pk}) for calculation {calc.pk}: {e}")
                    calc_data['occupation_matrices'] = None
            else:
                # No occupation_matrix_pk in extras (old calculation or failed extraction)
                calc_data['occupation_matrices'] = None
                if self.debug:
                    source_desc = "AFM" if calc_data['calculation_source'] == 'afm_workchain' else "constrained DFT"
                    print(f"Analyzed {source_desc} calculation (PK: {calc.pk}), no output matrix present (exit status: {calc.exit_status})")
        except Exception as e:
            if self.debug:
                print(f"Error extracting occupation matrices from {calc.pk}: {e}")
            calc_data['occupation_matrices'] = None
        
        return calc_data
    
    def _determine_source(self, calc: CalcJobNode) -> str:
        """Determine the source of a calculation."""
        process_type = getattr(calc, 'process_type', '').lower()

        if 'lordcapulet.constrained_pw' in process_type or 'constrainedpw' in process_type:
            return 'constrained_scan'
        elif 'quantumespresso.pw' in process_type:
            return 'afm_workchain'
        else:
            return 'unknown'

    def _detect_proposal_mode(self, workchain: WorkChainNode) -> Optional[str]:
        """Read proposal_mode from a workchain's inputs, if declared.

        Only GlobalConstrainedSearchWorkChain declares this input
        (lordcapulet/workflows/global_constrained_search.py). Returns None
        for AFMScanWorkChain / standalone ConstrainedScanWorkChain.
        """
        try:
            return workchain.inputs.proposal_mode.value
        except (AttributeError, KeyError, NotExistent):
            return None

    @staticmethod
    def _get_extra(node, key: str):
        """Read an AiiDA extra, returning None when it is absent."""
        try:
            if key in node.base.extras:
                return node.base.extras.get(key)
        except AttributeError:
            return None
        return None

    def _convergence_rates_by_source(
        self,
        calc_data_list: List[Dict[str, Any]],
        proposal_mode: Optional[str],
    ) -> Dict[str, Dict[str, float]]:
        """Group calc records by proposal-source label, count converge rate.

        Returns:
            { label: {'converged': int, 'total': int, 'rate': float} }
            rate in [0, 1]; 0.0 when total == 0.

        proposal_mode is the raw mode string (or None). It is normalized
        via _proposal_mode_to_kind before driving the constrained-calc
        labeling rule in _constrained_label.

        Meaningful only when include_non_converged=True; otherwise
        non-converged calcs are dropped upstream and every rate is 1.0.
        """
        kind = _proposal_mode_to_kind(proposal_mode)
        stats: Dict[str, Dict[str, float]] = {}
        for calc in calc_data_list:
            source = calc.get('calculation_source', 'unknown')
            if source == 'afm_workchain':
                label = 'standard magnetic scan'
            elif source == 'constrained_scan':
                label = (
                    _proposal_source_label(calc.get('proposal_source'))
                    or _constrained_label(kind, calc.get('generation_number'))
                )
            else:
                label = source
            bucket = stats.setdefault(label, {'converged': 0, 'total': 0, 'rate': 0.0})
            bucket['total'] += 1
            if calc.get('converged'):
                bucket['converged'] += 1
        for bucket in stats.values():
            total = bucket['total']
            bucket['rate'] = (bucket['converged'] / total) if total else 0.0
        return stats

    def _print_convergence_summary(self) -> None:
        """Print per-source convergence rates (mirrors regularization summary)."""
        if not self.convergence_stats:
            return
        print("\nConvergence Rate by Source:")
        for label, s in self.convergence_stats.items():
            pct = s['rate'] * 100.0
            print(f"  {label:<26} {int(s['converged']):>5} / {int(s['total']):<5}  {pct:6.2f}%")

    def _perform_so_n_decomposition(self, occ_data: OccupationMatrixData, calc_pk: int) -> Dict[str, Any]:
        """Perform SO(N) decomposition on occupation matrices."""
        results = {
            'decomposition_successful': False
        }
        
        sanity_check_data = {}
        
        try:
            for atom_label in occ_data.get_atom_labels():
                atom_results = {
                    'specie': occ_data[atom_label]['specie'],
                    'shell': occ_data[atom_label]['shell'],
                    'occupation_matrix_decomposition': {}
                }
                
                atom_sanity_check = {}
                
                for spin in ['up', 'down']:
                    try:
                        occ_matrix = np.array(occ_data.get_occupation_matrix(atom_label, spin))
                        
                        if occ_matrix.shape[0] != occ_matrix.shape[1]:
                            continue
                        
                        dim = occ_matrix.shape[0]
                        generators = get_so_n_lie_basis(dim)
                        
                        # Update statistics
                        self.regularization_stats['total_decompositions'] += 1
                        
                        # Decompose with gauge fixing
                        eigenvalues, eigenvectors, euler_angles, need_reg, degenerate_groups = \
                            decompose_rho_and_fix_gauge(occ_matrix, generators)
                        
                        if need_reg:
                            self.regularization_stats['regularizations_needed'] += 1
                            self.regularization_stats['regularizations_successful'] += 1
                            self.regularization_stats['details'].append((calc_pk, f"{atom_label}_{spin}", True))
                        
                        result_data = {
                            'eigenvalues': eigenvalues.tolist(),
                            'euler_angles': euler_angles.tolist(),
                            'matrix_dimension': dim,
                            'trace': float(np.trace(occ_matrix)),
                            'need_regularization': bool(need_reg),
                            'degenerate_groups': degenerate_groups
                        }
                        
                        atom_results['occupation_matrix_decomposition'][spin] = result_data
                        
                        # Add reconstruction for sanity check if requested
                        if self.sanity_check_reconstruct:
                            R_reconstructed = euler_angles_to_rotation(euler_angles, generators)
                            rho_reconstructed = R_reconstructed @ np.diag(eigenvalues) @ R_reconstructed.T
                            reconstruction_error = float(np.max(np.abs(rho_reconstructed - occ_matrix)))
                            
                            atom_sanity_check[spin] = rho_reconstructed.tolist()
                            
                            # Store reconstruction error (will be added once per atom)
                            if 'reconstruction_error' not in atom_sanity_check:
                                atom_sanity_check['reconstruction_error'] = {}
                            atom_sanity_check['reconstruction_error'][spin] = reconstruction_error
                    
                    except Exception as e:
                        if self.debug:
                            print(f"SO(N) decomposition failed for {atom_label}_{spin}: {e}")
                        atom_results['occupation_matrix_decomposition'][spin] = {'error': str(e)}
                
                if atom_results['occupation_matrix_decomposition']:
                    results[atom_label] = atom_results
                    
                    # Store sanity check data for this atom if we have any
                    if atom_sanity_check:
                        sanity_check_data[atom_label] = {
                            'reconstructed_occupation_matrix': {k: v for k, v in atom_sanity_check.items() if k != 'reconstruction_error'},
                            'reconstruction_error': atom_sanity_check.get('reconstruction_error', {})
                        }
            
            # Add sanity check section if we have any data
            if sanity_check_data and self.sanity_check_reconstruct:
                results['sanity_check'] = sanity_check_data
            
            # Mark as successful if we processed at least one atom (excluding metadata keys)
            processed_atoms = [k for k in results.keys() if k not in ['decomposition_successful', 'sanity_check']]
            if processed_atoms:
                results['decomposition_successful'] = True
        
        except Exception as e:
            if self.debug:
                print(f"SO(N) decomposition error: {e}")
        
        return results
    
    def _build_output_data(self, calc_data_list: List[Dict[str, Any]], 
                          workchain_info: Dict[str, Any]) -> Dict[str, Any]:
        """Build the complete output data structure."""
        # Compute statistics
        total_calcs = len(calc_data_list)
        converged_calcs = sum(1 for c in calc_data_list if c['converged'])
        non_converged = total_calcs - converged_calcs
        
        source_counts = {}
        for calc_data in calc_data_list:
            source = calc_data['calculation_source']
            source_counts[source] = source_counts.get(source, 0) + 1
        
        statistics = {
            'total_calculations': total_calcs,
            'converged_calculations': converged_calcs,
            'non_converged_calculations': non_converged,
            'calculation_sources': source_counts,
            'proposal_mode': self._active_proposal_mode,
            'convergence_rates_by_source': self.convergence_stats,
        }
        
        metadata = {
            'total_calculations_found': total_calcs,
            'extraction_timestamp': datetime.now().isoformat(),
            **workchain_info
        }
        try:
            metadata['profile'] = aiida.get_profile().name
        except Exception:
            pass
        
        output_data = {
            'metadata': metadata,
            'statistics': statistics,
            'calculations': {}
        }
        
        # Store calculations with PK as key
        for calc_data in calc_data_list:
            pk = calc_data['pk']
            output_data['calculations'][str(pk)] = calc_data
        
        return output_data
    
    def _print_regularization_summary(self):
        """Print regularization statistics."""
        stats = self.regularization_stats
        if stats['total_decompositions'] == 0:
            return
        
        print(f"\nSO(N) Decomposition Summary:")
        print(f"  Total decompositions: {stats['total_decompositions']}")
        print(f"  Regularizations needed: {stats['regularizations_needed']}")
        
        if stats['regularizations_needed'] > 0:
            success_rate = stats['regularizations_successful'] / stats['regularizations_needed'] * 100
            print(f"  Regularizations successful: {stats['regularizations_successful']} ({success_rate:.1f}%)")
            if stats['regularizations_failed'] > 0:
                print(f"  Regularizations failed: {stats['regularizations_failed']}")
    
    def save_to_json(self, data: Dict[str, Any], filename: str):
        """Save extracted data to JSON file."""
        output_path = Path(filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Data saved to: {output_path.absolute()}")


# Convenience function for simple usage
def gather_workchain_data(workchain_pk: int,
                         output_filename: Optional[str] = None,
                         perform_so_n: bool = False,
                         sanity_check_reconstruct: bool = False,
                         debug: bool = False,
                         include_non_converged: bool = False,
                         proposal_mode_override: Optional[str] = None) -> Dict[str, Any]:
    """
    Extract data from a workchain.

    Args:
        workchain_pk: Primary key of the workchain
        output_filename: Optional JSON output filename
        perform_so_n: Perform SO(N) decomposition
        sanity_check_reconstruct: Include reconstruction validation
        debug: Print detailed debug information
        include_non_converged: Include non-converged calculations (exit 410)
        proposal_mode_override: Force proposal_mode for convergence-rate labeling;
            auto-detected from workchain inputs if None.

    Returns:
        Complete data dictionary
    """
    extractor = WorkchainDataExtractor(
        perform_so_n=perform_so_n,
        sanity_check_reconstruct=sanity_check_reconstruct,
        debug=debug,
        include_non_converged=include_non_converged,
        proposal_mode_override=proposal_mode_override,
    )
    
    data = extractor.extract_from_workchain(workchain_pk)
    
    if output_filename:
        extractor.save_to_json(data, output_filename)
    
    # Print summary
    print(f"\nExtraction Summary:")
    print(f"Total calculations: {data['statistics']['total_calculations']}")
    print(f"Converged: {data['statistics']['converged_calculations']}")
    if include_non_converged:
        print(f"Non-converged: {data['statistics']['non_converged_calculations']}")
    print(f"Sources: {data['statistics']['calculation_sources']}")
    
    return data
