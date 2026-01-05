#!/usr/bin/env python3
"""
Search Simulator for DFT+U Occupation Matrix Discovery

This module simulates the iterative search process for discovering DFT+U occupation
matrix configurations. It uses a ground truth DataBank as an oracle and evaluates
proposed configurations based on Euclidean distance matching.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from copy import deepcopy

from lordcapulet.data_structures import DataBank, OccupationMatrixData
from lordcapulet.functions.propose import propose_new_constraints


class SearchSimulator:
    """
    Simulator for iterative occupation matrix discovery process.
    
    The simulator:
    1. Uses a ground truth DataBank as the complete search space
    2. Starts with M randomly selected matrices
    3. Iteratively proposes new matrices using proposal functions
    4. Evaluates proposals by finding nearest neighbors in ground truth
    5. Tracks discovered states and failed proposals
    
    Attributes:
        ground_truth: Complete DataBank used as oracle
        results: DataBank containing discovered states with metadata
        failed_proposals: List of proposals that exceeded distance threshold
        current_pool: Current set of matrices used for next proposals
        distance_threshold: Maximum distance to consider a match successful
    """
    
    def __init__(
        self, 
        ground_truth_databank: DataBank,
        distance_threshold: float = 0.1,
        atom_ids: Optional[List[str]] = None,
        spins: Optional[List[str]] = None
    ):
        """
        Initialize the search simulator.
        
        Args:
            ground_truth_databank: DataBank with all converged calculations
            distance_threshold: Maximum distance for successful discovery
            atom_ids: Atom IDs to use for distance calculations (default: all)
            spins: Spin channels to use for distance calculations (default: ['up', 'down'])
        """
        self.ground_truth = ground_truth_databank
        self.distance_threshold = distance_threshold
        self.atom_ids = atom_ids if atom_ids is not None else ground_truth_databank.atom_ids
        self.spins = spins if spins is not None else ['up', 'down']
        
        # Initialize results storage
        self.results = DataBank()
        self.failed_proposals: List[Dict[str, Any]] = []
        
        # Track discovered PKs to avoid re-discovery
        self.discovered_pks = set()
        
        # Current pool for proposals
        self.current_pool: List[OccupationMatrixData] = []
        
        # Statistics
        self.iteration_count = 0
        self.total_proposals = 0
        self.total_discoveries = 0
        self.total_failures = 0
    
    def initialize_from_random(self, M: int, seed: Optional[int] = None) -> None:
        """
        Initialize search with M randomly selected matrices from ground truth.
        
        Args:
            M: Number of initial matrices to select
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
        
        if M > len(self.ground_truth):
            raise ValueError(f"M={M} exceeds ground truth size ({len(self.ground_truth)})")
        
        # Randomly select M indices
        indices = np.random.choice(len(self.ground_truth), size=M, replace=False)
        
        # Initialize results with these matrices
        for idx in indices:
            record = self.ground_truth.get_record(idx)
            self._add_to_results(
                occ_data=record['occ_data'],
                energy=record['energy'],
                pk=record.get('pk', None),
                iteration=0,
                source_pks=[],
                nearest_distance=0.0,
                discovery_status='initial'
            )
            
            # Add to current pool
            self.current_pool.append(record['occ_data'])
            
            # Mark as discovered
            if 'pk' in record:
                self.discovered_pks.add(record['pk'])
        
        print(f"Initialized with {M} random matrices from ground truth")
        print(f"Initial pool energies: {[r['energy'] for r in self.results._records]}")
    
    def generate_proposals(
        self,
        N: int,
        proposal_mode: str = 'random',
        debug: bool = False,
        **proposal_kwargs
    ) -> List[OccupationMatrixData]:
        """
        Generate N new occupation matrix proposals from current pool.
        
        This method only generates proposals without evaluating them.
        Use evaluate_proposals() to check them against ground truth.
        
        Args:
            N: Number of proposals to generate
            proposal_mode: Proposal generation mode ('random', 'random_so_n', etc.)
            debug: Whether to print debug information
            **proposal_kwargs: Additional arguments for proposal function
        
        Returns:
            List of OccupationMatrixData proposals
        """
        if debug:
            print(f"\n{'='*60}")
            print(f"Generating {N} proposals")
            print(f"Current pool size: {len(self.current_pool)}")
            print(f"{'='*60}")
        
        # Generate proposals using current pool
        proposals = propose_new_constraints(
            occ_matr_list=self.current_pool,
            N=N,
            mode=proposal_mode,
            debug=debug,
            **proposal_kwargs
        )
        
        if debug:
            print(f"\nGenerated {len(proposals)} proposals")
        
        return proposals
    
    def evaluate_proposals(
        self,
        proposals: List[OccupationMatrixData],
        debug: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate proposals against ground truth and update state.
        
        For each proposal:
        - Find nearest neighbor in ground truth
        - If distance < threshold: add to discoveries
        - If distance >= threshold: mark as failed
        - Update current pool with new discoveries
        
        Args:
            proposals: List of OccupationMatrixData proposals to evaluate
            debug: Whether to print debug information
        
        Returns:
            Dictionary with evaluation statistics
        """
        self.iteration_count += 1
        iteration_discoveries = 0
        iteration_failures = 0
        N = len(proposals)
        
        if debug:
            print(f"\n{'='*60}")
            print(f"Iteration {self.iteration_count}: Evaluating {N} proposals")
            print(f"{'='*60}")
        
        self.total_proposals += N
        
        self.total_proposals += N
        
        # Evaluate each proposal
        new_discoveries = []
        for i, proposal in enumerate(proposals):
            if debug:
                print(f"\nEvaluating proposal {i+1}/{N}...")
            
            # Find nearest neighbor in ground truth
            distances = self.ground_truth.compute_distances(
                reference=proposal,
                atom_label=None,  # Total distance across all atoms
                spins=self.spins
            )
            
            nearest_idx = distances.argmin()
            nearest_distance = distances[nearest_idx]
            
            if debug:
                print(f"  Nearest distance: {nearest_distance:.6f} (threshold: {self.distance_threshold})")
            
            # Check if discovery is successful
            if nearest_distance < self.distance_threshold:
                nearest_record = self.ground_truth.get_record(nearest_idx)
                nearest_pk = nearest_record.get('pk', None)
                
                # Check if already discovered
                if nearest_pk is not None and nearest_pk in self.discovered_pks:
                    if debug:
                        print(f"  Status: Re-discovery (PK={nearest_pk}) - skipping")
                    continue
                
                # New discovery!
                if debug:
                    print(f"  Status: SUCCESS - New discovery (PK={nearest_pk}, E={nearest_record['energy']:.4f} eV)")
                
                self._add_to_results(
                    occ_data=nearest_record['occ_data'],
                    energy=nearest_record['energy'],
                    pk=nearest_pk,
                    iteration=self.iteration_count,
                    source_pks=self._get_current_pool_pks(),
                    nearest_distance=nearest_distance,
                    discovery_status='discovered'
                )
                
                new_discoveries.append(nearest_record['occ_data'])
                
                if nearest_pk is not None:
                    self.discovered_pks.add(nearest_pk)
                
                iteration_discoveries += 1
                self.total_discoveries += 1
            else:
                # Failed proposal
                if debug:
                    print(f"  Status: FAILED - Too far from ground truth")
                
                self.failed_proposals.append({
                    'iteration': self.iteration_count,
                    'proposal': proposal,
                    'nearest_distance': nearest_distance,
                    'nearest_pk': self.ground_truth.get_record(nearest_idx).get('pk', None)
                })
                
                iteration_failures += 1
                self.total_failures += 1
        
        # Update current pool with new discoveries
        if new_discoveries:
            self.current_pool.extend(new_discoveries)
        
        # Iteration statistics
        stats = {
            'iteration': self.iteration_count,
            'proposals': N,
            'discoveries': iteration_discoveries,
            'failures': iteration_failures,
            'discovery_rate': iteration_discoveries / N if N > 0 else 0.0,
            'pool_size': len(self.current_pool),
            'total_discovered': len(self.discovered_pks)
        }
        
        if debug:
            print(f"\n--- Iteration {self.iteration_count} Summary ---")
            print(f"Discoveries: {iteration_discoveries}/{N} ({stats['discovery_rate']*100:.1f}%)")
            print(f"Failures: {iteration_failures}/{N}")
            print(f"Total discovered so far: {len(self.discovered_pks)}/{len(self.ground_truth)}")
        
        return stats
    
    def run_iteration(
        self,
        N: int,
        proposal_mode: str = 'random',
        debug: bool = False,
        **proposal_kwargs
    ) -> Dict[str, Any]:
        """
        Run one complete iteration: generate proposals and evaluate them.
        
        This is a convenience method that combines generate_proposals() and
        evaluate_proposals(). For more control, use those methods separately.
        
        Args:
            N: Number of proposals to generate
            proposal_mode: Proposal generation mode ('random', 'random_so_n', etc.)
            debug: Whether to print debug information
            **proposal_kwargs: Additional arguments for proposal function
        
        Returns:
            Dictionary with iteration statistics
        """
        # Generate proposals
        proposals = self.generate_proposals(
            N=N,
            proposal_mode=proposal_mode,
            debug=debug,
            **proposal_kwargs
        )
        
        # Evaluate proposals
        stats = self.evaluate_proposals(proposals, debug=debug)
        
        return stats
    
    def run_simulation(
        self,
        num_iterations: int,
        N_per_iteration: int,
        proposal_mode: str = 'random',
        debug: bool = False,
        **proposal_kwargs
    ) -> List[Dict[str, Any]]:
        """
        Run multiple iterations of the search process.
        
        Args:
            num_iterations: Number of iterations to run
            N_per_iteration: Number of proposals per iteration
            proposal_mode: Proposal generation mode
            debug: Whether to print debug information
            **proposal_kwargs: Additional arguments for proposal function
        
        Returns:
            List of iteration statistics
        """
        all_stats = []
        
        for i in range(num_iterations):
            stats = self.run_iteration(
                N=N_per_iteration,
                proposal_mode=proposal_mode,
                debug=debug,
                **proposal_kwargs
            )
            all_stats.append(stats)
            
            # Early stopping if no new discoveries
            if stats['discoveries'] == 0 and i > 0:
                print(f"\nNo new discoveries in iteration {i+1}. Consider stopping or changing strategy.")
        
        return all_stats
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of the simulation.
        
        Returns:
            Dictionary with summary statistics
        """
        return {
            'iterations': self.iteration_count,
            'total_proposals': self.total_proposals,
            'total_discoveries': self.total_discoveries,
            'total_failures': self.total_failures,
            'overall_discovery_rate': self.total_discoveries / self.total_proposals if self.total_proposals > 0 else 0.0,
            'unique_discovered': len(self.discovered_pks),
            'ground_truth_size': len(self.ground_truth),
            'coverage': len(self.discovered_pks) / len(self.ground_truth) if len(self.ground_truth) > 0 else 0.0,
            'current_pool_size': len(self.current_pool),
            'results_size': len(self.results)
        }
    
    def print_summary(self) -> None:
        """Print a formatted summary of the simulation."""
        summary = self.get_summary()
        
        print(f"\n{'='*60}")
        print("SIMULATION SUMMARY")
        print(f"{'='*60}")
        print(f"Iterations completed: {summary['iterations']}")
        print(f"Total proposals: {summary['total_proposals']}")
        print(f"Successful discoveries: {summary['total_discoveries']} ({summary['overall_discovery_rate']*100:.1f}%)")
        print(f"Failed proposals: {summary['total_failures']}")
        print(f"Unique states discovered: {summary['unique_discovered']}/{summary['ground_truth_size']} ({summary['coverage']*100:.1f}% coverage)")
        print(f"Current pool size: {summary['current_pool_size']}")
        print(f"{'='*60}\n")
    
    def _add_to_results(
        self,
        occ_data: OccupationMatrixData,
        energy: float,
        pk: Optional[int],
        iteration: int,
        source_pks: List[int],
        nearest_distance: float,
        discovery_status: str
    ) -> None:
        """Add a discovered state to results DataBank with metadata."""
        record = {
            'occ_data': occ_data,
            'energy': energy,
            'pk': pk,
            'converged': True,
            # Metadata
            'iteration': iteration,
            'source_pks': source_pks,
            'nearest_distance': nearest_distance,
            'discovery_status': discovery_status  # 'initial', 'discovered'
        }
        
        self.results._records.append(record)
    
    def _get_current_pool_pks(self) -> List[int]:
        """Get PKs of current pool matrices."""
        pks = []
        for record in self.results._records:
            if record.get('pk') is not None:
                pks.append(record['pk'])
        return pks
    
    def export_results(self, filepath: str) -> None:
        """
        Export results DataBank to JSON file.
        
        Args:
            filepath: Path to save JSON file
        """
        self.results.to_json(filepath)
        print(f"Results exported to {filepath}")
    
    def get_failed_proposals_summary(self) -> Dict[str, Any]:
        """
        Get summary of failed proposals.
        
        Returns:
            Dictionary with failure analysis
        """
        if not self.failed_proposals:
            return {'count': 0, 'mean_distance': None, 'min_distance': None, 'max_distance': None}
        
        distances = [fp['nearest_distance'] for fp in self.failed_proposals]
        
        return {
            'count': len(self.failed_proposals),
            'mean_distance': np.mean(distances),
            'min_distance': np.min(distances),
            'max_distance': np.max(distances),
            'distances': distances
        }
