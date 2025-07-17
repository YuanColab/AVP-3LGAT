"""
AAIndex Feature Processor

This module provides functionality to extract and process AAIndex features
from protein sequences for machine learning applications.

Features:
- Extract 553-dimensional AAIndex features for each amino acid
- Support for global normalization using fixed AAIndex ranges
- Batch processing with progress tracking
- Intelligent file skipping to avoid reprocessing
- Comprehensive error handling and logging
"""

import os
import time
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Tuple, Union
from pathlib import Path
from tqdm import tqdm
from loguru import logger
from sklearn.preprocessing import MinMaxScaler

from .utils import (
    setup_dataset_logging, 
    filter_existing_files, 
    create_output_directory
)


class AAIndexProcessor:
    """
    AAIndex feature processor for protein sequences.
    
    This class handles the extraction of AAIndex features from protein sequences,
    provides normalization options, and supports batch processing.
    
    Attributes:
        aaindex_file (str): Path to the AAIndex CSV file
        aaindex_dict (Dict): Mapping of amino acids to feature vectors
        normalization_method (str): Method for feature normalization
        global_stats_computed (bool): Whether global statistics have been computed
        default_base_dir (str): Default base directory for output files
    """
    
    def __init__(self, aaindex_file: str = "Feature_extraction/aaindex1.csv"):
        """
        Initialize the AAIndex processor.
        
        Args:
            aaindex_file (str): Path to the AAIndex CSV file
        """
        self.aaindex_file = aaindex_file
        self.default_base_dir = "3_Graph_Data"  # 固定的默认目录
        self.aaindex_dict = None
        self.normalization_method = 'fixed_range'
        self.global_stats_computed = False
        
        # Global normalization statistics
        self.global_min = None
        self.global_max = None
        self.valid_features = None
        
        # Legacy scaler for backward compatibility
        self.scaler = MinMaxScaler()
        
        logger.info(f"AAIndex Processor initialized with file: {aaindex_file}")
        logger.info(f"Default output directory: {self.default_base_dir}")
    
    # ==================== Core Feature Extraction ====================
    
    def load_aaindex_data(self) -> None:
        """
        Load AAIndex data from CSV file with comprehensive validation.
        
        Raises:
            FileNotFoundError: If AAIndex file doesn't exist
            ValueError: If AAIndex file format is invalid
            Exception: For other loading errors
        """
        if self.aaindex_dict is not None:
            return
        
        try:
            # Validate file existence
            if not os.path.exists(self.aaindex_file):
                raise FileNotFoundError(f"AAIndex file not found: {self.aaindex_file}")
            
            logger.info(f"Loading AAIndex data from: {self.aaindex_file}")
            
            # Load and validate data
            aaindex1_df = pd.read_csv(self.aaindex_file, index_col='Description')
            
            # Check for required amino acids
            expected_aa = set('ACDEFGHIKLMNPQRSTVWY')
            available_aa = set(aaindex1_df.columns)
            missing_aa = expected_aa - available_aa
            
            if missing_aa:
                logger.warning(f"Missing amino acids in AAIndex file: {missing_aa}")
            
            # Handle missing values
            if aaindex1_df.isnull().any().any():
                logger.warning("AAIndex data contains NaN values, replacing with 0")
                aaindex1_df = aaindex1_df.fillna(0)
            
            # Create amino acid to feature mapping
            self.aaindex_dict = {
                aa: aaindex1_df[aa].values.astype(np.float32) 
                for aa in aaindex1_df.columns
            }
            
            logger.info(
                f"AAIndex data loaded successfully: "
                f"{len(self.aaindex_dict)} amino acids, "
                f"{len(aaindex1_df)} features"
            )
            
        except Exception as e:
            logger.error(f"Failed to load AAIndex data: {e}")
            raise
    
    def extract_features_from_sequence(self, sequence: str) -> np.ndarray:
        """
        Extract AAIndex features from a protein sequence.
        
        Args:
            sequence (str): Protein sequence string
            
        Returns:
            np.ndarray: Feature matrix of shape (seq_len, 553) with dtype float32
            
        Note:
            Unknown amino acids are represented as zero vectors
        """
        if self.aaindex_dict is None:
            self.load_aaindex_data()
        
        # Pre-allocate feature matrix for efficiency
        feature_dim = len(next(iter(self.aaindex_dict.values())))
        features = np.zeros((len(sequence), feature_dim), dtype=np.float32)
        
        unknown_count = 0
        for i, aa in enumerate(sequence):
            if aa in self.aaindex_dict:
                features[i] = self.aaindex_dict[aa]
            else:
                # features[i] is already zero vector
                unknown_count += 1
        
        # Log unknown amino acids (debug level to avoid spam)
        if unknown_count > 0:
            logger.debug(
                f"Sequence contains {unknown_count}/{len(sequence)} "
                f"unknown amino acids"
            )
        
        return features
    
    # ==================== Normalization Methods ====================
    
    def compute_global_normalization_stats(self) -> None:
        """
        Compute global normalization statistics from AAIndex database.
        
        This method calculates min/max values for each feature across all
        amino acids in the AAIndex database, ensuring consistent normalization.
        """
        if self.global_stats_computed:
            return
        
        if self.aaindex_dict is None:
            self.load_aaindex_data()
        
        logger.info("Computing global normalization statistics...")
        
        # Stack all amino acid feature vectors
        all_values = np.array(list(self.aaindex_dict.values()))  # (20, 553)
        
        # Compute global min/max for each feature
        self.global_min = np.min(all_values, axis=0)  # (553,)
        self.global_max = np.max(all_values, axis=0)  # (553,)
        
        # Identify features with valid ranges (avoid division by zero)
        feature_ranges = self.global_max - self.global_min
        self.valid_features = feature_ranges > 1e-8
        
        self.global_stats_computed = True
        
        valid_count = np.sum(self.valid_features)
        total_count = len(self.valid_features)
        logger.info(
            f"Global normalization stats computed: "
            f"{valid_count}/{total_count} features have valid ranges"
        )
    
    def normalize_features_fixed_range(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features using fixed AAIndex database ranges.
        
        This is the recommended normalization method as it ensures consistency
        across all sequences and maintains the relative relationships between
        amino acid properties.
        
        Args:
            features (np.ndarray): Input features to normalize
            
        Returns:
            np.ndarray: Normalized features with values in [0, 1] range
        """
        if not self.global_stats_computed:
            self.compute_global_normalization_stats()
        
        normalized = np.zeros_like(features, dtype=np.float32)
        
        # Apply normalization only to valid features
        if len(features.shape) == 2:  # (seq_len, feature_dim)
            valid_mask = self.valid_features
            normalized[:, valid_mask] = (
                (features[:, valid_mask] - self.global_min[valid_mask]) /
                (self.global_max[valid_mask] - self.global_min[valid_mask])
            )
        else:  # 1D features
            valid_mask = self.valid_features
            normalized[valid_mask] = (
                (features[valid_mask] - self.global_min[valid_mask]) /
                (self.global_max[valid_mask] - self.global_min[valid_mask])
            )
        
        # Clip to [0, 1] range to handle any numerical issues
        normalized = np.clip(normalized, 0.0, 1.0)
        
        return normalized
    
    def normalize_features_legacy(self, features: np.ndarray) -> np.ndarray:
        """
        Legacy normalization method using MinMaxScaler.
        
        Warning: This method normalizes each sequence independently,
        which can lead to inconsistent feature scales across sequences.
        Use normalize_features_fixed_range() instead.
        
        Args:
            features (np.ndarray): Input features to normalize
            
        Returns:
            np.ndarray: Normalized features
        """
        logger.warning(
            "Using legacy normalization method. Consider using 'fixed_range' "
            "for better consistency across sequences."
        )
        
        original_shape = features.shape
        if features.ndim == 3:
            features_reshaped = features.reshape(-1, features.shape[-1])
            normalized = self.scaler.fit_transform(features_reshaped)
            normalized = normalized.reshape(original_shape)
        else:
            normalized = self.scaler.fit_transform(features)
        
        return normalized.astype(np.float32)
    
    def normalize_features(self, features: np.ndarray, 
                          method: str = None) -> np.ndarray:
        """
        Normalize features using the specified method.
        
        Args:
            features (np.ndarray): Input features to normalize
            method (str, optional): Normalization method ('fixed_range' or 'legacy')
                                  If None, uses self.normalization_method
            
        Returns:
            np.ndarray: Normalized features
        """
        if method is None:
            method = self.normalization_method
        
        if method == 'fixed_range':
            return self.normalize_features_fixed_range(features)
        elif method == 'legacy':
            return self.normalize_features_legacy(features)
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    # ==================== Single Sequence Processing ====================
    
    def process_single_sequence(self, sequence: str, seq_id: str, 
                              output_dir: str, normalize: bool = True,
                              normalization_method: str = None) -> bool:
        """
        Process a single sequence and save features to file.
        
        Args:
            sequence (str): Protein sequence
            seq_id (str): Sequence identifier
            output_dir (str): Output directory for saving features
            normalize (bool): Whether to normalize features
            normalization_method (str, optional): Normalization method to use
            
        Returns:
            bool: True if processing successful, False otherwise
        """
        try:
            # Extract features
            features = self.extract_features_from_sequence(sequence)
            
            # Normalize if requested
            if normalize:
                features = self.normalize_features(features, normalization_method)
            
            # Save features
            output_file = os.path.join(output_dir, f"{seq_id}.npy")
            np.save(output_file, features)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to process sequence {seq_id}: {e}")
            return False
    
    # ==================== Batch Processing ====================
    
    def process_dataset(self, csv_file: str, output_dir: Optional[str] = None, 
                       skip_existing: bool = True, normalize: bool = True,
                       normalization_method: str = 'fixed_range') -> Tuple[int, int, str, Dict]:
        """
        Process entire dataset and save AAIndex features.
        
        Args:
            csv_file (str): Path to CSV file containing sequences
            output_dir (str, optional): Output directory for features. 
                                      If None, uses {default_base_dir}/{csv_basename}_aaindex
            skip_existing (bool): Whether to skip existing feature files
            normalize (bool): Whether to normalize features
            normalization_method (str): Normalization method to use
            
        Returns:
            Tuple[int, int, str, Dict]: (successful_count, failed_count, 
                                       output_dir, processing_stats)
        """
        start_time = time.time()
        logger.info(f"Starting AAIndex feature processing: {csv_file}")
        
        # Set normalization method
        self.normalization_method = normalization_method
        
        # Handle output directory
        if output_dir is None:
            csv_basename = os.path.splitext(os.path.basename(csv_file))[0]
            
            # 确保默认基础目录存在
            os.makedirs(self.default_base_dir, exist_ok=True)
            output_dir = os.path.join(self.default_base_dir, f"{csv_basename}_aaindex")
            
            logger.info(f"No output directory specified, using default: {output_dir}")
        
        # 确保输出目录存在（无论是指定的还是默认的）
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory created/verified: {output_dir}")
        
        # Read and validate data
        try:
            df = pd.read_csv(csv_file)
            required_columns = ['Id', 'Sequence']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            sequences = df["Sequence"].tolist()
            ids = df["Id"].tolist()
            
        except Exception as e:
            logger.error(f"Failed to read CSV file: {e}")
            raise
        
        logger.info(f"Dataset size: {len(df)} sequences")
        logger.info(f"Output directory: {output_dir}")
        
        # Handle existing files
        existing_count = 0
        if skip_existing:
            unprocessed_ids, unprocessed_sequences, existing_files = filter_existing_files(
                ids, sequences, output_dir, "npy", min_size=50
            )
            
            existing_count = len(existing_files)
            if existing_count > 0:
                logger.info(f"Found {existing_count} existing AAIndex feature files")
            
            if not unprocessed_sequences:
                logger.info("All AAIndex feature files exist, skipping processing")
                stats = self._create_processing_stats(
                    existing_count, 0, 0, output_dir, normalize, 
                    normalization_method, time.time() - start_time
                )
                return existing_count, 0, output_dir, stats
            
            logger.info(f"Processing {len(unprocessed_sequences)} new sequences")
            ids, sequences = unprocessed_ids, unprocessed_sequences
        
        # Pre-compute normalization statistics if needed
        if normalize:
            self.compute_global_normalization_stats()
        
        # Process sequences with progress tracking
        successful_count, failed_count = self._process_sequences_with_progress(
            sequences, ids, output_dir, normalize, normalization_method
        )
        
        # Calculate total results
        total_successful = successful_count + existing_count
        processing_time = time.time() - start_time
        
        # Create processing statistics
        stats = self._create_processing_stats(
            total_successful, successful_count, failed_count, 
            output_dir, normalize, normalization_method, processing_time
        )
        
        # Log results
        if existing_count > 0:
            logger.info(
                f"AAIndex processing completed: "
                f"new {successful_count}, existing {existing_count}, "
                f"total success {total_successful}, failed {failed_count}"
            )
        else:
            logger.info(
                f"AAIndex processing completed: "
                f"success {successful_count}, failed {failed_count}"
            )
        
        return total_successful, failed_count, output_dir, stats
    
    def _process_sequences_with_progress(self, sequences: List[str], ids: List[str],
                                       output_dir: str, normalize: bool,
                                       normalization_method: str) -> Tuple[int, int]:
        """
        Process sequences with progress bar display.
        
        Args:
            sequences (List[str]): List of protein sequences
            ids (List[str]): List of sequence identifiers
            output_dir (str): Output directory
            normalize (bool): Whether to normalize features
            normalization_method (str): Normalization method
            
        Returns:
            Tuple[int, int]: (successful_count, failed_count)
        """
        successful_count = 0
        failed_count = 0
        
        progress_bar = tqdm(
            total=len(sequences), 
            desc="AAIndex features", 
            unit="seq",
            ncols=100,
            leave=True
        )
        
        try:
            for sequence, seq_id in zip(sequences, ids):
                if self.process_single_sequence(
                    sequence, seq_id, output_dir, normalize, normalization_method
                ):
                    successful_count += 1
                else:
                    failed_count += 1
                
                progress_bar.update(1)
                progress_bar.set_postfix({
                    'success': successful_count,
                    'failed': failed_count,
                    'rate': f"{successful_count/(successful_count+failed_count)*100:.1f}%"
                })
        
        finally:
            progress_bar.close()
        
        return successful_count, failed_count
    
    def _create_processing_stats(self, total_successful: int, new_successful: int,
                               failed_count: int, output_dir: str, normalize: bool,
                               normalization_method: str, processing_time: float) -> Dict:
        """Create comprehensive processing statistics."""
        return {
            'total_successful': total_successful,
            'new_processed': new_successful,
            'failed': failed_count,
            'output_dir': output_dir,
            'normalization_used': normalize,
            'normalization_method': normalization_method if normalize else None,
            'processing_time_seconds': round(processing_time, 2),
            'global_stats_computed': self.global_stats_computed,
            'feature_dimension': len(next(iter(self.aaindex_dict.values()))) if self.aaindex_dict else 0,
            'amino_acids_supported': len(self.aaindex_dict) if self.aaindex_dict else 0
        }
    
    # ==================== Feature Loading ====================
    
    def load_features(self, seq_id: str, features_dir: str) -> Optional[np.ndarray]:
        """
        Load AAIndex features for a specific sequence.
        
        Args:
            seq_id (str): Sequence identifier
            features_dir (str): Directory containing feature files
            
        Returns:
            Optional[np.ndarray]: Loaded features or None if loading failed
        """
        try:
            feature_file = os.path.join(features_dir, f"{seq_id}.npy")
            if os.path.exists(feature_file):
                features = np.load(feature_file)
                logger.debug(f"Loaded AAIndex features for {seq_id}: {features.shape}")
                return features
            else:
                logger.warning(f"AAIndex feature file not found: {feature_file}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to load AAIndex features for {seq_id}: {e}")
            return None
    
    # ==================== Utility Methods ====================
    
    def get_feature_info(self) -> Dict:
        """
        Get information about loaded features.
        
        Returns:
            Dict: Feature information including dimensions and statistics
        """
        if self.aaindex_dict is None:
            return {'loaded': False}
        
        info = {
            'loaded': True,
            'amino_acids_count': len(self.aaindex_dict),
            'feature_dimension': len(next(iter(self.aaindex_dict.values()))),
            'normalization_method': self.normalization_method,
            'global_stats_computed': self.global_stats_computed
        }
        
        if self.global_stats_computed:
            info.update({
                'valid_features_count': int(np.sum(self.valid_features)),
                'total_features_count': len(self.valid_features),
                'global_min_range': [float(np.min(self.global_min)), float(np.max(self.global_min))],
                'global_max_range': [float(np.min(self.global_max)), float(np.max(self.global_max))]
            })
        
        return info


# ==================== Convenience Functions ====================

def process_aaindex_features(csv_file: str, output_dir: Optional[str] = None, 
                           skip_existing: bool = True, normalize: bool = True,
                           normalization_method: str = 'fixed_range',
                           aaindex_file: str = "Feature_extraction/aaindex1.csv") -> Tuple[int, int, str, Dict]:
    """
    Convenience function to process AAIndex features.
    
    Args:
        csv_file (str): Path to CSV file containing sequences
        output_dir (str, optional): Output directory for features. 
                                  If None, uses 3_Graph_Data/{csv_basename}_aaindex
        skip_existing (bool): Whether to skip existing feature files
        normalize (bool): Whether to normalize features
        normalization_method (str): Normalization method ('fixed_range' or 'legacy')
        aaindex_file (str): Path to AAIndex CSV file
        
    Returns:
        Tuple[int, int, str, Dict]: (successful_count, failed_count, 
                                   output_dir, processing_stats)
    """
    processor = AAIndexProcessor(aaindex_file=aaindex_file)
    return processor.process_dataset(
        csv_file, output_dir, skip_existing, normalize, normalization_method
    )


def load_aaindex_features(seq_id: str, features_dir: str) -> Optional[np.ndarray]:
    """
    Convenience function to load AAIndex features for a sequence.
    
    Args:
        seq_id (str): Sequence identifier
        features_dir (str): Directory containing feature files
        
    Returns:
        Optional[np.ndarray]: Loaded features or None if loading failed
    """
    processor = AAIndexProcessor()
    return processor.load_features(seq_id, features_dir)


# ==================== Module Information ====================

__all__ = [
    'AAIndexProcessor',
    'process_aaindex_features', 
    'load_aaindex_features'
]

if __name__ == "__main__":
    # Example usage
    print("AAIndex Feature Processor")
    print("=========================")
    
    # Initialize processor
    processor = AAIndexProcessor()
    
    # Display feature information
    info = processor.get_feature_info()
    print(f"Feature info: {info}")
    
    # Example sequence processing
    example_sequence = "MKTVRQERLKSIVRILERSKEPVSGAQ"
    print(f"\nProcessing example sequence: {example_sequence}")
    
    try:
        features = processor.extract_features_from_sequence(example_sequence)
        print(f"Features shape: {features.shape}")
        print(f"Features dtype: {features.dtype}")
        
        # Normalize features
        normalized_features = processor.normalize_features(features)
        print(f"Normalized features shape: {normalized_features.shape}")
        print(f"Normalized features range: [{np.min(normalized_features):.3f}, {np.max(normalized_features):.3f}]")
        
    except Exception as e:
        print(f"Error processing example: {e}")