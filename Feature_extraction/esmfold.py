"""
ESMFold Structure Predictor
"""

import os
import time
import torch
import pandas as pd
from typing import List, Optional
from pathlib import Path
from tqdm import tqdm
from loguru import logger
from transformers import EsmForProteinFolding

from .utils import (
    setup_dataset_logging, 
    filter_existing_files, 
    format_time, 
    create_output_directory,
    get_device
)

class ESMFoldStructurePredictor:
    """ESMFold structure predictor"""
    
    def __init__(self, device='auto', batch_size=5, model_cache_dir="./model_cache"):
        self.device = torch.device(get_device() if device == 'auto' else device)
        self.batch_size = batch_size
        self.model_cache_dir = model_cache_dir
        self.model = None

        logger.info(f"ESMFold Processor initialized with device: {self.device}, batch size: {self.batch_size}")
        
    def load_model(self):
        """Load ESMFold model"""
        if self.model is None:
            logger.info(f"Loading ESMFold model to {self.device}")
            
            os.makedirs(self.model_cache_dir, exist_ok=True)
            
            try:
                # First try to load from local cache
                self.model = EsmForProteinFolding.from_pretrained(
                    "facebook/esmfold_v1",
                    cache_dir=self.model_cache_dir,
                    local_files_only=True
                )
                logger.info("ESMFold model loaded from local cache successfully")
            except Exception as e:
                logger.warning(f"Failed to load from local cache, trying download: {e}")
                # If local cache doesn't exist, download
                self.model = EsmForProteinFolding.from_pretrained(
                    "facebook/esmfold_v1",
                    cache_dir=self.model_cache_dir,
                    local_files_only=False
                )
                logger.info("ESMFold model downloaded successfully")
            
            self.model.to(self.device)
            self.model.eval()
            logger.info("ESMFold model loading completed")
        
    def predict_structures_with_progress(self, sequences: List[str], ids: List[str], output_dir: str):
        """Batch predict protein structures with detailed progress display"""
        if self.model is None:
            self.load_model()
            
        os.makedirs(output_dir, exist_ok=True)
        
        total_seqs = len(sequences)
        processed = 0
        start_time = time.time()
        
        logger.info(f"Starting structure prediction, total {total_seqs} sequences")
        logger.info(f"Batch size: {self.batch_size}, estimated batches: {(total_seqs + self.batch_size - 1) // self.batch_size}")
        
        # Create progress bar
        progress_bar = tqdm(
            total=total_seqs, 
            desc="ESMFold prediction", 
            unit="seq",
            ncols=None,
            leave=True
        )
        
        try:
            for i in range(0, total_seqs, self.batch_size):
                batch_start_time = time.time()
                batch_seqs = sequences[i:i + self.batch_size]
                batch_ids = ids[i:i + self.batch_size]
                batch_num = i // self.batch_size + 1
                total_batches = (total_seqs + self.batch_size - 1) // self.batch_size
                
                try:
                    with torch.no_grad():
                        progress_bar.set_description(f"ESMFold batch {batch_num}/{total_batches}")
                        pdbs = self.model.infer_pdbs(batch_seqs)
                    
                    # Save PDB files
                    for pdb, seq_id in zip(pdbs, batch_ids):
                        pdb_path = os.path.join(output_dir, f"{seq_id}.pdb")
                        with open(pdb_path, 'w') as f:
                            f.write(pdb)
                    
                    processed += len(batch_seqs)
                    batch_time = time.time() - batch_start_time
                    
                    # Calculate speed and ETA
                    elapsed_total = time.time() - start_time
                    avg_time_per_seq = elapsed_total / processed if processed > 0 else 0
                    eta_seconds = avg_time_per_seq * (total_seqs - processed)
                    
                    # Update progress bar
                    progress_bar.update(len(batch_seqs))
                    progress_bar.set_postfix({
                        'completed': f"{processed}/{total_seqs}",
                        'batch_time': f"{batch_time:.1f}s",
                        'ETA': format_time(eta_seconds) if eta_seconds > 0 else "calculating..."
                    })
                    
                    # Clean GPU cache
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    logger.error(f"Batch {batch_num} prediction failed: {e}")
                    progress_bar.update(len(batch_seqs))
                    
        finally:
            progress_bar.close()
        
        total_time = time.time() - start_time
        logger.info(f"Structure prediction completed!")
        logger.info(f"Total processed: {processed}/{total_seqs} sequences")
        logger.info(f"Total time: {format_time(total_time)}")
        if total_time > 0:
            logger.info(f"Average speed: {processed/total_time:.2f} sequences/second")
        
        return processed
    
    def process_dataset(self, csv_file: str, output_dir: Optional[str] = None, skip_existing: bool = True):
        """Process dataset and predict structures"""
        logger.info(f"Starting ESMFold structure prediction: {csv_file}")
        
        # Create output directory
        if output_dir is None:
            output_dir = create_output_directory(csv_file, "_esmfold")
        else:
            os.makedirs(output_dir, exist_ok=True)
        
        df = pd.read_csv(csv_file)
        sequences = df["Sequence"].tolist()
        ids = df["Id"].tolist()
        
        logger.info(f"Dataset size: {len(df)} sequences")
        logger.info(f"Output directory: {output_dir}")
        
        # Check existing files
        if skip_existing:
            unprocessed_ids, unprocessed_sequences, existing_files = filter_existing_files(
                ids, sequences, output_dir, "pdb", min_size=500
            )
            
            if existing_files:
                logger.info(f"Found {len(existing_files)} existing PDB files, will skip")
            
            if not unprocessed_sequences:
                logger.info("All PDB files exist, skipping processing")
                return len(existing_files), output_dir
            
            logger.info(f"Need to process {len(unprocessed_sequences)} sequences (skipping {len(existing_files)})")
            ids, sequences = unprocessed_ids, unprocessed_sequences
        
        # Use improved prediction method
        processed = self.predict_structures_with_progress(sequences, ids, output_dir)
        
        # Calculate total results
        if skip_existing and existing_files:
            total_processed = processed + len(existing_files)
            logger.info(f"ESMFold processing completed: new {processed}, existing {len(existing_files)}, total {total_processed}")
            return total_processed, output_dir
        else:
            return processed, output_dir

def process_esmfold_structures(csv_file: str, output_dir: Optional[str] = None, skip_existing: bool = True):
    """Convenience function to process ESMFold structures"""
    predictor = ESMFoldStructurePredictor()
    return predictor.process_dataset(csv_file, output_dir, skip_existing)