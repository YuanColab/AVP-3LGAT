"""
ESMC Embedding Processor
"""

import os
import time
import torch
import pandas as pd
from typing import List, Optional
from pathlib import Path
from tqdm import tqdm
from loguru import logger

# ESM related imports
from esm.models.esmc import ESMC
from esm.sdk.api import (
    ESMProtein,
    LogitsConfig,
    LogitsOutput,
    ESMProteinError,
)

from .utils import (
    setup_dataset_logging, 
    filter_existing_files, 
    format_time, 
    create_output_directory,
    get_device
)

class ESMCEmbeddingProcessor:
    """ESMC embedding processor"""
    
    def __init__(self, device='auto', model_name="esmc_600m"):
        self.device = get_device() if device == 'auto' else device
        self.model_name = model_name
        self.model = None
        self.embedding_config = LogitsConfig(
            sequence=True,
            return_embeddings=True
        )
        
        logger.info(f"ESMC Processor initialized with device: {self.device}")
        
    def load_model(self):
        """Load ESMC model"""
        if self.model is None:
            logger.info(f"Loading ESMC model {self.model_name} to {self.device}")
            self.model = ESMC.from_pretrained(self.model_name).to(self.device)
            logger.info("ESMC model loaded successfully")
        
    def embed_sequence(self, sequence: str) -> LogitsOutput:
        """Embedding processing for single protein sequence"""
        protein = ESMProtein(sequence=sequence)
        protein_tensor = self.model.encode(protein)
        output = self.model.logits(protein_tensor, self.embedding_config)
        return output
    
    def batch_embed_with_progress(self, sequences: List[str], ids: List[str]) -> List[LogitsOutput]:
        """Batch process protein sequences with detailed progress"""
        results = []
        start_time = time.time()
        
        progress_bar = tqdm(
            total=len(sequences), 
            desc="ESMC embedding", 
            unit="seq",
            ncols=None,
            leave=True
        )
        
        for i, (sequence, seq_id) in enumerate(zip(sequences, ids)):
            iter_start = time.time()
            try:
                result = self.embed_sequence(sequence)
                results.append(result)
                success = True
            except Exception as e:
                logger.warning(f"Embedding failed for sequence {seq_id}: {e}")
                results.append(ESMProteinError(500, str(e)))
                success = False
            
            # Calculate time statistics
            iter_time = time.time() - iter_start
            elapsed_total = time.time() - start_time
            avg_time_per_seq = elapsed_total / (i + 1)
            eta_seconds = avg_time_per_seq * (len(sequences) - i - 1)
            
            # Update progress bar
            progress_bar.update(1)
            progress_bar.set_postfix({
                'current': seq_id[:8] + "..." if len(seq_id) > 8 else seq_id,
                'time': f"{iter_time:.2f}s",
                'ETA': format_time(eta_seconds),
                'status': "✓" if success else "✗"
            })
            
            # Clean memory every 50 sequences
            if (i + 1) % 50 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        progress_bar.close()
        return results
    
    def process_dataset(self, csv_file: str, output_dir: Optional[str] = None, skip_existing: bool = True):
        """Process dataset and save embeddings"""
        logger.info(f"Starting ESMC embedding processing: {csv_file}")
        
        # Create output directory
        if output_dir is None:
            output_dir = create_output_directory(csv_file, "_esmc")
        else:
            os.makedirs(output_dir, exist_ok=True)
        
        # Read data
        df = pd.read_csv(csv_file, dtype={'ID': str})
        sequences = df["Sequence"].tolist()
        ids = df["Id"].tolist()
        
        logger.info(f"Dataset size: {len(df)} sequences")
        logger.info(f"Output directory: {output_dir}")
        
        # Check existing files
        if skip_existing:
            unprocessed_ids, unprocessed_sequences, existing_files = filter_existing_files(
                ids, sequences, output_dir, "pt", min_size=100
            )
            
            if existing_files:
                logger.info(f"Found {len(existing_files)} existing ESMC embedding files, will skip")
            
            if not unprocessed_sequences:
                logger.info("All ESMC embedding files exist, skipping processing")
                return len(existing_files), 0, output_dir
            
            logger.info(f"Need to process {len(unprocessed_sequences)} sequences (skipping {len(existing_files)})")
            ids, sequences = unprocessed_ids, unprocessed_sequences
        
        # Load model if not loaded
        if self.model is None:
            self.load_model()
        
        # Batch processing
        outputs = self.batch_embed_with_progress(sequences, ids)
        
        # Calculate results
        successful_count = 0
        failed_count = 0
        
        # Save individual sequence embeddings
        logger.info("Saving ESMC embedding files...")
        for i, (output, seq_id) in enumerate(tqdm(zip(outputs, ids), total=len(outputs), desc="Saving embeddings")):
            try:
                if hasattr(output, 'embeddings'):
                    sequence_embeddings = output.embeddings[0, 1:-1, :]  # Remove [CLS] and [SEP] tokens
                    output_file = os.path.join(output_dir, f"{seq_id}.pt")
                    torch.save(sequence_embeddings, output_file)
                    successful_count += 1
                else:
                    failed_count += 1
            except Exception as e:
                logger.error(f"Failed to save embedding for {seq_id}: {e}")
                failed_count += 1
        
        # Add existing files to success count if skipped
        if skip_existing and existing_files:
            total_successful = successful_count + len(existing_files)
            logger.info(f"ESMC processing completed: new {successful_count}, existing {len(existing_files)}, total success {total_successful}, failed {failed_count}")
            return total_successful, failed_count, output_dir
        else:
            logger.info(f"ESMC processing completed: success {successful_count}, failed {failed_count}")
            return successful_count, failed_count, output_dir

def process_esmc_features(csv_file: str, output_dir: Optional[str] = None, skip_existing: bool = True):
    """Convenience function to process ESMC features"""
    processor = ESMCEmbeddingProcessor()
    return processor.process_dataset(csv_file, output_dir, skip_existing)