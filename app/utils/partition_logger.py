"""
Partition Logger
================

Logging utility for dataset partitioning process with detailed tracking
of bonafide and spoof file discovery, processing statistics, and warnings.
"""

from loguru import logger
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import sys


class PartitionLogger:
    """
    Handles logging operations for dataset partitioning pipeline.
    
    Provides detailed logging of:
    - Speaker discovery (bonafide and spoof)
    - File counts per technique
    - Processing progress
    - Warnings for missing spoof data
    - Final statistics and discrepancies
    """

    def __init__(self, log_file: str = "partition_debug.log"):
        """
        Initialize the partition logger with file and console handlers.

        Args:
            log_file: Path to the log file
        """
        try:
            self.log_file = Path(log_file)
            self.session_start = datetime.now()
            
            # Remove default logger
            logger.remove()
            
            # Add console handler
            logger.add(
                sys.stdout,
                format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
                level="INFO",
                colorize=True
            )
            
            # Add file handler
            logger.add(
                self.log_file,
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
                level="DEBUG",
                rotation="10 MB",
                retention="7 days",
                compression="zip",
                mode="w"
            )
            
            logger.info("=" * 80)
            logger.info("DATASET PARTITIONING LOG")
            logger.info(f"Session started: {self.session_start}")
            logger.info("=" * 80)
            
        except Exception as e:
            print(f"PartitionLogger.__init__: Error initializing logger. Error: {e}")
    
    def log_section(self, title: str):
        """
        Log a section separator with title.

        Args:
            title: Section title
        """
        try:
            separator = "=" * 70
            logger.info("")
            logger.info(separator)
            logger.info(title)
            logger.info(separator)
        except Exception as e:
            print(f"PartitionLogger.log_section: Error. {e}")
    
    def log_config(self, source_dir: str, output_dir: str, split_ratios: tuple, seed: int):
        """
        Log partitioning configuration.

        Args:
            source_dir: Source directory path
            output_dir: Output directory path
            split_ratios: Train/val/test split ratios
            seed: Random seed
        """
        try:
            self.log_section("CONFIGURATION")
            logger.info(f"Source directory: {source_dir}")
            logger.info(f"Output directory: {output_dir}")
            logger.info(f"Split ratios (train/val/test): {split_ratios}")
            logger.info(f"Random seed: {seed}")
        except Exception as e:
            print(f"PartitionLogger.log_config: Error. {e}")
    
    def log_checkpoint_resume(self, processed_count: int, bonafide_files: int, spoof_files: int):
        """
        Log checkpoint resumption information.

        Args:
            processed_count: Number of already processed speakers
            bonafide_files: Number of bonafide files processed
            spoof_files: Number of spoof files processed
        """
        try:
            logger.info("")
            logger.warning("RESUMING FROM CHECKPOINT")
            logger.info(f"  Already processed: {processed_count} speakers")
            logger.info(f"  Bonafide files: {bonafide_files:,}")
            logger.info(f"  Spoof files: {spoof_files:,}")
        except Exception as e:
            print(f"PartitionLogger.log_checkpoint_resume: Error. {e}")
    
    def log_country_scan(self, country: str, speaker_count: int):
        """
        Log country scanning progress.

        Args:
            country: Country name
            speaker_count: Number of speakers found
        """
        try:
            logger.info(f"Scanning {country}: {speaker_count} speakers found")
        except Exception as e:
            print(f"PartitionLogger.log_country_scan: Error. {e}")
    
    def log_speaker_discovery(self, speaker_id: str, bonafide_count: int):
        """
        Log discovery of bonafide speaker.

        Args:
            speaker_id: Speaker identifier
            bonafide_count: Number of bonafide files
        """
        try:
            logger.debug(f"  Found {speaker_id}: {bonafide_count} bonafide files")
        except Exception as e:
            print(f"PartitionLogger.log_speaker_discovery: Error. {e}")
    
    def log_speaker_processing_start(self, speaker_id: str, index: int, total: int):
        """
        Log start of speaker processing.

        Args:
            speaker_id: Speaker identifier
            index: Current speaker index
            total: Total speakers to process
        """
        try:
            logger.info("")
            logger.info(f"[{index}/{total}] Processing {speaker_id}...")
        except Exception as e:
            print(f"PartitionLogger.log_speaker_processing_start: Error. {e}")
    
    def log_spoof_discovery(
        self, 
        speaker_id: str, 
        spoof_counts: Dict[str, int],
        total_bonafide: int
    ):
        """
        Log spoof file discovery for a speaker with warning if none found.

        Args:
            speaker_id: Speaker identifier
            spoof_counts: Dictionary mapping technique to file count
            total_bonafide: Total bonafide files for this speaker
        """
        try:
            total_spoof = sum(spoof_counts.values())
            
            if total_spoof == 0:
                logger.warning(
                    f"  WARNING: No spoof files found for {speaker_id} "
                    f"(has {total_bonafide} bonafide files)"
                )
            else:
                logger.debug(f"  Spoof files for {speaker_id}:")
                for technique, count in spoof_counts.items():
                    if count > 0:
                        logger.debug(f"    {technique.upper()}: {count} files")
        except Exception as e:
            print(f"PartitionLogger.log_spoof_discovery: Error. {e}")
    
    def log_speaker_complete(
        self, 
        speaker_id: str, 
        bonafide_count: int, 
        spoof_counts: Dict[str, int],
        splits: Dict[str, Dict[str, int]] = None
    ):
        """
        Log completion of speaker partitioning with split details.

        Args:
            speaker_id: Speaker identifier
            bonafide_count: Total bonafide files
            spoof_counts: Dictionary mapping technique to file count
            splits: Dictionary with split counts (train/val/test per technique)
        """
        try:
            total_spoof = sum(spoof_counts.values())
            total_files = bonafide_count + total_spoof
            
            logger.info(f"  Processed {speaker_id}:")
            logger.info(f"    Bonafide: {bonafide_count} files")
            
            for technique, count in spoof_counts.items():
                if count > 0:
                    logger.info(f"    {technique.upper()}: {count} files")
            
            logger.info(f"    Total: {total_files} files")
            
            # Log split distribution
            if splits:
                logger.debug(f"    Splits:")
                for split_name in ['train', 'val', 'test']:
                    if split_name in splits:
                        split_total = sum(splits[split_name].values())
                        logger.debug(f"      {split_name}: {split_total} files")
                        
        except Exception as e:
            print(f"PartitionLogger.log_speaker_complete: Error. {e}")
    
    def log_speaker_skipped(self, speaker_id: str, reason: str):
        """
        Log skipped speaker with reason.

        Args:
            speaker_id: Speaker identifier
            reason: Reason for skipping
        """
        try:
            logger.warning(f"  SKIPPED {speaker_id}: {reason}")
        except Exception as e:
            print(f"PartitionLogger.log_speaker_skipped: Error. {e}")
    
    def log_checkpoint_saved(self, processed_count: int):
        """
        Log checkpoint save event.

        Args:
            processed_count: Number of processed speakers
        """
        try:
            logger.debug(f"  [Checkpoint saved: {processed_count} speakers completed]")
        except Exception as e:
            print(f"PartitionLogger.log_checkpoint_saved: Error. {e}")
    
    def log_final_statistics(
        self,
        total_speakers: int,
        successful: int,
        skipped: int,
        bonafide_files: int,
        spoof_counts: Dict[str, int],
        speakers_without_spoof: List[str]
    ):
        """
        Log final processing statistics with detailed breakdown.

        Args:
            total_speakers: Total speakers found
            successful: Successfully partitioned speakers
            skipped: Skipped speakers
            bonafide_files: Total bonafide files
            spoof_counts: Dictionary mapping technique to total files
            speakers_without_spoof: List of speakers with no spoof data
        """
        try:
            self.log_section("PARTITIONING COMPLETE")
            
            logger.info(f"Total speakers found: {total_speakers}")
            logger.info(f"Successfully partitioned: {successful}")
            
            if skipped > 0:
                logger.warning(f"Skipped: {skipped}")
            else:
                logger.success(f"Skipped: {skipped}")
            
            logger.info("")
            logger.info("File counts:")
            logger.info(f"  Bonafide: {bonafide_files:,}")
            
            total_spoof = sum(spoof_counts.values())
            logger.info(f"  Spoof (total): {total_spoof:,}")
            
            for technique, count in sorted(spoof_counts.items()):
                if count > 0:
                    logger.info(f"    {technique.upper()}: {count:,}")
            
            logger.info(f"  Total files: {bonafide_files + total_spoof:,}")
            
            # Calculate and log imbalance
            if total_spoof > 0:
                ratio = bonafide_files / total_spoof
                logger.info("")
                logger.info(f"Bonafide/Spoof ratio: {ratio:.2f}")
                
                if ratio < 0.4:
                    logger.warning("  Significant imbalance: Much more spoof than bonafide")
                    logger.warning("  Recommend data augmentation for bonafide samples")
                elif ratio > 2.5:
                    logger.warning("  Significant imbalance: Much more bonafide than spoof")
            
            # Log speakers without spoof data
            if speakers_without_spoof:
                logger.info("")
                logger.warning(
                    f"Speakers without spoof data: {len(speakers_without_spoof)}/{total_speakers} "
                    f"({len(speakers_without_spoof)/total_speakers*100:.1f}%)"
                )
                
                if len(speakers_without_spoof) <= 10:
                    for speaker in speakers_without_spoof:
                        logger.warning(f"  - {speaker}")
                else:
                    for speaker in speakers_without_spoof[:5]:
                        logger.warning(f"  - {speaker}")
                    logger.warning(f"  ... and {len(speakers_without_spoof) - 5} more")
                    logger.warning(f"  (See log file for complete list)")
                    
                    # Write complete list to log file only
                    logger.debug("")
                    logger.debug("Complete list of speakers without spoof data:")
                    for speaker in speakers_without_spoof:
                        logger.debug(f"  - {speaker}")
            
        except Exception as e:
            print(f"PartitionLogger.log_final_statistics: Error. {e}")
    
    def log_error(self, context: str, error: Exception):
        """
        Log an error with context.

        Args:
            context: Context where error occurred
            error: Exception object
        """
        try:
            logger.error(f"ERROR in {context}: {str(error)}")
            logger.debug(f"Full traceback:", exc_info=error)
        except Exception as e:
            print(f"PartitionLogger.log_error: Error. {e}")
    
    def close_session(self):
        """
        Close the logging session and log duration.
        """
        try:
            session_end = datetime.now()
            duration = session_end - self.session_start
            
            logger.info("")
            logger.info("=" * 80)
            logger.info(f"Session ended: {session_end}")
            logger.info(f"Total duration: {duration}")
            logger.info("=" * 80)
            
            logger.complete()
            
        except Exception as e:
            print(f"PartitionLogger.close_session: Error. {e}")