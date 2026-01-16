"""
Checkpoint Management Utility
=============================

CLI tool for checkpoint operations.

Usage:
    python checkpoint_tool.py partition status
    python checkpoint_tool.py partition clear
    python checkpoint_tool.py augment status --factor 3x
    python checkpoint_tool.py augment clear --factor 3x
"""

import argparse
from app.utils.checkpoint_manager import PartitionCheckpointManager, AugmentationCheckpointManager


def handle_partition(args):
    manager = PartitionCheckpointManager(args.checkpoint_file)
    
    if args.subcommand == 'status':
        if not manager.exists():
            print(f"No checkpoint: {args.checkpoint_file}")
            return
        
        data = manager.load_partition()
        print("\nPARTITION CHECKPOINT")
        print("="*50)
        print(f"File: {args.checkpoint_file}")
        print(f"Speakers: {len(data['processed_speakers'])}")
        
        if data.get('speaker_splits'):
            for split, spks in data['speaker_splits'].items():
                print(f"  {split}: {len(spks)}")
        print("="*50)
    
    elif args.subcommand == 'clear':
        if manager.exists():
            response = input(f"Clear {args.checkpoint_file}? [y/N]: ")
            if response.lower() in ['y', 'yes']:
                manager.clear()
                print("✓ Cleared")


def handle_augment(args):
    file = f"augmentation_{args.factor}_checkpoint.json" if args.factor else "augmentation_checkpoint.json"
    manager = AugmentationCheckpointManager(file)
    
    if args.subcommand == 'status':
        if not manager.exists():
            print(f"No checkpoint: {file}")
            return
        
        data = manager.load()
        print("\nAUGMENTATION CHECKPOINT")
        print("="*50)
        print(f"File: {file}")
        print(f"Progress: {manager.get_progress_percentage():.1f}%")
        print("="*50)
    
    elif args.subcommand == 'clear':
        if manager.exists():
            response = input(f"Clear {file}? [y/N]: ")
            if response.lower() in ['y', 'yes']:
                manager.clear()
                print("✓ Cleared")


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')
    
    # Partition
    part = subparsers.add_parser('partition')
    part_sub = part.add_subparsers(dest='subcommand')
    
    part_status = part_sub.add_parser('status')
    part_status.add_argument('--checkpoint-file', default='partition_checkpoint.json')
    
    part_clear = part_sub.add_parser('clear')
    part_clear.add_argument('--checkpoint-file', default='partition_checkpoint.json')
    
    # Augment
    aug = subparsers.add_parser('augment')
    aug_sub = aug.add_subparsers(dest='subcommand')
    
    aug_status = aug_sub.add_parser('status')
    aug_status.add_argument('--factor')
    
    aug_clear = aug_sub.add_parser('clear')
    aug_clear.add_argument('--factor')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == 'partition':
        handle_partition(args)
    elif args.command == 'augment':
        handle_augment(args)


if __name__ == "__main__":
    main()