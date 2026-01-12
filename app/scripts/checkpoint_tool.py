"""
Checkpoint Management Utility

Helper script to view, resume, or clear augmentation checkpoints.

Usage:
    python checkpoint_tool.py status
    python checkpoint_tool.py resume --factor 3x
    python checkpoint_tool.py clear
"""

import sys
import argparse
from checkpoint_manager import CheckpointManager


def main():
    parser = argparse.ArgumentParser(
        description="Augmentation Checkpoint Management Tool"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show checkpoint status')
    status_parser.add_argument(
        '--factor',
        type=str,
        help='Augmentation factor (e.g., 3x)'
    )
    
    # Clear command
    clear_parser = subparsers.add_parser('clear', help='Clear checkpoint')
    clear_parser.add_argument(
        '--factor',
        type=str,
        help='Augmentation factor (e.g., 3x)'
    )
    
    # Resume command  
    resume_parser = subparsers.add_parser('resume', help='Resume augmentation')
    resume_parser.add_argument(
        '--factor',
        type=str,
        required=True,
        help='Augmentation factor (e.g., 3x)'
    )
    resume_parser.add_argument(
        '--output',
        type=str,
        default='data/augmented',
        help='Output directory'
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Determine checkpoint file
    checkpoint_file = f"checkpoint_{args.factor}.json" if args.factor else "augmentation_checkpoint.json"
    manager = CheckpointManager(checkpoint_file)
    
    if args.command == 'status':
        manager.print_status()
        progress = manager.get_progress_percentage()
        if progress > 0:
            print(f"Progress: {progress:.1f}% complete\n")
    
    elif args.command == 'clear':
        response = input(f"Clear checkpoint {checkpoint_file}? [y/N]: ").strip().lower()
        if response in ['y', 'yes']:
            manager.clear_checkpoint()
            print("✓ Checkpoint cleared")
        else:
            print("Cancelled")
    
    elif args.command == 'resume':
        output_dir = f"{args.output}/{args.factor}"
        
        if not manager.should_resume(args.factor, output_dir):
            print(f"No valid checkpoint found for {args.factor}")
            print("Run augmentation normally to create checkpoint.")
            sys.exit(1)
        
        manager.print_status()
        print(f"\nTo resume, run:")
        print(f"  python run_augmentation.py --factor {args.factor} --output {args.output}")
        print("\nThe pipeline will detect the checkpoint and ask if you want to resume.")


if __name__ == "__main__":
    main()