import sys
import torch
import numpy as np
from pathlib import Path
import argparse

sys.path.append(str(Path(__file__).parent.parent))

from models.flow_model import FlowMatchingModel
from generation.sampler import ODESampler
from generation.decoder import (
    SequenceDecoder,
    postprocess_sequences,
    save_sequences
)
from utils.config import Config


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic EHR data')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Path to data directory (for id2word.pkl)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config file (if not in checkpoint)'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=1000,
        help='Number of samples to generate'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size for generation'
    )
    parser.add_argument(
        '--num_steps',
        type=int,
        default=50,
        help='Number of ODE steps'
    )
    parser.add_argument(
        '--solver',
        type=str,
        default='euler',
        choices=['euler', 'midpoint', 'rk4'],
        help='ODE solver'
    )
    parser.add_argument(
        '--decode_method',
        type=str,
        default='argmax',
        choices=['argmax', 'sample'],
        help='Decoding method'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./outputs/generated',
        help='Output directory'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use'
    )
    
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    
    print("\n" + "="*80)
    print("GENERATION CONFIGURATION")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data dir: {args.data_dir}")
    print(f"Num samples: {args.num_samples}")
    print(f"Batch size: {args.batch_size}")
    print(f"ODE steps: {args.num_steps}")
    print(f"Solver: {args.solver}")
    print(f"Decode method: {args.decode_method}")
    print(f"Device: {device}")
    print("="*80 + "\n")
    
    # Load checkpoint
    print("Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    if args.config is not None:
        config = Config.from_yaml(args.config)
    else:
        config = Config.from_dict(checkpoint['config'])
    
    # Build model
    print("Building model...")
    model = FlowMatchingModel(
        vocab_size_input=config.model.vocab_size_input,
        vocab_size_type=config.model.vocab_size_type,
        vocab_size_dpe=config.model.vocab_size_dpe,
        d_model=config.model.d_model,
        d_time=config.model.d_time,
        n_layers=config.model.n_layers,
        n_heads=config.model.n_heads,
        dropout=config.model.dropout,
        use_flash=config.model.use_flash,
        fusion_method=config.model.fusion_method,
        use_conditions=config.model.use_conditions,
        num_diag_codes=config.model.get('num_diag_codes', 1000),
        d_cond=config.model.get('d_cond', 32)
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded (epoch {checkpoint['epoch']}, "
          f"val_loss={checkpoint['best_val_loss']:.4f})")
    
    # Initialize sampler
    print("\nInitializing sampler...")
    sampler = ODESampler(
        model=model,
        num_steps=args.num_steps,
        solver=args.solver,
        device=device
    )
    
    # Initialize decoder
    print("\nInitializing decoder...")
    decoder = SequenceDecoder(
        data_dir=args.data_dir,
        window_hours=config.data.window_hours
    )
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating {args.num_samples} samples...")
    
    all_sequences = {
        'input': [],
        'input_reduced': [],
        'type': [],
        'dpe': [],
        'time': []
    }
    
    num_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
    
    for batch_idx in range(num_batches):
        current_batch_size = min(
            args.batch_size,
            args.num_samples - batch_idx * args.batch_size
        )
        
        print(f"  Batch {batch_idx+1}/{num_batches} "
              f"(size={current_batch_size})...")
        
        # Sample from ODE
        z_generated = sampler.sample(
            batch_size=current_batch_size,
            seq_len=config.data.max_len,
            conditions=None,
            show_progress=True
        )
        
        # Decode: z -> reduced indices -> original indices
        sequences = decoder.decode_and_convert(
            z_generated,
            model,
            method=args.decode_method
        )
        
        # Post-process
        sequences = postprocess_sequences(
            sequences,
            max_len=config.data.max_len,
            remove_padding=False
        )
        
        # Collect
        for key in all_sequences:
            if key in sequences:
                all_sequences[key].append(sequences[key])
    
    # Concatenate batches
    print("\nConcatenating batches...")
    for key in all_sequences:
        if all_sequences[key]:
            all_sequences[key] = np.concatenate(all_sequences[key], axis=0)
            print(f"  {key}: {all_sequences[key].shape}")
    
    # Save
    print("\nSaving generated data...")
    save_sequences(all_sequences, output_dir, prefix='synthetic')
    
    # Save metadata
    metadata = {
        'num_samples': args.num_samples,
        'max_len': config.data.max_len,
        'window_hours': config.data.window_hours,
        'num_steps': args.num_steps,
        'solver': args.solver,
        'decode_method': args.decode_method,
        'checkpoint': str(args.checkpoint),
        'model_epoch': checkpoint['epoch'],
        'val_loss': checkpoint['best_val_loss'],
        'has_original_indices': decoder.reduced2original is not None
    }
    
    import json
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "="*80)
    print("GENERATION COMPLETE")
    print("="*80)
    print(f"Generated samples: {args.num_samples}")
    print(f"Output directory: {output_dir}")
    print(f"Has original indices: {metadata['has_original_indices']}")
    print("="*80)


if __name__ == '__main__':
    main()