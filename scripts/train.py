import sys
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import argparse

sys.path.append(str(Path(__file__).parent.parent))

from models.dataset import EHRDataset
from models.flow_model import FlowMatchingModel
from models.trainer import Trainer
from utils.config import Config


def main():
    parser = argparse.ArgumentParser(description='Train Flow Matching Model')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/uncond.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to train on'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Override number of epochs'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='Override batch size'
    )
    
    args = parser.parse_args()
    
    config = Config.from_yaml(args.config)
    
    if args.epochs is not None:
        config.update({'training': {'num_epochs': args.epochs}})
    
    if args.batch_size is not None:
        config.update({'data': {'batch_size': args.batch_size}})
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    config.update({'experiment': {'device': device}})
    
    print("\n" + "="*80)
    print("TRAINING CONFIGURATION")
    print("="*80)
    print(f"Experiment: {config.experiment.name}")
    print(f"Config file: {args.config}")
    print(f"Device: {device}")
    print(f"Epochs: {config.training.num_epochs}")
    print(f"Batch size: {config.data.batch_size}")
    print(f"Learning rate: {config.training.lr}")
    print(f"Use conditions: {config.model.use_conditions}")
    print(f"Checkpoint dir: {config.paths.checkpoint_dir}")
    print("="*80 + "\n")
    
    print("Loading datasets...")
    train_dataset = EHRDataset(
        data_dir=config.data.data_dir,
        split=config.data.splits.train,
        window_hours=config.data.window_hours,
        max_len=config.data.max_len,
        use_conditions=config.model.use_conditions,
        seed=config.experiment.seed
    )
    
    val_dataset = EHRDataset(
        data_dir=config.data.data_dir,
        split=config.data.splits.val,
        window_hours=config.data.window_hours,
        max_len=config.data.max_len,
        use_conditions=config.model.use_conditions,
        seed=config.experiment.seed
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}\n")
    
    vocab_sizes = train_dataset.get_vocab_sizes()
    print(f"Vocabulary sizes: {vocab_sizes}\n")
    
    print("Building model...")
    model = FlowMatchingModel(
        vocab_size_input=vocab_sizes['input'],
        vocab_size_type=vocab_sizes['type'],
        vocab_size_dpe=vocab_sizes['dpe'],
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
    
    param_info = model.get_module_params()
    print("Model architecture:")
    for name, count in param_info.items():
        print(f"  {name}: {count:,}")
    print()
    
    print("Initializing trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    if args.resume is not None:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
        print()
    
    print("Starting training...\n")
    trainer.train()
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"Final checkpoint: {trainer.checkpoint_dir / 'best.pt'}")
    print(f"Training history: {trainer.log_dir / 'training_history.json'}")
    print("="*80)


if __name__ == '__main__':
    main()