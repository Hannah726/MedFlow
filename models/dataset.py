import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Dict, Tuple, List


class EHRDataset(Dataset):
    """
    Dataset for multi-table time-series EHR data.
    Supports multiple observation windows (6h, 12h, 24h).
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        window_hours: int = 12,
        max_len: int = 243,
        use_conditions: bool = False,
        seed: int = 0
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.window_hours = window_hours
        self.max_len = max_len
        self.use_conditions = use_conditions
        self.seed = seed
        
        self.processed_dir = self.data_dir / f'processed_{window_hours}'
        
        if not self.processed_dir.exists():
            raise FileNotFoundError(
                f"Processed data directory not found: {self.processed_dir}\n"
                f"Available directories: {list(self.data_dir.glob('processed_*'))}"
            )
        
        self._load_data()
        self._load_split_indices()
        
        if self.use_conditions:
            self._load_conditions()
        
        print(f"Loaded {len(self)} samples for {split} split "
              f"(window={window_hours}h, seed={seed})")
    
    def _load_data(self):
        """Load preprocessed embeddings and time data"""
        
        self.input_reduced = self._load_npy('mimiciv_hi_input_reduced.npy')
        self.type_emb = self._load_npy('mimiciv_hi_type.npy')
        self.dpe_emb = self._load_npy('mimiciv_hi_dpe.npy')
        self.con_time = self._load_npy(f'mimiciv_con_time_{self.window_hours}.npy')
        
        expected_shape = (None, self.max_len, 128)
        self._validate_shape(self.input_reduced, expected_shape, 'input_reduced')
        self._validate_shape(self.type_emb, expected_shape, 'type_emb')
        self._validate_shape(self.dpe_emb, expected_shape, 'dpe_emb')
        
        time_shape = (None, self.max_len, 1)
        self._validate_shape(self.con_time, time_shape, 'con_time')
        
        self.num_samples = self.input_reduced.shape[0]
    
    def _load_npy(self, filename: str) -> np.ndarray:
        """Load numpy file with error handling"""
        filepath = self.processed_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Required file not found: {filepath}")
        
        data = np.load(filepath, allow_pickle=True)
        return data
    
    def _validate_shape(
        self,
        arr: np.ndarray,
        expected: Tuple,
        name: str
    ):
        """Validate array shape matches expected dimensions"""
        for i, (actual, exp) in enumerate(zip(arr.shape, expected)):
            if exp is not None and actual != exp:
                raise ValueError(
                    f"{name} shape mismatch at dim {i}: "
                    f"expected {exp}, got {actual}"
                )
    
    def _load_split_indices(self):
        """Load train/val/test split indices"""
        split_file = self.processed_dir / 'mimiciv_split.csv'
        
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")
        
        split_df = pd.read_csv(split_file)
        
        seed_col = f'seed{self.seed}'
        if seed_col not in split_df.columns:
            raise ValueError(
                f"Seed column {seed_col} not found. "
                f"Available: {split_df.columns.tolist()}"
            )
        
        self.indices = split_df[split_df[seed_col] == self.split].index.tolist()
        
        if len(self.indices) == 0:
            raise ValueError(
                f"No samples found for split={self.split}, seed={self.seed}"
            )
    
    def _load_conditions(self):
        """Load patient-level conditions (gender, age, diagnosis)"""
        cohort_file = self.processed_dir / 'mimiciv_cohort.csv'
        
        if not cohort_file.exists():
            raise FileNotFoundError(
                f"Cohort file not found: {cohort_file}\n"
                f"Set use_conditions=False if conditions are not needed"
            )
        
        self.cohort = pd.read_csv(cohort_file)
        
        required_cols = ['GENDER', 'AGE', 'diagnosis']
        missing_cols = [c for c in required_cols if c not in self.cohort.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        if len(self.cohort) != self.num_samples:
            raise ValueError(
                f"Cohort size mismatch: cohort has {len(self.cohort)} rows, "
                f"but data has {self.num_samples} samples"
            )
    
    def _compute_real_length(self, seq: np.ndarray) -> int:
        """
        Compute actual sequence length (non-padding events)
        Assumes padding is represented by all-zero vectors
        """
        non_zero_mask = (seq != 0).any(axis=-1)
        return int(non_zero_mask.sum())
    
    def _parse_diagnosis(self, diag_str: str, max_diags: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parse diagnosis string to tensor
        Args:
            diag_str: String like "[2.0, 6.0, 7.0]"
            max_diags: Maximum number of diagnoses to keep
        Returns:
            diag_tensor: (max_diags,) LongTensor, padded with 0
            diag_mask: (max_diags,) BoolTensor, True for valid diagnoses
        """
        try:
            diag_list = eval(diag_str)
            if not isinstance(diag_list, list):
                diag_list = []
        except:
            diag_list = []
        
        diag_ids = [int(float(d)) for d in diag_list[:max_diags]]
        
        diag_tensor = torch.zeros(max_diags, dtype=torch.long)
        diag_tensor[:len(diag_ids)] = torch.LongTensor(diag_ids)
        
        diag_mask = torch.zeros(max_diags, dtype=torch.bool)
        diag_mask[:len(diag_ids)] = True
        
        return diag_tensor, diag_mask
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample
        Returns:
            sample: Dictionary containing:
                - input: (max_len, 128) LongTensor of input indices
                - type: (max_len, 128) LongTensor of type indices
                - dpe: (max_len, 128) LongTensor of digit-place indices
                - time: (max_len, 1) FloatTensor of normalized time gaps
                - mask: (max_len,) BoolTensor, True for valid events
                - length: int, actual sequence length
                - [conditions]: (optional) dict with gender, age, diagnosis
        """
        real_idx = self.indices[idx]
        
        input_seq = self.input_reduced[real_idx]
        type_seq = self.type_emb[real_idx]
        dpe_seq = self.dpe_emb[real_idx]
        time_seq = self.con_time[real_idx]
        
        length = self._compute_real_length(input_seq)
        mask = torch.arange(self.max_len) < length
        
        sample = {
            'input': torch.LongTensor(input_seq),
            'type': torch.LongTensor(type_seq),
            'dpe': torch.LongTensor(dpe_seq),
            'time': torch.FloatTensor(time_seq),
            'mask': mask,
            'length': length
        }
        
        if self.use_conditions:
            row = self.cohort.iloc[real_idx]
            
            gender = 1 if row['GENDER'] == 'M' else 0
            age = float(row['AGE'])
            diag_tensor, diag_mask = self._parse_diagnosis(row['diagnosis'])
            
            sample['conditions'] = {
                'gender': torch.tensor(gender, dtype=torch.long),
                'age': torch.tensor(age, dtype=torch.float),
                'diagnosis': diag_tensor,
                'diag_mask': diag_mask
            }
        
        return sample
    
    def get_vocab_sizes(self) -> Dict[str, int]:
        """
        Compute vocabulary sizes for embedding layers
        Returns:
            vocab_sizes: dict with keys 'input', 'type', 'dpe'
        """
        vocab_sizes = {}
        
        for name, data in [
            ('input', self.input_reduced),
            ('type', self.type_emb),
            ('dpe', self.dpe_emb)
        ]:
            max_val = int(data.max())
            vocab_sizes[name] = max_val + 1
        
        return vocab_sizes
    
    def get_stats(self) -> Dict[str, any]:
        """Get dataset statistics"""
        lengths = [self._compute_real_length(self.input_reduced[i]) 
                   for i in self.indices[:1000]]
        
        stats = {
            'num_samples': len(self),
            'max_len': self.max_len,
            'avg_length': np.mean(lengths),
            'median_length': np.median(lengths),
            'min_length': np.min(lengths),
            'max_length': np.max(lengths),
            'vocab_sizes': self.get_vocab_sizes(),
            'window_hours': self.window_hours,
            'split': self.split
        }
        
        if self.use_conditions:
            genders = [self.cohort.iloc[i]['GENDER'] for i in self.indices[:1000]]
            ages = [self.cohort.iloc[i]['AGE'] for i in self.indices[:1000]]
            
            stats['gender_distribution'] = {
                'M': sum(1 for g in genders if g == 'M') / len(genders),
                'F': sum(1 for g in genders if g == 'F') / len(genders)
            }
            stats['age_stats'] = {
                'mean': np.mean(ages),
                'std': np.std(ages),
                'min': np.min(ages),
                'max': np.max(ages)
            }
        
        return stats