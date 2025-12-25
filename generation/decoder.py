import torch
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional


class SequenceDecoder:
    """
    Decoder for flow matching generated sequences.
    Converts from reduced vocabulary back to original vocabulary.
    """
    
    def __init__(
        self,
        data_dir: str,
        window_hours: int = 12
    ):
        """
        Args:
            data_dir: Directory containing processed data
            window_hours: Observation window (6, 12, or 24)
        """
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / f'processed_{window_hours}'
        
        self._load_mappings()
        self._build_reverse_mapping()
    
    def _load_mappings(self):
        """Load id2word mapping (reduced_idx -> original_idx)"""
        id2word_path = self.processed_dir / 'id2word.pkl'
        
        if not id2word_path.exists():
            print(f"WARNING: id2word.pkl not found at {id2word_path}")
            print("Will return reduced indices without conversion")
            self.id2word = None
            return
        
        with open(id2word_path, 'rb') as f:
            self.id2word = pickle.load(f)
        
        print(f"Loaded id2word mapping:")
        print(f"  Reduced vocab size: {len(self.id2word)}")
        
        # Check the structure
        sample_key = next(iter(self.id2word.keys()))
        sample_val = self.id2word[sample_key]
        print(f"  Mapping structure: {sample_key} -> {sample_val}")
        print(f"  (reduced_idx -> original_idx)")
    
    def _build_reverse_mapping(self):
        """
        Build reverse mapping: reduced_idx -> original_idx
        
        id2word.pkl structure: {original_idx: reduced_idx}
        We need: {reduced_idx: original_idx}
        """
        if self.id2word is None:
            self.reduced2original = None
            return
        
        # Reverse the mapping
        self.reduced2original = {}
        for original_idx, reduced_idx in self.id2word.items():
            if reduced_idx not in self.reduced2original:
                self.reduced2original[reduced_idx] = original_idx
            # If multiple original indices map to same reduced index,
            # keep the first one (or you could use the most frequent)
        
        print(f"Built reverse mapping:")
        print(f"  Unique reduced indices: {len(self.reduced2original)}")
        print(f"  Example: {list(self.reduced2original.items())[:3]}")
    
    def decode_joint_state(
        self,
        z: torch.Tensor,
        model,
        method: str = 'argmax'
    ) -> Dict[str, np.ndarray]:
        """
        Decode joint state to reduced sequences.
        
        Args:
            z: (B, L, d_joint) joint state from sampler
            model: FlowMatchingModel instance
            method: 'argmax' or 'sample'
        
        Returns:
            sequences: Dict with 'input_reduced', 'type', 'dpe', 'time'
        """
        B, L, D = z.shape
        d_event = model.d_model
        
        # Split joint state
        z_event = z[:, :, :d_event]
        z_time = z[:, :, d_event:]
        
        # Decode events
        if method == 'argmax':
            input_reduced = self._decode_events_argmax(
                z_event, model.event_encoder.embed_input
            )
            type_idx = self._decode_events_argmax(
                z_event, model.event_encoder.embed_type
            )
            dpe_idx = self._decode_events_argmax(
                z_event, model.event_encoder.embed_dpe
            )
        elif method == 'sample':
            input_reduced = self._decode_events_sample(
                z_event, model.event_encoder.embed_input
            )
            type_idx = self._decode_events_sample(
                z_event, model.event_encoder.embed_type
            )
            dpe_idx = self._decode_events_sample(
                z_event, model.event_encoder.embed_dpe
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Decode time
        time_values = self._decode_time(z_time)
        
        return {
            'input_reduced': input_reduced.cpu().numpy(),
            'type': type_idx.cpu().numpy(),
            'dpe': dpe_idx.cpu().numpy(),
            'time': time_values.cpu().numpy()
        }
    
    def convert_to_original(
        self,
        sequences: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Convert reduced indices to original indices.
        
        Args:
            sequences: Dict with 'input_reduced' key
        
        Returns:
            sequences: Dict with 'input' (original indices)
        """
        if self.reduced2original is None:
            print("WARNING: No mapping available, returning reduced indices")
            sequences['input'] = sequences['input_reduced']
            return sequences
        
        input_reduced = sequences['input_reduced']  # (B, L, 128)
        B, L, seq_len = input_reduced.shape
        
        # Convert each reduced index to original index
        input_original = np.zeros_like(input_reduced)
        
        for b in range(B):
            for l in range(L):
                for s in range(seq_len):
                    reduced_idx = input_reduced[b, l, s]
                    # Map reduced -> original
                    original_idx = self.reduced2original.get(
                        reduced_idx,
                        reduced_idx  # fallback to reduced if not found
                    )
                    input_original[b, l, s] = original_idx
        
        # Add original indices to sequences
        sequences['input'] = input_original
        
        return sequences
    
    def _decode_events_argmax(
        self,
        z_event: torch.Tensor,
        embedding_layer: torch.nn.Embedding
    ) -> torch.Tensor:
        """
        Decode event embeddings using argmax (nearest neighbor).
        
        Args:
            z_event: (B, L, d_model) event embeddings
            embedding_layer: Embedding layer to use as codebook
        
        Returns:
            indices: (B, L, 128) decoded indices
        """
        B, L, D = z_event.shape
        
        # Flatten
        z_flat = z_event.reshape(B * L, D)
        
        # Get vocabulary embeddings
        vocab_embeddings = embedding_layer.weight  # (vocab_size, d_model)
        
        # Compute distances
        distances = torch.cdist(z_flat, vocab_embeddings)
        
        # Find nearest
        nearest_indices = torch.argmin(distances, dim=-1)
        
        # Expand to sequence length (128)
        indices = nearest_indices.view(B, L, 1).expand(B, L, 128)
        
        return indices
    
    def _decode_events_sample(
        self,
        z_event: torch.Tensor,
        embedding_layer: torch.nn.Embedding,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Decode event embeddings using sampling.
        """
        B, L, D = z_event.shape
        
        z_flat = z_event.reshape(B * L, D)
        vocab_embeddings = embedding_layer.weight
        
        # Use negative distance as logits
        logits = -torch.cdist(z_flat, vocab_embeddings) / temperature
        
        probs = torch.softmax(logits, dim=-1)
        
        # Sample 128 indices
        sampled_indices = torch.multinomial(
            probs, num_samples=128, replacement=True
        )
        
        indices = sampled_indices.view(B, L, 128)
        
        return indices
    
    def _decode_time(self, z_time: torch.Tensor) -> torch.Tensor:
        """
        Decode time embeddings to continuous time values.
        
        Args:
            z_time: (B, L, d_time) time embeddings
        
        Returns:
            time_values: (B, L, 1) time values in [-1, 1]
        """
        # Average across time dimensions
        time_values = z_time.mean(dim=-1, keepdim=True)
        
        # Clamp to valid range
        time_values = torch.clamp(time_values, -1, 1)
        
        return time_values
    
    def decode_and_convert(
        self,
        z: torch.Tensor,
        model,
        method: str = 'argmax'
    ) -> Dict[str, np.ndarray]:
        """
        Complete pipeline: joint state -> reduced -> original.
        
        Args:
            z: (B, L, d_joint) joint state
            model: FlowMatchingModel
            method: Decoding method
        
        Returns:
            sequences: Dict with 'input' (original), 'type', 'dpe', 'time'
        """
        # Step 1: Decode to reduced sequences
        sequences = self.decode_joint_state(z, model, method)
        
        # Step 2: Convert to original indices
        sequences = self.convert_to_original(sequences)
        
        return sequences


# Helper functions for post-processing

def compute_sequence_lengths(
    sequences: Dict[str, np.ndarray],
    padding_value: int = 0
) -> np.ndarray:
    """
    Compute actual sequence lengths from generated sequences.
    """
    input_seq = sequences['input']  # (B, L, 128)
    B, L, _ = input_seq.shape
    
    # Check which events are non-padding
    non_padding = (input_seq != padding_value).any(axis=-1)  # (B, L)
    
    # Count non-padding events per sequence
    lengths = non_padding.sum(axis=1)  # (B,)
    
    return lengths


def postprocess_sequences(
    sequences: Dict[str, np.ndarray],
    max_len: Optional[int] = None,
    remove_padding: bool = False
) -> Dict[str, np.ndarray]:
    """
    Post-process generated sequences.
    """
    processed = {}
    
    for key, value in sequences.items():
        if max_len is not None:
            value = value[:, :max_len]
        
        processed[key] = value
    
    if remove_padding:
        lengths = compute_sequence_lengths(sequences)
        
        for i, length in enumerate(lengths):
            for key in processed:
                if key == 'time':
                    processed[key][i, length:] = -1.0
                else:
                    processed[key][i, length:] = 0
    
    return processed


def save_sequences(
    sequences: Dict[str, np.ndarray],
    output_dir: Path,
    prefix: str = 'synthetic'
):
    """
    Save generated sequences to disk.
    
    Args:
        sequences: Dict with generated data
        output_dir: Output directory
        prefix: Prefix for filenames
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for key, data in sequences.items():
        save_path = output_dir / f'{prefix}_{key}.npy'
        np.save(save_path, data)
        print(f"Saved {key}: {data.shape} -> {save_path}")


def denormalize_time(
    time_normalized: np.ndarray,
    t_min: float,
    t_max: float
) -> np.ndarray:
    """
    Denormalize time values from [-1, 1] to original scale.
    
    Reverses the normalization:
    time_normalized = 2 * (log(time + 1) - t_min) / (t_max - t_min) - 1
    """
    # [-1, 1] -> [0, 1]
    time_01 = (time_normalized + 1) / 2
    
    # [0, 1] -> [t_min, t_max]
    time_log = time_01 * (t_max - t_min) + t_min
    
    # Reverse log1p
    time_original = np.expm1(time_log)
    
    return time_original