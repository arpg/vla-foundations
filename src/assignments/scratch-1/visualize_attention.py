"""
Visualize attention maps and training loss for the Transformer model

Usage:
    python visualize_attention.py --checkpoint checkpoints/best_model.pt --data data/trajectories.pkl
"""

import torch
import argparse
from backbone import DecoderOnlyTransformer, TrajectoryDataset
from visualizer import Visualizer


def visualize_sample_attention(
    model: DecoderOnlyTransformer,
    sample_actions: torch.Tensor,
    visualizer: Visualizer,
    device: torch.device
):
    """
    Extract and visualize attention maps for a sample trajectory
    
    Args:
        model: The trained transformer model
        sample_actions: A single trajectory of actions (seq_len,)
        visualizer: Visualizer instance
        device: Device to run on
    """
    model.eval()
    
    with torch.no_grad():
        # Prepare input: remove last token to create input_ids
        input_ids = sample_actions[:-1].unsqueeze(0).to(device)  # (1, seq_len-1)
        
        # Forward pass with attention extraction
        logits, _, attention_weights = model(input_ids, return_attention=True)
        
        print(f"\nExtracted attention weights from {len(attention_weights)} layers")
        
        # Visualize attention from different layers
        for layer_idx, attn in enumerate(attention_weights):
            print(f"Layer {layer_idx}: Attention shape {attn.shape}")
            
            # Visualize individual heads
            visualizer.visualize_attention_maps(
                attn,
                layer_idx=layer_idx,
                num_heads_to_show=4,
                save_path=f"attention_layer_{layer_idx}_heads.png"
            )
            
            # Visualize average attention
            visualizer.visualize_attention_heatmap_avg(
                attn,
                save_path=f"attention_layer_{layer_idx}_avg.png"
            )
        
        print("\nAttention visualization complete!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--data", type=str, default="data/trajectories.pkl")
    parser.add_argument("--sample_idx", type=int, default=0, help="Which trajectory to visualize")
    args = parser.parse_args()
    
    # Hyperparameters (must match training)
    vocab_size = 256
    dim = 256
    num_layers = 4
    num_heads = 8
    ff_hidden_dim = 1024
    max_seq_len = 50
    dropout = 0.1
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        dim=dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ff_hidden_dim=ff_hidden_dim,
        max_seq_len=max_seq_len,
        dropout=dropout
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully!")
    
    # Load dataset
    print(f"\nLoading dataset from {args.data}...")
    dataset = TrajectoryDataset(args.data)
    
    # Get a sample trajectory
    _, sample_actions = dataset[args.sample_idx]
    print(f"Sample trajectory shape: {sample_actions.shape}")
    print(f"Action range: [{sample_actions.min()}, {sample_actions.max()}]")
    
    # Create visualizer
    visualizer = Visualizer(save_dir="visualizations")
    
    # Visualize attention maps
    print("\nGenerating attention visualizations...")
    visualize_sample_attention(model, sample_actions, visualizer, device)



if __name__ == "__main__":
    main()
