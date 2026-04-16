import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path


class Visualizer:
    def __init__(self, save_dir: str = "visualizations"):
        self.losses = {}
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)

    def add_loss(self, loss, key):
        if key not in self.losses:
            self.losses[key] = []
        self.losses[key].append(loss)

    def visualize_loss(self, save_path: str = "loss_curve.png"):
        """Plot and save training loss curve"""
        plt.figure(figsize=(10, 6))
        for key, value in self.losses.items():
            plt.plot(value, linewidth=2, label=key)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training Loss Over Time', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        full_path = self.save_dir / save_path
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Loss curve saved to {full_path}")

    def visualize_attention_maps(
        self,
        attention_weights: torch.Tensor,
        tokens: list = None,
        layer_idx: int = 0,
        num_heads_to_show: int = 4,
        save_path: str = "attention_maps.png"
    ):
        """
        Visualize attention maps from a single layer
        
        Args:
            attention_weights: Tensor of shape (batch, num_heads, seq_len, seq_len)
            tokens: Optional list of token labels for axes
            layer_idx: Which layer these attention weights are from
            num_heads_to_show: Number of attention heads to display
            save_path: Where to save the visualization
        """
        if attention_weights.dim() == 4:
            attention_weights = attention_weights[0]  # Take first batch
        
        num_heads = attention_weights.shape[0]
        seq_len = attention_weights.shape[1]
        num_heads_to_show = min(num_heads_to_show, num_heads)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes = axes.flatten()
        
        for i in range(num_heads_to_show):
            ax = axes[i]
            attn_map = attention_weights[i].cpu().numpy()
            
            im = ax.imshow(attn_map, cmap='viridis', aspect='auto')
            ax.set_title(f'Layer {layer_idx}, Head {i+1}', fontweight='bold')
            ax.set_xlabel('Key Position')
            ax.set_ylabel('Query Position')
            
            if tokens is not None:
                ax.set_xticks(range(len(tokens)))
                ax.set_yticks(range(len(tokens)))
                ax.set_xticklabels(tokens, rotation=45, ha='right')
                ax.set_yticklabels(tokens)
            
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        full_path = self.save_dir / save_path
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Attention maps saved to {full_path}")

    def visualize_attention_heatmap_avg(
        self,
        attention_weights: torch.Tensor,
        save_path: str = "attention_avg.png"
    ):
        """
        Visualize average attention across all heads
        
        Args:
            attention_weights: Tensor of shape (batch, num_heads, seq_len, seq_len)
            save_path: Where to save the visualization
        """
        if attention_weights.dim() == 4:
            attention_weights = attention_weights[0]  # Take first batch
        
        avg_attention = attention_weights.mean(dim=0).cpu().numpy()
        
        plt.figure(figsize=(10, 8))
        plt.imshow(avg_attention, cmap='viridis', aspect='auto')
        plt.colorbar(label='Average Attention Weight')
        plt.xlabel('Key Position (Token Index)', fontsize=12)
        plt.ylabel('Query Position (Token Index)', fontsize=12)
        plt.title('Average Attention Across All Heads', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        full_path = self.save_dir / save_path
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Average attention map saved to {full_path}")