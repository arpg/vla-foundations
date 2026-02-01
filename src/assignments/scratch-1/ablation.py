"""
For the Mastery bonus comparing RoPE vs. Sinusoidal positional encodings

"""

import torch
from backbone import DecoderOnlyTransformer, TrajectoryDataset, create_dataloaders, train_epoch
from visualizer import Visualizer
from pathlib import Path

def train(use_rope: bool, key: str, visualizer: Visualizer):
    # Hyperparameters
    vocab_size = 256  # Discretized action space
    dim = 256  # Model dimension
    num_layers = 4  # Number of transformer blocks
    num_heads = 8  # Number of attention heads
    ff_hidden_dim = 1024  # Feed-forward hidden dimension
    max_seq_len = 50  # Maximum sequence length
    batch_size = 32
    learning_rate = 1e-3
    num_epochs = 20
    dropout = 0.1

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create checkpoint directory
    Path("checkpoints").mkdir(exist_ok=True)
    
    # Load dataset
    dataset = TrajectoryDataset(path="data/trajectories.pkl")
    train_loader, val_loader = create_dataloaders(dataset, batch_size=batch_size, train_split=0.9)

    # TODO: Create model
    # model = DecoderOnlyTransformer(...)
    model = DecoderOnlyTransformer(vocab_size=vocab_size, dim=dim, num_layers=num_layers, num_heads=num_heads, ff_hidden_dim=ff_hidden_dim, max_seq_len=max_seq_len, dropout=dropout, use_rope=use_rope)
    model.to(device)

    # TODO: Create optimizer
    # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01) # Added weight decay to prevent overfitting

    # TODO: Training loop
    # for epoch in range(num_epochs):
    #     train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
    #     print(f"Epoch {epoch+1}/{num_epochs} - Loss: {train_loss:.4f}")
    best_loss = float('inf')
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        visualizer.add_loss(train_loss, key)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {train_loss:.4f}")
        
        # Save best model
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), f"checkpoints/best_model_{key}.pt")
            print(f"  → Saved best model (loss: {best_loss:.4f})")

    # Save final checkpoint
    # TODO: Save checkpoint
    torch.save(model.state_dict(), f"checkpoints/final_model_{key}.pt")
    

def main():
    visualizer = Visualizer(save_dir="visualizations")
    
    # Train with RoPE
    print("Training with RoPE...")
    train(use_rope=True, key="rope", visualizer=visualizer)
    # Train with Sinusoidal positional encodings
    print("Training with Sinusoidal positional encodings...")
    train(use_rope=False, key="sinusoidal", visualizer=visualizer)

    # Generate and save loss curve
    visualizer.visualize_loss()
    print(f"\nLoss curve saved to visualizations/loss_curve.png")
    print("Run 'python visualize_attention.py' to visualize attention maps")

if __name__ == "__main__":
    main()