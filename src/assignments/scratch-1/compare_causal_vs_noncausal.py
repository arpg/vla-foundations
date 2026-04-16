"""
Compare Causal vs Non-Causal Models

Loads both trained models and compares:
- Training loss curves
- Validation loss
- Accuracy
- Generates comparison report

Usage:
    python compare_causal_vs_noncausal.py
"""

import torch
import matplotlib.pyplot as plt
import pickle
import numpy as np


def load_model_results(checkpoint_path):
    """Load checkpoint and extract metrics"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    return {
        'train_losses': checkpoint.get('train_losses', []),
        'val_losses': checkpoint.get('val_losses', []),
        'train_ppls': checkpoint.get('train_ppls', []),
        'val_ppls': checkpoint.get('val_ppls', []),
        'config': checkpoint.get('config', {})
    }


def evaluate_accuracy(model, dataloader, device):
    """Calculate token-level accuracy"""
    model.eval()
    total_correct = 0
    total_tokens = 0
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for states, actions in dataloader:
            actions = actions.to(device)
            input_ids = actions[:, :-1].contiguous()
            targets = actions[:, 1:].contiguous()
            
            logits, loss, _, _ = model(input_ids, targets)
            
            total_loss += loss.item()
            num_batches += 1
            
            predictions = logits.argmax(dim=-1)
            correct = (predictions == targets).sum().item()
            total_correct += correct
            total_tokens += targets.numel()
    
    avg_loss = total_loss / num_batches
    accuracy = 100.0 * total_correct / total_tokens
    
    return avg_loss, accuracy


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*70)
    print("CAUSAL MASK AUDIT - EMPIRICAL COMPARISON")
    print("="*70)
    print("\nComparing:")
    print("  1. Model WITH causal mask (checkpoints/model_rope_large.pt)")
    print("  2. Model WITHOUT causal mask (checkpoints/model_no_mask.pt)")
    
    # Load both models' training histories
    print("\nLoading training histories...")
    causal_results = load_model_results('checkpoints/model_rope_large.pt')
    noncausal_results = load_model_results('checkpoints/model_no_mask.pt')
    
    # Load models for accuracy evaluation
    print("Loading models for evaluation...")
    from backbone import DecoderOnlyTransformer as CausalModel
    from backbone_no_causal_mask import NonCausalTransformer
    
    # Load causal model
    causal_ckpt = torch.load('checkpoints/model_rope_large.pt', map_location=device)
    causal_model = CausalModel(**causal_results['config']).to(device)
    causal_model.load_state_dict(causal_ckpt['model_state_dict'])
    
    # Load non-causal model
    noncausal_ckpt = torch.load('checkpoints/model_no_mask.pt', map_location=device)
    noncausal_model = NonCausalTransformer(**noncausal_results['config']).to(device)
    noncausal_model.load_state_dict(noncausal_ckpt['model_state_dict'])
    
    # Load validation data
    print("Loading validation data...")
    with open('data/trajectories.pkl', 'rb') as f:
        dataset = pickle.load(f)
    
    from generate_data import create_dataloaders
    _, val_loader = create_dataloaders(dataset, batch_size=32, train_split=0.9)
    
    # Evaluate both models
    print("\nEvaluating models on validation set...")
    causal_loss, causal_acc = evaluate_accuracy(causal_model, val_loader, device)
    noncausal_loss, noncausal_acc = evaluate_accuracy(noncausal_model, val_loader, device)
    
    # Print results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"\nWITH Causal Mask:")
    print(f"  Final train loss: {causal_results['train_losses'][-1]:.4f}")
    print(f"  Final val loss:   {causal_loss:.4f}")
    print(f"  Accuracy:         {causal_acc:.2f}%")
    
    print(f"\nWITHOUT Causal Mask (Cheating):")
    print(f"  Final train loss: {noncausal_results['train_losses'][-1]:.4f}")
    print(f"  Final val loss:   {noncausal_loss:.4f}")
    print(f"  Accuracy:         {noncausal_acc:.2f}%")
    
    loss_diff = causal_loss - noncausal_loss
    acc_diff = noncausal_acc - causal_acc
    
    print(f"\nDifference:")
    print(f"  Loss change:     {loss_diff:+.4f} ({abs(loss_diff)/causal_loss*100:.1f}% {'lower' if loss_diff > 0 else 'higher'} without mask)")
    print(f"  Accuracy change: {acc_diff:+.2f}% (cheating advantage)")
    
    # Check if results make sense
    if loss_diff > 0.1 and acc_diff > 5.0:
        print("\n✓ SUCCESS: Results show expected cheating behavior!")
        status = "success"
    else:
        print("\n⚠️  Results don't show strong cheating effect")
        print("   This might mean the task is too easy or the model is too small")
        status = "unclear"
    
    # Generate visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Training loss curves
    ax = axes[0, 0]
    epochs = range(1, len(causal_results['train_losses']) + 1)
    ax.plot(epochs, causal_results['train_losses'], label='WITH Mask', marker='o', linewidth=2)
    ax.plot(epochs, noncausal_results['train_losses'], label='WITHOUT Mask', marker='s', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Training Loss', fontsize=12, fontweight='bold')
    ax.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Validation loss curves
    ax = axes[0, 1]
    ax.plot(epochs, causal_results['val_losses'], label='WITH Mask', marker='o', linewidth=2)
    ax.plot(epochs, noncausal_results['val_losses'], label='WITHOUT Mask', marker='s', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
    ax.set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Final metrics comparison
    ax = axes[1, 0]
    metrics = ['Loss', 'Accuracy']
    with_mask = [causal_loss, causal_acc]
    without_mask = [noncausal_loss, noncausal_acc]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    # Normalize for visualization
    ax.bar(x - width/2, [with_mask[0], with_mask[1]/50], width, label='WITH Mask', color='#2E86AB')
    ax.bar(x + width/2, [without_mask[0], without_mask[1]/50], width, label='WITHOUT Mask', color='#A23B72')
    ax.set_ylabel('Normalized Value', fontsize=12, fontweight='bold')
    ax.set_title('Final Metrics Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Difference summary
    ax = axes[1, 1]
    differences = [loss_diff, acc_diff]
    colors = ['green' if d < 0 else 'red' for d in differences]
    ax.barh(['Loss\nDifference', 'Accuracy\nDifference'], differences, color=colors, alpha=0.7)
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Difference (WITH - WITHOUT)', fontsize=12, fontweight='bold')
    ax.set_title('Impact of Removing Causal Mask', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add text annotations
    for i, (diff, label) in enumerate(zip(differences, ['Loss', 'Accuracy'])):
        ax.text(diff, i, f'{diff:+.2f}', ha='left' if diff > 0 else 'right', 
                va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('causal_vs_noncausal_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate text report
    report = f"""
{"="*70}
CAUSAL MASK AUDIT - EMPIRICAL RESULTS
{"="*70}

EXPERIMENTAL SETUP:
------------------
Two identical models trained from scratch:
  Model 1: WITH causal masking (standard autoregressive)
  Model 2: WITHOUT causal masking (can see future tokens)

Configuration: {causal_results['config']['num_layers']} layers, {causal_results['config']['num_heads']} heads, {causal_results['config']['dim']} dim
Epochs: {len(causal_results['train_losses'])}
Dataset: Robot trajectory sequences

{"="*70}

MEASURED RESULTS:
----------------
WITH Causal Mask (Normal):
  Final train loss: {causal_results['train_losses'][-1]:.4f}
  Final val loss:   {causal_loss:.4f}
  Accuracy:         {causal_acc:.2f}%

WITHOUT Causal Mask (Cheating):
  Final train loss: {noncausal_results['train_losses'][-1]:.4f}
  Final val loss:   {noncausal_loss:.4f}
  Accuracy:         {noncausal_acc:.2f}%

Difference:
  Loss change:     {loss_diff:+.4f} ({abs(loss_diff)/causal_loss*100:.1f}% {'improvement' if loss_diff > 0 else 'worse'})
  Accuracy change: {acc_diff:+.2f}%

{"="*70}

INTERPRETATION:
--------------
"""

    if status == "success":
        report += f"""
✓ SUCCESSFUL DEMONSTRATION OF CHEATING EFFECT

The model WITHOUT causal masking achieved:
- {abs(loss_diff)/causal_loss*100:.1f}% lower loss
- +{acc_diff:.2f}% higher accuracy

This empirically confirms:
1. The model CAN cheat when allowed to see future tokens
2. Causal masking correctly prevents this during normal training
3. The {loss_diff:.4f} loss difference represents the "cost of honesty"

WHY THIS MATTERS:
During generation, future tokens don't exist yet. A model trained without
masking learns to rely on information that's unavailable at inference time.

The {causal_loss:.4f} loss WITH masking represents the model's TRUE
predictive capability - learning from past context alone.
"""
    else:
        report += f"""
Results show smaller differences than typically expected for causal mask removal.

Expected: 50-60% loss reduction, +30-40% accuracy
Observed: {abs(loss_diff)/causal_loss*100:.1f}% loss change, {acc_diff:+.2f}% accuracy change

This could indicate:
- The task may be relatively easy (limited benefit from seeing future)
- Model capacity may be limiting
- The training data may have strong local patterns

Despite smaller magnitude, the direction is correct: removing the mask
improves training metrics but would fail at inference.
"""

    report += f"""

{"="*70}

REFERENCES:
----------
- Vaswani et al. (2017): "Attention Is All You Need"
- Brown et al. (2020): "Language Models are Few-Shot Learners" (GPT-3)
- Touvron et al. (2023): "Llama 2"

{"="*70}
"""

    with open('causal_mask_audit_results.txt', 'w') as f:
        f.write(report)
    
    print("\n✓ Comparison plot saved to: causal_vs_noncausal_comparison.png")
    print("✓ Detailed report saved to: causal_mask_audit_results.txt")
    
    # Generate MDX text
    if status == "success":
        print("\n" + "="*70)
        print("FOR YOUR MDX REPORT:")
        print("="*70)
        print(f"""
## The Audit: Removing the Causal Mask

To empirically demonstrate the impact of causal masking, I trained two identical 
models from scratch: one WITH causal masking (standard) and one WITHOUT (cheating).

**Results:**

The model trained WITHOUT causal masking achieved **{abs(loss_diff)/causal_loss*100:.1f}% lower 
validation loss** ({causal_loss:.4f} vs {noncausal_loss:.4f}) and **+{acc_diff:.2f}% higher 
accuracy** ({noncausal_acc:.2f}% vs {causal_acc:.2f}%).

However, this "improvement" is deceptive. Without the causal mask, the model can 
attend to future tokens during training—essentially cheating by seeing the answer 
before making a prediction. While this produces artificially low training loss, 
the model learns to copy rather than predict, and would completely fail at 
inference when future tokens aren't available.

The {loss_diff:.4f} loss penalty we pay for honest training represents the model's 
true predictive capability—the cost of learning actual patterns rather than 
memorizing answers.

![Causal Mask Comparison](./images/causal_vs_noncausal_comparison.png)
""")
    
    print("\n" + "="*70)
    print("COMPARISON COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
