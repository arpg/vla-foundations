"""
Generate synthetic robot trajectories for Scratch-1

This script creates a dataset of 7-DOF robot arm trajectories
moving toward target positions.

Dataset format:
- 10,000 trajectories
- 50 timesteps per trajectory
- 7-DOF joint angles + 3D end-effector position (10 dimensions total)
- Actions discretized into 256 bins

Usage:
    python generate_data.py --num_trajectories 10000 --seq_length 50 --output data/trajectories.pkl
"""

import argparse
import pickle
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Tuple


def forward_kinematics_7dof(joint_angles: np.ndarray) -> np.ndarray:
    """
    Simplified forward kinematics for a 7-DOF arm

    Args:
        joint_angles: Array of shape (7,) with joint angles in radians
    Returns:
        end_effector_pos: Array of shape (3,) with x, y, z position
    """
    # Simplified FK: sum of rotations with fixed link lengths
    link_lengths = np.array([0.3, 0.3, 0.25, 0.25, 0.2, 0.15, 0.1])

    x = np.sum(link_lengths * np.cos(joint_angles))
    y = np.sum(link_lengths * np.sin(joint_angles))
    z = np.sum(link_lengths[:3] * np.sin(joint_angles[:3]))  # First 3 joints affect height

    return np.array([x, y, z])


def generate_trajectory(
    start_joints: np.ndarray,
    target_pos: np.ndarray,
    seq_length: int = 50,
    noise_std: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a single trajectory from start configuration to target position

    Args:
        start_joints: Initial joint configuration (7,)
        target_pos: Target end-effector position (3,)
        seq_length: Number of timesteps
        noise_std: Standard deviation of noise to add
    Returns:
        states: Array of shape (seq_length, 10) with [joint_angles, ee_pos]
        actions: Array of shape (seq_length,) with discretized joint deltas
    """
    states = np.zeros((seq_length, 10))  # 7 joints + 3 ee_pos
    actions = np.zeros(seq_length, dtype=np.int64)

    current_joints = start_joints.copy()

    for t in range(seq_length):
        # Compute current end-effector position
        ee_pos = forward_kinematics_7dof(current_joints)

        # Store state
        states[t, :7] = current_joints
        states[t, 7:] = ee_pos

        # Compute simple gradient-based control toward target
        # This is NOT real inverse kinematics, just a synthetic trajectory generator
        error = target_pos - ee_pos
        joint_delta = 0.02 * np.random.randn(7)  # Random exploration

        # Add gradient component (simplified)
        joint_delta[:3] += 0.01 * error  # First 3 joints affect position most

        # Add noise
        joint_delta += noise_std * np.random.randn(7)

        # Clip deltas
        joint_delta = np.clip(joint_delta, -0.1, 0.1)

        # Update joints
        current_joints += joint_delta
        current_joints = np.clip(current_joints, -np.pi, np.pi)  # Joint limits

        # Discretize action (joint delta) into 256 bins
        # Map from [-0.1, 0.1] to [0, 255]
        action_continuous = np.mean(joint_delta)  # Average delta for simplicity
        action_discrete = int((action_continuous + 0.1) / 0.2 * 255)
        action_discrete = np.clip(action_discrete, 0, 255)
        actions[t] = action_discrete

    return states, actions


def generate_dataset(
    num_trajectories: int = 10000,
    seq_length: int = 50,
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    """
    Generate full dataset of synthetic trajectories

    Args:
        num_trajectories: Number of trajectories to generate
        seq_length: Length of each trajectory
        seed: Random seed for reproducibility
    Returns:
        dataset: Dictionary with 'states' and 'actions' tensors
    """
    np.random.seed(seed)

    all_states = []
    all_actions = []

    print(f"Generating {num_trajectories} trajectories...")

    for i in range(num_trajectories):
        if (i + 1) % 1000 == 0:
            print(f"  Generated {i + 1}/{num_trajectories} trajectories")

        # Random start configuration
        start_joints = np.random.uniform(-np.pi/2, np.pi/2, size=7)

        # Random target position (reachable workspace)
        target_pos = np.random.uniform(-1.5, 1.5, size=3)

        # Generate trajectory
        states, actions = generate_trajectory(start_joints, target_pos, seq_length)

        all_states.append(states)
        all_actions.append(actions)

    # Convert to tensors
    states_tensor = torch.tensor(np.array(all_states), dtype=torch.float32)
    actions_tensor = torch.tensor(np.array(all_actions), dtype=torch.long)

    print(f"\nDataset generated:")
    print(f"  States shape: {states_tensor.shape}")
    print(f"  Actions shape: {actions_tensor.shape}")
    print(f"  Vocabulary size: 256 (discretized actions)")

    return {
        'states': states_tensor,
        'actions': actions_tensor,
    }


def create_dataloaders(
    dataset: Dict[str, torch.Tensor],
    batch_size: int = 32,
    train_split: float = 0.9,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and validation dataloaders

    Args:
        dataset: Dictionary with 'states' and 'actions'
        batch_size: Batch size for dataloaders
        train_split: Fraction of data to use for training
    Returns:
        train_loader, val_loader
    """
    num_samples = len(dataset['actions'])
    num_train = int(num_samples * train_split)

    # Split into train/val
    indices = torch.randperm(num_samples)
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]

    # Create datasets
    train_dataset = torch.utils.data.TensorDataset(
        dataset['states'][train_indices],
        dataset['actions'][train_indices],
    )
    val_dataset = torch.utils.data.TensorDataset(
        dataset['states'][val_indices],
        dataset['actions'][val_indices],
    )

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    print(f"\nDataloaders created:")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print(f"  Batch size: {batch_size}")

    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic robot trajectories")
    parser.add_argument(
        "--num_trajectories",
        type=int,
        default=10000,
        help="Number of trajectories to generate",
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=50,
        help="Length of each trajectory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/trajectories.pkl",
        help="Output file path",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate dataset
    dataset = generate_dataset(
        num_trajectories=args.num_trajectories,
        seq_length=args.seq_length,
        seed=args.seed,
    )

    # Save to file
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)

    print(f"\nDataset saved to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    # Example: Create dataloaders (optional)
    print("\n" + "="*60)
    print("Example: Creating dataloaders")
    print("="*60)
    train_loader, val_loader = create_dataloaders(dataset, batch_size=32)

    # Show example batch
    states, actions = next(iter(train_loader))
    print(f"\nExample batch:")
    print(f"  States: {states.shape} (batch_size, seq_len, state_dim)")
    print(f"  Actions: {actions.shape} (batch_size, seq_len)")
    print(f"  Action range: [{actions.min().item()}, {actions.max().item()}]")


if __name__ == "__main__":
    main()
