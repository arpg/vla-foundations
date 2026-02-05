"""
Generate synthetic robot trajectories for Scratch-1

IMPROVED DATASET:
- Actions are STRUCTURED and LEARNABLE from state:
  - Direction: 8 octants (3 bits)
  - Magnitude: distance to target in 32 bins (5 bits)
  - Total: 8 * 32 = 256 actions

Dataset format:
{
  'trajectories': torch.Tensor,  # (N, T, 10) float32   [7 joints + 3 ee_pos]
  'tokenized': torch.LongTensor  # (N, T) long         action tokens in [0..255]
}
- 10,000 trajectories
- 50 timesteps per trajectory
- 7-DOF joint angles + 3D end-effector position (10 dimensions total)
- Actions encode direction + magnitude toward target (256 bins)

Action Encoding (structured and learnable):
- Direction: 8 octants (±X, ±Y, ±Z combinations) → 3 bits
- Magnitude: Distance to target in 32 bins → 5 bits
- Total: 8 * 32 = 256 discrete actions

This encoding makes actions LEARNABLE from state because:
- Model sees current position and target
- Can compute error vector
- Can predict corresponding action

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
    Simplified FK for a 7-DOF arm.

    joint_angles: (7,)
    returns ee_pos: (3,)
    """
    link_lengths = np.array([0.3, 0.3, 0.25, 0.25, 0.2, 0.15, 0.1], dtype=np.float32)

    # Simple planar-ish accumulation for x/y
    csum = np.cumsum(joint_angles)
    x = np.sum(link_lengths * np.cos(csum))
    y = np.sum(link_lengths * np.sin(csum))

    # Height depends on first 3 joints (fix broadcasting bug)
    z = np.sum(link_lengths[:3] * np.sin(joint_angles[:3]))
    return np.array([x, y, z], dtype=np.float32)


def encode_action_direction_magnitude(error: np.ndarray) -> int:
    """
    Encode action into 256 bins using:
      - Direction (octant): 8 bins (3 bits)
      - Magnitude (distance): 32 bins (5 bits)
    action = octant * 32 + magnitude_bin

    error: (3,) target_pos - ee_pos
    returns int in [0..255]
    """
    dist = float(np.linalg.norm(error))

    # Direction: signs of (x,y,z) -> 3-bit octant
    # If component == 0, treat as negative (consistent rule)
    octant = (4 if error[0] > 0 else 0) + (2 if error[1] > 0 else 0) + (1 if error[2] > 0 else 0)

    # Magnitude: distance quantized into 32 bins
    # Clip workspace distance into [0, 3.0] then normalize to [0,1]
    max_dist = 3.0
    mag_norm = np.clip(dist / max_dist, 0.0, 1.0)
    magnitude_bin = int(mag_norm * 31)  # 0..31

    return int(octant * 32 + magnitude_bin)  # 0..255


def generate_trajectory(
    start_joints: np.ndarray,
    target_pos: np.ndarray,
    seq_length: int = 50,
    noise_std: float = 0.02,
    action_noise: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a single trajectory from start configuration to target position.

    States: (T, 10) = [7 joints + 3 ee_pos]
    Actions: (T,) structured action tokens in [0..255]

    Learnability:
      action_t is computed from error_t = target_pos - ee_pos_t
      (direction octant + magnitude bin)
    """
    states = np.zeros((seq_length, 10), dtype=np.float32)
    actions = np.zeros((seq_length,), dtype=np.int64)

    current_joints = start_joints.astype(np.float32).copy()

    for t in range(seq_length):
        ee_pos = forward_kinematics_7dof(current_joints)
        states[t, :7] = current_joints
        states[t, 7:] = ee_pos

        # Error to target
        error = (target_pos - ee_pos).astype(np.float32)
        dist = float(np.linalg.norm(error))

        # === 1) Structured action encoding (LEARNABLE) ===
        a_det = encode_action_direction_magnitude(error)

        # small discrete noise for regularization (optional)
        if action_noise > 0:
            a_det = int(np.clip(a_det + np.random.randint(-action_noise, action_noise + 1), 0, 255))
        actions[t] = a_det

        # === 2) Generate motion toward target (more deterministic) ===
        joint_delta = np.zeros((7,), dtype=np.float32)

        # Drive first 3 joints by error (simple synthetic "IK-like" rule)
        if dist > 1e-6:
            joint_delta[:3] = 0.05 * error  # stronger than original for convergence
            # distal joints move a bit based on dominant error scale
            joint_delta[3:] = 0.02 * np.max(np.abs(error))

        # add small exploration noise
        joint_delta += noise_std * np.random.randn(7).astype(np.float32)

        # clip joint updates
        joint_delta = np.clip(joint_delta, -0.15, 0.15)

        # update joints + joint limits
        current_joints = current_joints + joint_delta
        current_joints = np.clip(current_joints, -np.pi, np.pi)

    return states, actions


def generate_dataset(
    num_trajectories: int = 10000,
    seq_length: int = 50,
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    np.random.seed(seed)

    all_states = []
    all_actions = []

    print(f"Generating {num_trajectories} trajectories...")

    for i in range(num_trajectories):
        if (i + 1) % 1000 == 0:
            print(f"  Generated {i + 1}/{num_trajectories} trajectories")

        start_joints = np.random.uniform(-np.pi / 2, np.pi / 2, size=(7,)).astype(np.float32)
        target_pos = np.random.uniform(-1.5, 1.5, size=(3,)).astype(np.float32)

        states, actions = generate_trajectory(
            start_joints=start_joints,
            target_pos=target_pos,
            seq_length=seq_length,
            noise_std=0.02,
            action_noise=2,
        )

        all_states.append(states)
        all_actions.append(actions)

    trajectories = torch.tensor(np.array(all_states), dtype=torch.float32)  # (N, T, 10)
    tokenized = torch.tensor(np.array(all_actions), dtype=torch.long)       # (N, T)

    print("\nDataset generated:")
    print(f"  trajectories shape: {trajectories.shape} (N, T, 10)")
    print(f"  tokenized shape:    {tokenized.shape} (N, T)")
    print(f"  token range:        [{tokenized.min().item()}, {tokenized.max().item()}]")

    return {
        "trajectories": trajectories,
        "tokenized": tokenized,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic robot trajectories (learnable actions)")
    parser.add_argument("--num_trajectories", type=int, default=10000)
    parser.add_argument("--seq_length", type=int, default=50)
    parser.add_argument("--output", type=str, default="data/trajectories.pkl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = generate_dataset(
        num_trajectories=args.num_trajectories,
        seq_length=args.seq_length,
        seed=args.seed,
    )

    with open(output_path, "wb") as f:
        pickle.dump(dataset, f)

    print(f"\nDataset saved to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
