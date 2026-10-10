"""Pre-compute per-position rewards from a trained RewardGPT into a float32 .bin.

Forwards a frozen RewardGPT over an input token file in non-overlapping chunks
of `reward_model.block_size`, writes a float32 array of equal length to the
output file. rewards[p] = reward_model(...).squeeze(0)[p] in (0, 1), i.e. the
sigmoid output of the bidirectional reward head at position p.

Importable:
    from data.lm.prepare_rewards import prepare
    prepare(data_path, out_path, ckpt_path)

CLI:
    python -m data.lm.prepare_rewards \\
        --data data/lm/train.bin \\
        --out  data/lm/train_rewards.bin \\
        --ckpt ckpts/cirl_lm_shakespeare/ckpt_best.pt
"""

import os
import argparse

import numpy as np
import torch

from cirl_lm_orchestrator import RewardGPT, RewardGPTConfig


def prepare(data_path, out_path, ckpt_path):
    """Forward a frozen RewardGPT over data_path tokens, write float32 rewards to out_path."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    reward_cfg = RewardGPTConfig(**ckpt['reward_conf'])
    reward_model = RewardGPT(reward_cfg).to(device).eval()
    reward_model.load_state_dict(ckpt['reward_net'])
    for p in reward_model.parameters():
        p.requires_grad_(False)

    R = reward_cfg.block_size
    data = np.memmap(data_path, dtype=np.uint16, mode='r')
    rewards = np.empty(len(data), dtype=np.float32)
    assert len(rewards) == len(data), f"length mismatch: {len(rewards)} != {len(data)}"
    for start in range(0, len(data), R):
        end = min(start + R, len(data))
        chunk = torch.from_numpy(data[start:end].astype(np.int64)).unsqueeze(0).to(device)
        with torch.no_grad():
            rewards[start:end] = reward_model(chunk).squeeze(0).cpu().numpy()
    assert os.path.dirname(out_path) == os.path.dirname(data_path), \
        f"out_path must be in the same folder as data_path: {data_path} vs {out_path}"
    rewards.tofile(out_path)
    assert os.path.getsize(out_path) == len(data) * 4, \
        f"output file size mismatch: {os.path.getsize(out_path)} != {len(data) * 4}"
    print(f"wrote {out_path} ({len(rewards):,} positions, R={R})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pre-compute rewards from a trained RewardGPT.')
    parser.add_argument('--data', required=True, help='Input .bin token file (uint16)')
    parser.add_argument('--out',  required=True, help='Output .bin rewards file (float32)')
    parser.add_argument('--ckpt', required=True, help='Trained RewardGPT checkpoint path')
    args = parser.parse_args()
    prepare(args.data, args.out, args.ckpt)
