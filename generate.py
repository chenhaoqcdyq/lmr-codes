"""
Text-to-Motion Generation Inference Script

This script generates motion sequences from text prompts using a trained model.
Now follows the loading pattern from GPT_generate.py:
- Config and VQ-VAE checkpoint path are read from transformer checkpoint directory
- VQ-VAE config is loaded from VQ-VAE checkpoint directory

Usage:
    python generate.py --trans_checkpoint path/to/trans_checkpoint.pth --text "a person walks forward"

Requirements:
    - PyTorch
    - CLIP
    - numpy
"""
import os
import json
import argparse
import torch
import numpy as np
import clip
from tqdm import tqdm

import models.vqvae as vqvae
import models.t2m_trans_cache as trans
from utils.motion_process import recover_from_ric
from utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser(description='Text-to-Motion Generation')
    parser.add_argument('--trans_checkpoint', type=str, required=True,
                        help='Path to transformer checkpoint (config will be loaded from same directory)')
    parser.add_argument('--text', type=str, nargs='+',
                        default=['a person walks forward'],
                        help='Text prompts for generation (can specify multiple)')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Output directory for generated motions')
    parser.add_argument('--num_samples', type=int, default=20,
                        help='Number of motion samples to generate per prompt')
    parser.add_argument('--cond_scale', type=float, default=2.0,
                        help='Classifier-free guidance scale (1.0 = no guidance)')
    parser.add_argument('--min_motion_len', type=int, default=5,
                        help='Minimum motion sequence length')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    return parser.parse_args()


def load_models(args, device):
    """
    Load VQ-VAE and Transformer models following GPT_generate.py pattern

    Loading sequence:
    1. Load transformer config from checkpoint directory
    2. Get VQ-VAE checkpoint path from transformer config (resume_pth field)
    3. Load VQ-VAE config from VQ-VAE checkpoint directory
    4. Initialize and load both models
    """

    # Step 1: Load transformer config from checkpoint directory
    trans_checkpoint_dir = os.path.dirname(args.trans_checkpoint)
    config_path = os.path.join(trans_checkpoint_dir, 'train_config.json')

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Configuration file not found at {config_path}.\n"
            f"Make sure train_config.json exists in the same directory as the transformer checkpoint."
        )

    print(f'Loading transformer config from {config_path}')
    config = load_config(config_path)

    # Step 2: Get VQ-VAE checkpoint path from config
    if not hasattr(config, 'resume_pth') or config.resume_pth is None:
        raise ValueError(
            f"Config file must contain 'resume_pth' field pointing to VQ-VAE checkpoint.\n"
            f"Current config: {config_path}"
        )

    vq_checkpoint_path = config.resume_pth
    print(f'VQ-VAE checkpoint path from config: {vq_checkpoint_path}')

    # Step 3: Load VQ-VAE config from its checkpoint directory
    vq_checkpoint_dir = os.path.dirname(vq_checkpoint_path)
    vq_config_path = os.path.join(vq_checkpoint_dir, 'train_config.json')

    if not os.path.exists(vq_config_path):
        raise FileNotFoundError(
            f"VQ-VAE configuration file not found at {vq_config_path}.\n"
            f"Make sure train_config.json exists in the VQ-VAE checkpoint directory."
        )

    print(f'Loading VQ-VAE config from {vq_config_path}')
    vq_config = load_config(vq_config_path)

    # Step 4: Initialize VQ-VAE model
    print("\nInitializing VQ-VAE model...")
    vq_model = vqvae.HumanVQVAE(
        vq_config,
        vq_config.nb_code,
        vq_config.code_dim,
        vq_config.output_emb_width,
        vq_config.down_t,
        vq_config.stride_t,
        vq_config.width,
        vq_config.depth,
        vq_config.dilation_growth_rate,
        vq_config.vq_act,
        causal=vq_config.get('causal', 0)
    )

    # Step 5: Load VQ-VAE weights
    print(f'Loading VQ-VAE weights from {vq_checkpoint_path}')
    if not os.path.exists(vq_checkpoint_path):
        raise FileNotFoundError(f"VQ-VAE checkpoint not found at {vq_checkpoint_path}")

    ckpt = torch.load(vq_checkpoint_path, map_location='cpu')
    # Use strict=False because we don't need lgvq_encoder for inference
    missing_keys, unexpected_keys = vq_model.load_state_dict(ckpt['net'], strict=False)

    if missing_keys:
        print(f"Warning: {len(missing_keys)} missing keys (expected for inference-only model)")

    if unexpected_keys:
        # Filter out lgvq_encoder keys (expected to be absent in inference model)
        non_lgvq_keys = [k for k in unexpected_keys if 'lgvq_encoder' not in k]

        if non_lgvq_keys:
            print(f"\nError: Unexpected keys found in checkpoint (not related to lgvq_encoder):")
            for key in non_lgvq_keys[:10]:
                print(f"  - {key}")
            if len(non_lgvq_keys) > 10:
                print(f"  ... and {len(non_lgvq_keys) - 10} more keys")
            raise RuntimeError(
                f"Model architecture mismatch: {len(non_lgvq_keys)} unexpected keys found. "
                "Please check that your VQ-VAE checkpoint matches the model architecture."
            )
        else:
            print(f"Info: Ignoring {len(unexpected_keys)} lgvq_encoder keys (not needed for inference)")

    vq_model.eval()
    vq_model.to(device)
    print("✓ VQ-VAE model loaded successfully")

    # Step 6: Determine transformer parameters (following GPT_generate.py logic)
    lgvq = 1
    num_vq_trans = config.nb_code

    # Handle test_nb if present
    if config.get('test_nb', 0):
        num_vq_trans = config.nb_code * 2 + 1

    # Calculate unit_length and semantic_len (following GPT_generate.py logic)
    # if hasattr(vq_config, 'down_vqvae'):
    #     if vq_config.down_vqvae and vq_config.down_t == 2:
    #         unit_length = 4
    #     elif vq_config.down_vqvae and vq_config.down_t == 1:
    #         unit_length = 2
    #     else:
    #         unit_length = 1
    # else:
    unit_length = 1

    semantic_len = ((196 // unit_length) + 3) // 4 + 1

    print(f"\nModel configuration:")
    print(f"  - lgvq: {lgvq}")
    print(f"  - num_vq_trans: {num_vq_trans}")
    print(f"  - semantic_len: {semantic_len}")
    print(f"  - unit_length: {unit_length}")

    # Step 7: Initialize Transformer
    print("\nInitializing Transformer model...")
    trans_encoder = trans.Text2Motion_Transformer(
        num_vq=num_vq_trans,
        embed_dim=config.embed_dim_gpt,
        clip_dim=config.clip_dim,
        block_size=config.block_size,
        num_layers=config.num_layers,
        n_head=config.n_head_gpt,
        drop_out_rate=config.drop_out_rate,
        fc_rate=config.ff_rate,
        semantic_len=semantic_len,
        dual_head_flag=True,
        uncond_prob=config.get('classfg', 0)
    )

    # Step 8: Load Transformer weights
    print(f'Loading Transformer weights from {args.trans_checkpoint}')
    ckpt = torch.load(args.trans_checkpoint, map_location='cpu')
    trans_encoder.load_state_dict(ckpt['trans'], strict=True)
    trans_encoder.eval()
    trans_encoder.to(device)
    print("✓ Transformer model loaded successfully\n")

    return vq_model, trans_encoder, config, vq_config


def generate_motions(texts, vq_model, trans_encoder, clip_model, args, device, config, vq_config):
    """Generate motion sequences from text prompts"""
    # Encode text with CLIP
    text_tokens = clip.tokenize(texts, truncate=True).to(device)
    with torch.no_grad():
        feat_clip_text = clip_model.encode_text(text_tokens).float()

    if feat_clip_text.ndim == 2:
        feat_clip_text = feat_clip_text.unsqueeze(0) if feat_clip_text.shape[0] == 1 else feat_clip_text

    # Load mean and std for denormalization
    meta_dir = "checkpoints/meta"
    if not os.path.exists(meta_dir):
        print(f"Warning: Meta directory not found at {meta_dir}")
        raise FileNotFoundError(
            f"Meta directory not found at {meta_dir}.\n"
            f"Please ensure that mean.npy and std.npy are available for denormalization."
        )
    else:
        mean = np.load(os.path.join(meta_dir, 'mean.npy'))
        std = np.load(os.path.join(meta_dir, 'std.npy'))

    num_joints = 22 if config.dataname == 't2m' else 21

    all_motions = []

    # Generate multiple samples for each text
    for text_idx, text in enumerate(texts):
        print(f"\nGenerating motions for: '{text}'")
        text_motions = []

        for i in tqdm(range(args.num_samples), desc=f"Generating samples"):
            with torch.no_grad():
                # Sample motion tokens from transformer
                batch_parts_index_motion = trans_encoder.sample_batch(
                    feat_clip_text[text_idx:text_idx+1],
                    if_categorial=True,
                    cond_scale=args.cond_scale,
                    min_motion_len=args.min_motion_len
                )

                # Post-process tokens based on model configuration (following GPT_generate.py)
                batch_parts_index_motion = batch_parts_index_motion[..., trans_encoder.semantic_len:]

                max_motion_seq_len = batch_parts_index_motion.shape[1]
                part_index = batch_parts_index_motion

                # Find the earliest end token position
                idx = torch.nonzero(part_index == trans_encoder.num_vq)

                if idx.numel() == 0:
                    motion_seq_len = max_motion_seq_len
                else:
                    min_end_idx = idx[:, 1].min()
                    motion_seq_len = min_end_idx

                # Handle very short sequences
                if motion_seq_len <= 3:
                    parts_index_motion = torch.ones(1, 4).to(device).long()
                    print(f"Warning: Very short motion generated (len <= 3)")
                else:
                    parts_index_motion = part_index[:, :motion_seq_len]

                # Decode motion from tokens
                pred_pose = vq_model.forward_decoder(parts_index_motion)

                # Denormalize
                pred_denorm = (pred_pose.detach().cpu().numpy() * std) + mean

                # Convert to 3D positions
                pred_xyz = recover_from_ric(torch.from_numpy(pred_denorm).float().to(device), num_joints)

                text_motions.append(pred_xyz.cpu().numpy())

        all_motions.append(text_motions)

    return all_motions


def save_motions(motions, texts, output_dir):
    """Save generated motions to disk"""
    os.makedirs(output_dir, exist_ok=True)

    for text_idx, (text, text_motions) in enumerate(zip(texts, motions)):
        # Create subdirectory for each text prompt
        text_dir = os.path.join(output_dir, f"prompt_{text_idx:02d}")
        os.makedirs(text_dir, exist_ok=True)

        # Save text prompt
        with open(os.path.join(text_dir, "prompt.txt"), 'w') as f:
            f.write(text)

        # Save each motion sample
        for i, motion in enumerate(text_motions):
            motion_path = os.path.join(text_dir, f"motion_{i:03d}.npy")
            np.save(motion_path, motion)

    print(f"\n✓ Motions saved to {output_dir}")


def main():
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print("="*60)
    print("Text-to-Motion Generation")
    print("="*60)
    print(f"Device: {device}")
    print(f"Random seed: {args.seed}")
    print(f"Transformer checkpoint: {args.trans_checkpoint}")

    # Load CLIP model
    print("\nLoading CLIP model...")
    clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False
    print("✓ CLIP model loaded")

    # Load models
    print("\n" + "="*60)
    print("Loading models...")
    print("="*60)
    vq_model, trans_encoder, config, vq_config = load_models(args, device)

    # Generate motions
    print("="*60)
    print("Starting generation...")
    print("="*60)
    print(f"Text prompts: {args.text}")
    print(f"Samples per prompt: {args.num_samples}")
    print(f"CFG scale: {args.cond_scale}")
    print(f"Min motion length: {args.min_motion_len}")

    motions = generate_motions(
        args.text,
        vq_model,
        trans_encoder,
        clip_model,
        args,
        device,
        config,
        vq_config
    )
    os.makedirs(args.output_dir, exist_ok=True)
    # Save results
    save_motions(motions, args.text, args.output_dir)

    print("\n" + "="*60)
    print("✓ Generation complete!")
    print("="*60)


if __name__ == '__main__':
    main()
