# JEPA World Model for Future State Prediction

This project implements a **Joint Embedding Predictive Architecture (JEPA)** to model agent dynamics in toy environments. It learns to predict future latent representations of visual states given past pixel observations and actions using a VICReg-style objective, with added probing to evaluate spatial information retention.

## 🧠 Key Features

- **JEPA architecture** combining visual encoders, RNN dynamics, and BYOL-style projections  
- **VICReg loss** integrating prediction, variance, and covariance terms  
- **Action-conditioned latent prediction** using GRU-based recurrent modeling  
- **Pixel-to-latent learning** from 2.5M agent trajectories  
- **Evaluation with probing tasks** to assess embedding quality  

## 🔍 Motivation

Learning a compact yet informative latent space is crucial for planning and control. This JEPA model explores how well self-supervised predictive learning captures spatial dynamics in a minimal world using only image sequences and actions — without explicit supervision.

## 🧪 Architecture Overview

1. **Encoder**: Two CNN branches extract embeddings from sequential frames.  
2. **Action Embedding**: Actions are embedded into the same latent space.  
3. **RNN Module**: Predicts future latent representations conditioned on past latents and actions.  
4. **Projection Heads**: BYOL-style projections used for VICReg-based contrastive training.  
5. **Probing Heads**: MLP regressors assess how well representations preserve (x, y) position data.

Input: (states: [B, T, C, H, W], actions: [B, T-1, 2])
→ CNN Encoder
→ RNN Dynamics
→ VICReg Loss (Prediction + Variance + Covariance)
→ Evaluation via probing (val loss on (x, y) prediction)

## 📊 Evaluation Results

| Probe Task        | Validation Loss ↓ |
|-------------------|-------------------|
| `probe_normal`    | 3.16              |
| `probe_wall`      | 6.31              |
| `wall_other`      | 7.06              |
| `expert`          | 15.25             |

- **# of Trainable Parameters**: `477,664`

These results indicate meaningful spatial encoding, especially in normal and structured environments.

## ⚙️ Training Details

- **Optimizer**: AdamW  
- **Loss**: VICReg (prediction + variance + covariance) + MSE on raw latents  
- **Scheduler**: CosineAnnealingLR  
- **Epochs**: Configurable (e.g., 100)  
- **Batch Size**: From `JEPAConfig`  
- **Dataset**: 2.5M frames from toy environment agent trajectories  
- **Checkpoint**: Saved at path from config (`config.checkpt_path`)  


## 📌 Future Work

- Add support for longer horizon prediction
- Explore alternative loss functions (e.g., InfoNCE, SimSiam)
- Visualize learned representations with t-SNE or PCA
- Apply to more complex environments with richer dynamics
