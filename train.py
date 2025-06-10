import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import models
import dataset
from tqdm import tqdm
from configs import JEPAConfig
from omegaconf import OmegaConf


@torch.no_grad()
def update_target_nets(model, momentum):
    for param_q, param_k in zip(model.predictor_proj.parameters(), model.target_proj.parameters()):
        param_k.data = momentum * param_k.data + (1.0 - momentum) * param_q.data


def loss_func(proj_preds, proj_targets, pred_loss_w=25.0, var_loss_w=25.0, cov_loss_w=1.0, eps=1e-4):
    """
    VICReg Loss
    """
    D = proj_preds.shape[-1]
    pred_loss = F.mse_loss(proj_preds, proj_targets)

    # From VICReg: compute variance & covariance loss on proj_preds
    std_preds = torch.sqrt(proj_preds.var(dim=0) + 1e-4)
    std_targets = torch.sqrt(proj_targets.var(dim=0) + 1e-4)
    var_loss = torch.mean(F.relu(1 - std_preds)) + torch.mean(
        F.relu(1 - std_targets)
    )

    proj_preds_centered = proj_preds - proj_preds.mean(dim=0)
    target_latents_centered = proj_targets - proj_targets.mean(dim=0)

    # Flatten batch & time dims so we have [N, D] where N = B*T
    flat_preds_centered = proj_preds_centered.reshape(-1, D)  # [B*T, D]
    flat_targets_centered = target_latents_centered.reshape(-1, D)

    N = flat_preds_centered.shape[0]
    cov_preds = (flat_preds_centered.T @ flat_preds_centered) / (N - 1)  # [D, D]
    cov_targets = (flat_targets_centered.T @ flat_targets_centered) / (N - 1)

    # Zero out diagonals
    cov_preds.fill_diagonal_(0.0)
    cov_targets.fill_diagonal_(0.0)

    cov_loss = cov_preds.pow(2).sum() / D + cov_targets.pow(2).sum() / D

    loss = pred_loss_w * pred_loss + var_loss_w * var_loss + cov_loss_w * cov_loss
    loss_dict = {
        'total': loss,
        'pred': pred_loss,
        'var': var_loss,
        'cov': cov_loss
    }
    return loss, loss_dict


def train():
    """
    training func
    """
    # config = JEPAConfig.parse_from_file("jepa_config.json")
    config = JEPAConfig()
    print("The config of this training is:")
    print(OmegaConf.to_yaml(config))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataloader = dataset.create_wall_dataloader(
        data_path='/scratch/DL25SP/train',
        probing=False,
        device=device,
        batch_size=config.batch_size,
        train=True
    )
    
    model = models.JEPA(
        repr_dim=config.repr_dim,
        hidden_dim=config.hidden_dim,
        action_dim=config.action_dim
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(
        optimizer=optimizer, 
        T_max=config.num_epochs * len(train_dataloader), 
        eta_min=config.learning_rate * 0.1
    )

    for epoch in tqdm(range(config.num_epochs)):
        model.train()
        losses_sum_dict = {
            'total': 0.0,
            'pred': 0.0,
            'var': 0.0,
            'cov': 0.0,
            'raw_mse': 0.0
        }

        for batch in tqdm(train_dataloader):
            states = batch.states
            actions = batch.actions # (B, T-1, 2)

            optimizer.zero_grad()
            preds_raw, target_latents_raw, proj_preds, proj_targets = model(states, actions) 
            
            # vicreg loss
            loss, loss_dict = loss_func(
                proj_preds=proj_preds, 
                proj_targets=proj_targets,
                pred_loss_w=config.pred_loss_w, 
                var_loss_w=config.var_loss_w, 
                cov_loss_w=config.cov_loss_w, 
                eps=1e-4
            )

            # MSE loss for raw latents
            loss_dict['raw_mse'] = torch.tensor(0)
            # if config.num_epochs - epoch < 5:
            raw_mse_loss = F.mse_loss(preds_raw, target_latents_raw)
            loss_dict['raw_mse'] = raw_mse_loss
            loss += config.raw_mse_w * raw_mse_loss
            loss_dict['total'] += loss_dict['raw_mse']

            loss.backward()

            clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            update_target_nets(model, momentum=config.momentum)

            # record batch loss
            for loss_type in losses_sum_dict:
                losses_sum_dict[loss_type] += loss_dict[loss_type].item()
        
        # scheduler.step()
        # print avg batch loss
        avg_losses = {}
        for loss_type in losses_sum_dict:
            avg_losses[loss_type] = losses_sum_dict[loss_type] / len(train_dataloader)

        print(f"Epoch {epoch + 1}, " + ", ".join(f"{k} loss: {v:.6f}" for k, v in avg_losses.items()))
    
    torch.save(model.state_dict(), config.checkpt_path)
    print(f"Saved at {config.checkpt_path}")


if __name__ == '__main__':
    train()