from copy import deepcopy
import random

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
from torch import nn

from models.mlp import MLPRegressor
from models.cnn import CNNRegressor
from utils.data_pipeline import load_dataset, create_loader
from utils.training import train_model, evaluate_model, save_model

@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg.train.model not in {"mlp", "cnn", "resnet18", "resnet34", "resnet50"}:
        raise ValueError(f"Invalid training model: {cfg.train.model}. "
                         "Choose mlp, cnn, resnet18, resnet34, or resnet50.")
    cfg.dataset.root = to_absolute_path(cfg.dataset.root)
    seed = cfg.train.get("seed", 0)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if cfg.train.epochs < 1 or cfg.train.patience < 1:
        raise ValueError("epochs and patience must be at least 1")
    if cfg.train.batch_size < 2:
        raise ValueError("batch_size must be at least 2 for batch normalization")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading data. Device: {device}", flush=True)

    train_data, train_labels = load_dataset(cfg, "train")
    valid_data, valid_labels = load_dataset(cfg, "valid")
    test_data, test_labels = load_dataset(cfg, "test")

    if len(train_data) < 2 or len(train_data) % cfg.train.batch_size == 1:
        raise ValueError("Choose a batch_size that avoids a training batch of size 1")

    train_loader = create_loader(
        train_data, train_labels, cfg.train.batch_size, cfg.train.shuffle
    )
    valid_loader = create_loader(
        valid_data, valid_labels, cfg.train.batch_size, shuffle=False
    )
    test_loader = create_loader(
        test_data, test_labels, cfg.train.batch_size, shuffle=False
    )

    if cfg.train.model == "mlp":
        model = MLPRegressor(input_size=train_data[0].numel(), image=cfg.dataset.image,
                             normalize=cfg.train.normalize, in_channels=train_data.shape[1])
    elif cfg.train.model == "cnn":
        model = CNNRegressor(in_channels=train_data.shape[1], image=cfg.dataset.image,
                             normalize=cfg.train.normalize)
    elif cfg.train.model in {"resnet18", "resnet34", "resnet50"}:
        from models.resnet18 import ResNet18Regressor
        from models.resnet34 import ResNet34Regressor
        from models.resnet50 import ResNet50Regressor

        regressors = {"resnet18": ResNet18Regressor, "resnet34": ResNet34Regressor,
                      "resnet50": ResNet50Regressor}
        model = regressors[cfg.train.model](
            in_channels=train_data.shape[1], image=cfg.dataset.image,
            pretrained=cfg.train.transfer.pretrained,
            freeze_backbone=cfg.train.transfer.freeze_backbone,
            freeze_conv1=cfg.train.transfer.freeze_conv1,
            normalize=cfg.train.normalize,
        )
    model.fit_normalization(train_data)
    model = model.to(device)

    losses = {"mse": nn.MSELoss, "mae": nn.L1Loss}
    if cfg.train.loss not in losses:
        raise ValueError(f"Invalid loss: {cfg.train.loss}")
    criterion = losses[cfg.train.loss]()

    optimizers = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW,
                  "sgd": torch.optim.SGD}
    if cfg.train.optimizer not in optimizers:
        raise ValueError(f"Invalid optimizer: {cfg.train.optimizer}")
    parameters = (model.optimizer_parameter_groups(cfg.train.transfer.learning_rate_conv1)
                  if cfg.train.model in {"resnet18", "resnet34", "resnet50"}
                  else (parameter for parameter in model.parameters() if parameter.requires_grad))
    optimizer = optimizers[cfg.train.optimizer](
        parameters,
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )

    train_losses, valid_losses, test_losses = [], [], []
    best_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, cfg.train.epochs + 1):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        valid_loss = evaluate_model(model, valid_loader, criterion, device)
        test_loss = evaluate_model(model, test_loader, criterion, device)
        if not (torch.isfinite(torch.tensor(train_loss))
                and torch.isfinite(torch.tensor(valid_loss))):
            raise ValueError("Training or validation loss is not finite")
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        test_losses.append(test_loss)
        print(
            f"Epoch {epoch}/{cfg.train.epochs}: "
            f"train={train_loss:.4f}, valid={valid_loss:.4f}, "
            f"test={test_loss:.4f}", flush=True
        )

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            best_optimizer = deepcopy(optimizer.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.train.patience:
                print("Early stopping.", flush=True)
                break

    model.load_state_dict(best_state)
    optimizer.load_state_dict(best_optimizer)
    print(f"Best epoch: {best_epoch}. Validation {cfg.train.loss}: {best_loss:.4f}")
    conv1_lr = (cfg.train.transfer.learning_rate_conv1
                if cfg.train.model in {"resnet18", "resnet34", "resnet50"}
                and not cfg.train.transfer.freeze_conv1 else 0)
    if cfg.train.name == "dataset":
        name = (
            f"{cfg.train.model}_{cfg.dataset.crop}_{cfg.dataset.model}_"
            f"{cfg.dataset.image}_{cfg.dataset.mask}_{cfg.dataset.interpolation}_"
            f"{cfg.dataset.target}_{cfg.train.loss}_seed{seed}"
        )
    elif cfg.train.name == "train":
        name = (
            f"{cfg.train.model}_{cfg.dataset.crop}_{cfg.dataset.model}_"
            f"{cfg.dataset.image}_fc{cfg.train.learning_rate}_"
            f"cv{conv1_lr}_{cfg.train.loss}_seed{seed}"
        )
    elif cfg.train.name == "all":
        name = (
            f"{cfg.train.model}_{cfg.dataset.crop}_{cfg.dataset.model}_"
            f"{cfg.dataset.image}_{cfg.dataset.mask}_{cfg.dataset.interpolation}_"
            f"{cfg.dataset.target}_{cfg.train.normalize}_fc{cfg.train.learning_rate}_"
            f"cv{conv1_lr}_{cfg.train.loss}_seed{seed}"
        )
    else:
        raise ValueError(f"Invalid train name: {cfg.train.name}. Choose dataset, train, or all.")
    save_path = save_model(
        model, optimizer, best_epoch, train_losses, valid_losses, test_losses, name,
        to_absolute_path(cfg.train.output),
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    print(f"Saved checkpoint: {save_path}")


if __name__ == "__main__":
    main()
