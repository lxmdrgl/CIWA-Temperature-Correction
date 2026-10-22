import torch

from datetime import datetime
import os
from pathlib import Path
from textwrap import fill

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

# trains model and returns loss
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0
    total = 0
    
    for inputs, targets in train_loader:
        # use device
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        # go forward through model and find loss
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # backpropagation and optimize
        optimizer.zero_grad() # zero gradient buffers
        loss.backward()
        optimizer.step()
        
        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        total += batch_size

    epoch_loss = running_loss / total
    
    return epoch_loss

def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            total += batch_size

    epoch_loss = running_loss / total

    return epoch_loss

# Saves the checkpoint and learning curve under the configured output directory.
def save_model(model, optimizer, epoch, train_loss, val_loss, test_loss, name, output, config=None):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{name}_val{min(val_loss):.4f}_{timestamp}.pt" 
    
    save_path = os.path.join(output, filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "test_loss": test_loss,
        "timestamp": timestamp,
        "config": config,
    }
    torch.save(checkpoint, save_path)
    save_learning_curve(checkpoint, save_path)
    return save_path


def save_learning_curve(checkpoint, checkpoint_path):
    """Save loss histories and configuration beside the checkpoint folder."""
    checkpoint_path = Path(checkpoint_path)
    plot_path = checkpoint_path.parent / "learning_curves" / checkpoint_path.with_suffix(".png").name
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    config = checkpoint.get("config") or {}
    config_lines = []
    for section in ("dataset", "train"):
        values = ", ".join(f"{key}={value}" for key, value in config.get(section, {}).items())
        config_lines.append(fill(f"{section}: {values}", width=105, subsequent_indent="  "))
    if not config:
        config_lines = ["Configuration not recorded in this checkpoint."]
    if not checkpoint.get("test_loss"):
        config_lines.append("Test loss history not recorded in this checkpoint.")
    config_text = "\n".join(config_lines)
    config_height = max(1.0, 0.2 * len(config_text.splitlines()))

    figure = Figure(figsize=(10, 4 + config_height), layout="constrained")
    FigureCanvasAgg(figure)
    axes, details = figure.subplots(2, 1, gridspec_kw={"height_ratios": [4, config_height]})
    for key, label in (("train_loss", "Train"), ("val_loss", "Valid"), ("test_loss", "Test")):
        losses = checkpoint.get(key)
        if losses:
            axes.plot(range(1, len(losses) + 1), losses, label=label)
    axes.set(title="Learning curve", xlabel="Epoch", ylabel=config.get("train", {}).get("loss", "Loss").upper())
    axes.xaxis.set_major_locator(MaxNLocator(integer=True))
    axes.legend(frameon=False)
    axes.spines[["top", "right"]].set_visible(False)
    details.set_axis_off()
    details.text(0, 1, config_text, transform=details.transAxes, va="top", fontsize=9, family="monospace")
    figure.savefig(plot_path, dpi=150)
    return plot_path
