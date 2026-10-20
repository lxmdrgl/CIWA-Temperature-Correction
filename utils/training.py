import torch

from datetime import datetime
import os

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

# saves model to folder: checkpoints/name.pt
def save_model(model, optimizer, epoch, train_loss, val_loss, test_error, name, config=None):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{name}_val{min(val_loss):.4f}_{timestamp}.pt" 
    
    save_path = os.path.join("checkpoints", filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "test_error": test_error,
        "timestamp": timestamp,
        "config": config,
    }, save_path)
    return save_path
