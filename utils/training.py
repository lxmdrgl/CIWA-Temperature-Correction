import torch

from datetime import datetime
import os

# trains model and returns loss
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0
    
    for inputs, labels in train_loader:
        # use device
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # go forward through model and find loss
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # backpropagation and optimize
        optimizer.zero_grad() # zero gradient buffers
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    return epoch_loss

# evaluates the model and returns loss and accuracy
def evaluate_model(model, test_loader, criterion, device, num_classes):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0

    # per-class tracking
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()

            # update per-class stats
            for i in range(labels.size(0)):
                label = labels[i].item()
                pred = predicted[i].item()

                class_total[label] += 1
                if pred == label:
                    class_correct[label] += 1

    # overall accuracy
    overall_accuracy = 100 * correct / total
    epoch_loss = running_loss / len(test_loader)

    # per-class accuracy
    class_accuracy = [
        100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        for i in range(num_classes)
    ]

    return epoch_loss, overall_accuracy, class_accuracy

# saves model to folder: checkpoints/name.pt
def save_model(model, optimizer, epoch, train_loss, val_loss, test_error, name):
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
    }, save_path)