import torch
from tqdm import tqdm 
from model import resnet
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from model import resnet , efficientnet , mobilenet , squeezenet
from data_prep.data_load import UniformTrianingDataset , UniformValidationDataset


def train_one_ep(
        model,
        data_ld,
        loss_fn,
        optim,

):
    device = torch.device("cuda")
    model.to(device)
    model.train()
    run_loss = 0
    correct = 0
    total = 0

    for img , label in tqdm(data_ld , desc="Training", leave=False):
        label = torch.tensor(label)
        img , label = img.to(device) , label.to(device)

        optim.zero_grad()

        out = model(img)

        loss = loss_fn(out , label)

        loss.backward()
        optim.step()

        run_loss += loss.item()* img.size(0)

        _ , pred = torch.max(out , 1)
        correct += (pred == label).sum().item()

        total += label.size(0)

    ep_loss = run_loss / total
    ep_acc = correct / total
    return ep_loss , ep_acc


def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()  
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        
        for inputs, targets in tqdm(dataloader, desc="Validation", leave=False):
            targets = torch.tensor(targets)
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc



def main():

    device = torch.device("cuda")
    train_ds = UniformTrianingDataset()
    valid_ds = UniformValidationDataset()
    train_ld = DataLoader(train_ds , batch_size=8 , shuffle=True , num_workers=4)
    valid_ld = DataLoader(valid_ds, batch_size=8 ,shuffle=True , num_workers=4)

    model = resnet.Custom_resnet(num_classes=2)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_ep = 30
    best_val_acc = 96.0

    for ep in range(1 , num_ep + 1):

        print(f"\nEpoch {ep}/{num_ep}")

        train_loss, train_acc = train_one_ep(model=model , data_ld=train_ld , loss_fn=loss_fn , optim=optimizer)
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        
        # Validation phase
        val_loss, val_acc = validate_one_epoch(model, valid_ld, loss_fn, device)
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print("  -> Saved Best Model!")

    print("\nTraining complete!")


if __name__ == "__main__":

    main()