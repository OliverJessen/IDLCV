import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.models import resnet18, ResNet18_Weights

from datasets import DEFAULT_ROOT_DIR, FrameVideoDataset
from modelpme import EarlyFusionCNN, Simple3DCNN, PretrainedEarlyFusionCNN

root_dir = DEFAULT_ROOT_DIR

weights = ResNet18_Weights.DEFAULT
pretrained_transform = T.Compose([
    T.Resize((112, 112)),
    T.ToTensor(),
    T.Normalize(
        mean=weights.transforms().mean,
        std=weights.transforms().std
    )
])

# Number of epochs for each model (the 3D CNN gets more epochs)
EPOCHS_EARLY = 50
EPOCHS_3D = 30

# First we load the dataset and create the dataloaders
# Original datasets: scratch 2D CNN and 3D CNN
transform = T.Compose([
    T.Resize((64, 64)),
    T.ToTensor()
])

train_dataset = FrameVideoDataset(
    root_dir=root_dir,
    split="train",
    transform=transform,
    stack_frames=True
)

val_dataset = FrameVideoDataset(
    root_dir=root_dir,
    split="val",
    transform=transform,
    stack_frames=True
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)


pretrained_train_dataset = FrameVideoDataset(
    root_dir=root_dir,
    split="train",
    transform=pretrained_transform,
    stack_frames=True
)

pretrained_val_dataset = FrameVideoDataset(
    root_dir=root_dir,
    split="val",
    transform=pretrained_transform,
    stack_frames=True
)

pretrained_train_loader = DataLoader(
    pretrained_train_dataset, batch_size=16, shuffle=True
)

pretrained_val_loader = DataLoader(
    pretrained_val_dataset, batch_size=16, shuffle=False
)

#Check before training if the dataloaders are working properly, should see Videos: torch.Size([8, 3, 10, 64, 64]) and Labels: torch.Size([8])
for videos, labels in pretrained_train_loader:
    print("Videos:", videos.shape)
    print("Labels:", labels.shape)
    break

# DATA AUGMENTATION (only used on the training data)
# It is applied on the whole batch [B, C, T, H, W], so every frame of a video gets
# the SAME transformation and the motion is preserved. (A per-frame transform in the
# dataset would flip each frame independently and corrupt the video.)
def augment(videos):
    B, C, T_, H, W = videos.shape
 
    # STEP A: random horizontal flip, decided independently for each video
    flip = torch.rand(B, device=videos.device) < 0.5
    videos = torch.where(flip.view(B, 1, 1, 1, 1), videos.flip(-1), videos)
    return videos

# Basic training loop  as usual
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, train_loader, val_loader, epochs=10):
    model = model.to(device)
 
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
 
    for epoch in range(epochs):
        # ---- Training phase ----
        model.train()
        running_loss, train_correct, train_total = 0, 0, 0
 
        for videos, labels in train_loader:
            videos, labels = videos.to(device), labels.to(device)
            videos = augment(videos) 
 
            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
 
            running_loss += loss.item()
            train_correct += (outputs.argmax(dim=1) == labels).sum().item()
            train_total += labels.size(0)
 
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
 
        # ---- Validation phase ----
        model.eval()
        val_loss_sum, val_correct, val_total = 0, 0, 0
 
        with torch.no_grad():
            for videos, labels in val_loader:
                videos, labels = videos.to(device), labels.to(device)
 
                outputs = model(videos)
                val_loss_sum += criterion(outputs, labels).item()
                val_correct += (outputs.argmax(dim=1) == labels).sum().item()
                val_total += labels.size(0)
 
        val_loss = val_loss_sum / len(val_loader)
        val_acc = 100 * val_correct / val_total
 
        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train loss={train_loss:.4f}, Train acc={train_acc:.2f}% | "
              f"Val loss={val_loss:.4f}, Val acc={val_acc:.2f}%", flush=True)
 
    return val_acc

def train_pretrained_model(model, train_loader, val_loader, epochs=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam([
        {
            "params": [
                p for name, p in model.backbone.named_parameters()
                if not name.startswith("conv1.")
                and not name.startswith("fc.")
            ],
            "lr": 1e-4
        },
        {
            "params": model.backbone.conv1.parameters(),
            "lr": 1e-4
        },
        {
            "params": model.backbone.fc.parameters(),
            "lr": 1e-3
        }
    ], weight_decay=1e-4)

    best_val_acc = 0.0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        # --------------------
        # Training
        # --------------------
        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for videos, labels in train_loader:
            videos = videos.to(device)
            labels = labels.to(device).long()

            # Horizontal flip, applied consistently to every frame
            if torch.rand(1).item() < 0.5:
                videos = videos.flip(-1)

            optimizer.zero_grad()

            outputs = model(videos)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            train_correct += (
                outputs.argmax(dim=1) == labels
            ).sum().item()
            train_total += labels.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # --------------------
        # Validation
        # --------------------
        model.eval()

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for videos, labels in val_loader:
                videos = videos.to(device)
                labels = labels.to(device).long()

                outputs = model(videos)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * labels.size(0)
                val_correct += (
                    outputs.argmax(dim=1) == labels
                ).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2%} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2%}"
        )

        # Save best validation checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)

    print(f"\nBest validation accuracy: {best_val_acc:.2%}")

    return model, best_val_acc

# Now to run the code 
print("\n=== Early Fusion 2D CNN (Scratch) ===", flush=True)
early_model = EarlyFusionCNN()
early_acc = train_model(
    early_model, train_loader, val_loader, epochs=EPOCHS_EARLY
)

print("\n=== Early Fusion 2D CNN (Pretrained ResNet-18) ===", flush=True)
early_modelpre = PretrainedEarlyFusionCNN(num_classes=10, num_frames=10)
early_modelpre, early_accpre = train_pretrained_model(
    early_modelpre,
    pretrained_train_loader,
    pretrained_val_loader,
    epochs=50
)

print("\n=== Simple 3D CNN ===", flush=True)
cnn3d_model = Simple3DCNN()
cnn3d_acc = train_model(
    cnn3d_model, train_loader, val_loader, epochs=EPOCHS_3D
)

print(
    f"\nValidation accuracies -> "
    f"Early Fusion (Scratch): {early_acc:.2f}% | "
    f"Early Fusion (Pretrained): {early_accpre * 100:.2f}% | "
    f"3D CNN: {cnn3d_acc:.2f}%",
    flush=True  
)
