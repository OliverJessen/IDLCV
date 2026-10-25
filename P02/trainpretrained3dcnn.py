import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision import transforms as T

from datasets import FrameVideoDataset
from modelpme import Pretrained3DCNN


EPOCHS = 15
BATCH_SIZE = 8      # if you get "CUDA out of memory", try 4
LR = 1e-4           # much lower than from scratch: we only want to fine-tune the pretrained weights

# The pretrained network was trained on 112x112 frames, so we use that size here
transform = T.Compose([T.Resize((112, 112)), T.ToTensor()])

#root_dir = "/zhome/e5/5/233066/project2/ucf101/ufc10"
root_dir = "/dtu/datasets1/02516/ucf101_noleakage"

# The stack frames true is important because it returns [C, T, H, W] for each video
train_dataset = FrameVideoDataset(root_dir=root_dir, split="train", transform=transform, stack_frames=True)
val_dataset = FrameVideoDataset(root_dir=root_dir, split="val", transform=transform, stack_frames=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("Train:", len(train_dataset), "| Val:", len(val_dataset), flush=True)

# Check that the dataloaders work: expect Videos: torch.Size([8, 3, 10, 112, 112])
for videos, labels in train_loader:
    print("Videos:", videos.shape)
    print("Labels:", labels.shape, flush=True)
    break

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device, flush=True)


# DATA AUGMENTATION (training only): the same flip for all the frames of a video
def augment(videos):
    B = videos.shape[0]
    flip = torch.rand(B, device=videos.device) < 0.5
    videos = torch.where(flip.view(B, 1, 1, 1, 1), videos.flip(-1), videos)
    return videos


def train_model(model, train_loader, val_loader, epochs=10):
    model = model.to(device)
    print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad), flush=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    val_accs = []
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
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train loss={train_loss:.4f}, Train acc={train_acc:.2f}% | "
              f"Val loss={val_loss:.4f}, Val acc={val_acc:.2f}%", flush=True)

    return val_accs


print("\n=== Pretrained 3D CNN (R3D-18, Kinetics-400) ===", flush=True)
model = Pretrained3DCNN()
val_accs = train_model(model, train_loader, val_loader, epochs=EPOCHS)

last5 = val_accs[-5:]
print(f"\nFinal val acc: {val_accs[-1]:.2f}% | mean of last 5 epochs: {sum(last5)/len(last5):.2f}%", flush=True)