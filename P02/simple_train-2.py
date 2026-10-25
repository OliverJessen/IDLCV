import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms as T
from datasets import FrameImageDataset, FrameVideoDataset

root = 'ufc10'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)

# 1) Data: single frames for training, videos (list of 10 frames) for validation and test
norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])   # what pretrained ResNets expect
train_tf = T.Compose([T.RandomResizedCrop(224, scale=(0.6, 1.0)), T.RandomHorizontalFlip(),
                      T.ToTensor(), norm])                          # augmentation (training only)
eval_tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), norm])

train_set = FrameImageDataset(root_dir=root, split='train', transform=train_tf)
val_set = FrameVideoDataset(root_dir=root, split='val', transform=eval_tf, stack_frames=False)
test_set = FrameVideoDataset(root_dir=root, split='test', transform=eval_tf, stack_frames=False)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_set, batch_size=8, num_workers=4)
test_loader = DataLoader(test_set, batch_size=8, num_workers=4)

loss_fn = nn.CrossEntropyLoss()


# 2) Evaluation: classify every frame, then average the predictions of the 10 frames per video
def evaluate(model, loader):
    model.eval()
    loss_sum, frame_correct, video_correct, n_videos = 0, 0, 0, 0
    with torch.no_grad():
        for frames, labels in loader:
            x = torch.stack(frames, dim=1).to(device)           # [batch, 10, 3, H, W]
            labels = labels.to(device)
            B, n = x.shape[:2]
            out = model(x.flatten(0, 1)).view(B, n, -1)         # [batch, 10, 10 classes]

            loss_sum += loss_fn(out.flatten(0, 1), labels.repeat_interleave(n)).item() * B
            frame_correct += (out.argmax(-1) == labels[:, None]).sum().item()          # single frames
            video_correct += (out.softmax(-1).mean(1).argmax(-1) == labels).sum().item()  # averaged
            n_videos += B
    return loss_sum / n_videos, frame_correct / (n_videos * 10), video_correct / n_videos


# 3) Training loop: keeps the best epoch according to VALIDATION video accuracy
def train_model(model, lr, epochs):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    best_acc, best_state = -1, None

    for epoch in range(epochs):
        model.train()
        loss_sum, correct, total = 0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            total += len(y)

        val_loss, val_frame_acc, val_video_acc = evaluate(model, val_loader)
        print(f'Epoch {epoch + 1}/{epochs}: Train loss={loss_sum / total:.4f}, '
              f'Train acc={100 * correct / total:.2f}% | Val loss={val_loss:.4f}, '
              f'Val frame acc={100 * val_frame_acc:.2f}%, Val video acc={100 * val_video_acc:.2f}%',
              flush=True)

        if val_video_acc > best_acc:
            best_acc, best_state = val_video_acc, copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model, best_acc


# 4) Run: pretrained ResNet-18 and the same model from scratch (for comparison)
results = {}
for name, pretrained, lr, epochs in [('Pretrained ResNet-18', True, 1e-4, 15),
                                     ('Scratch ResNet-18', False, 1e-3, 30)]:
    print(f'\n=== Per-frame aggregation: {name} ===', flush=True)
    model = models.resnet18(weights='DEFAULT' if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, 10)              # 10 classes
    model, best_val = train_model(model, lr, epochs)

    _, test_frame_acc, test_video_acc = evaluate(model, test_loader)   # test set: used once
    results[name] = (best_val, test_frame_acc, test_video_acc)
    print(f'Best val video acc: {100 * best_val:.2f}% | TEST frame acc: {100 * test_frame_acc:.2f}% '
          f'| TEST video acc: {100 * test_video_acc:.2f}%', flush=True)

print('\nSummary (val video acc / test frame acc / test video acc):')
for name, (v, tf, tv) in results.items():
    print(f'{name}: {100 * v:.2f}% / {100 * tf:.2f}% / {100 * tv:.2f}%')
