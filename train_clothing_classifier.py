import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# parâmetros
BATCH_SIZE = 32
EPOCHS     = 15
IMG_SIZE   = 448
DATA_ROOT  = os.path.join(os.path.dirname(__file__), "Lisas_Estampadas")
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# transforms (ImageNet norm)
train_tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(.2, .2, .2),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(.8,1.0)),
    transforms.ToTensor(),
    transforms.Normalize([.485, .456, .406], [ .229, .224, .225 ])
])
val_tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([.485, .456, .406], [ .229, .224, .225 ])
])

def train_part(part: str):
    print(f"\n=== Treinando classificador para '{part}' ===")
    data_dir = os.path.join(DATA_ROOT, part)
    ds_full = datasets.ImageFolder(data_dir)
    n = len(ds_full)
    n_train = int(.8 * n)
    n_val   = n - n_train
    ds_train, ds_val = random_split(ds_full, [n_train, n_val])
    # aplica transforms
    ds_train.dataset.transform = train_tfm
    ds_val.dataset.transform   = val_tfm

    loader_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    loader_val   = DataLoader(ds_val,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # modelo MobileNetV2 fine-tune
    model = models.mobilenet_v2(pretrained=True)
    # congela todas as features, exceto ultima camada convolucional (opcional)
    for name, p in model.features.named_parameters():
        p.requires_grad = False
        if "17" in name or "18" in name:
            p.requires_grad = True
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    model = model.to(DEVICE)

    # calcular peso de classes
    y = [label for _, label in ds_train]
    class_w = compute_class_weight("balanced", classes=np.unique(y), y=y)
    class_w = torch.tensor(class_w, dtype=torch.float, device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_w)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-4)

    # loop de treino
    for epoch in range(1, EPOCHS+1):
        model.train()
        running_loss = 0
        for xb, yb in loader_train:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(loader_train)
        print(f"Epoch {epoch}/{EPOCHS}  loss: {avg_loss:.4f}")

    # avaliação
    print("\n--- Avaliação ---")
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in loader_val:
            xb = xb.to(DEVICE)
            logits = model(xb)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            y_true.extend(yb.numpy())
            y_pred.extend(preds)
    print(classification_report(y_true, y_pred, target_names=ds_full.classes))

    # salvar
    out_path = os.path.join(os.path.dirname(__file__),
                            f"clothing_{part}_classifier.pt")
    torch.save(model.state_dict(), out_path)
    print(f"Modelo '{part}' salvo em {out_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["top","bottom","all"],
                        default="all", help="Treina só 'top', 'bottom' ou ambos")
    args = parser.parse_args()

    if args.part in ("top","all"):
        train_part("top")
    if args.part in ("bottom","all"):
        train_part("bottom")
