from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch


MODEL_PATH = "epoch_model.pth"

def train(model, dataset, num_epochs):
    data_loader = DataLoader(dataset, batch_size=20, shuffle=True, collate_fn=collate_fn, num_workers=0)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        print('epoch '+str(epoch+1))
        for i, (images, targets) in enumerate(data_loader):
            print(f"Batch {i}")
            images = list(img for img in images)
            targets = [{k: v for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        print(f"Epoch {epoch+1} loss: {losses.item():.4f}")
        torch.save(model.state_dict(), MODEL_PATH)

def collate_fn(batch):
    return tuple(zip(*batch))
