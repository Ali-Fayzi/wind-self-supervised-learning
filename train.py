import os
import torch
import wandb
import numpy as np 
from torch import functional as F 
from torch import nn
import torch.optim as optim
from torchvision import transforms as T
from torch.utils.data import DataLoader , Dataset
from model import Decoder,vit_base
from dataset_v2 import WindDataset as Dataset
from matplotlib import pyplot as plt 
def validate_model(epoch,model, val_loader, criterion):
    model.eval()
    val_loss = []
    with torch.no_grad():
        for index,(images, masked_patches) in enumerate(val_loader):
            images = images.to(device)
            masked_patches = masked_patches.to(device)
            outputs = model(images)
            loss = criterion(outputs, masked_patches)
            val_loss.append(loss.item())
    epoch_val_loss = np.mean(val_loss)
    wandb.log({"epoch":epoch+1,"mode":"validation","loss":f"epoch_val_loss:.4f"})
    return epoch_val_loss

def train_model(model, train_dataloader,val_dataloader, criterion, optimizer, num_epochs=5):
    best_val_loss = np.inf
    for epoch in range(num_epochs):
        model.train()
        training_loss = []
        for index,(images,masked_patches) in enumerate(train_dataloader):
            # Move tensors to the appropriate device
            
            images = images.to(device)
            masked_patches = masked_patches.to(device)
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(images)

            # for lbl,b in zip(masked_patches,outputs):
            #     for targ,img in zip(lbl,b):
            #         img = img.detach().cpu().numpy().transpose(2,1,0)
            #         targ = targ.detach().cpu().numpy().transpose(2,1,0)
            #         plt.imshow(np.hstack([img,targ]),cmap='gray')
            #         plt.show()

            # print(images.shape , masked_patches.shape , outputs.shape)
            # import sys;sys.exit()
            loss = criterion(outputs, masked_patches)
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            # Accumulate loss
            training_loss.append(loss.item())
        epoch_loss = np.mean(training_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

        wandb.log({"epoch":epoch+1,"mode":"train","loss":f"epoch_loss:.4f"})
        val_loss = validate_model(epoch,model, val_dataloader, criterion)
        print(f"Val Loss: {val_loss:.4f}")
        if val_loss < best_val_loss :
            best_val_loss = val_loss
            os.makedirs(f"./checkpoint/",exist_ok=True)
            torch.save(model.state_dict(), f"./checkpoint/model_{epoch}_{val_loss:.4f}.pth")
            torch.save(model.encoder.state_dict(), f"./checkpoint/encoder_{epoch}_{val_loss:.4f}.pth")
            torch.save(model.decoder.state_dict(), f"./checkpoint/decode_{epoch}_{val_loss:.4f}.pth")
    print('Training complete')
    return model

class Model(nn.Module):
  def __init__(self,encoder,decoder):
    super(Model, self).__init__()
    self.encoder = encoder
    self.decoder = decoder
  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
    return x

if __name__ == "__main__":

    wandb.login()

    crop_size = 224
    patch_size = 16
    num_frames = 8
    tubelet_size = 2
    uniform_power = True
    use_sdpa = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = vit_base(
            img_size=crop_size,
            patch_size=patch_size,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            uniform_power=uniform_power,
            use_sdpa=use_sdpa,
        )

    decoder = Decoder(768)
    model = Model(encoder,decoder)
    model = model.to(device)

    transform = T.Compose([
        # T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    batch_size = 4
    frame_count = 8
    mask_patch_size = 80
    dataset = Dataset(dataset_path=f"/content/wind_dataset/train",mode="train",frame_count=frame_count,transform=transform,patch_size=mask_patch_size)
    train_dataloader = DataLoader(dataset,shuffle=True,batch_size = batch_size ,num_workers=48)
    val_dataset = Dataset(dataset_path=f"/content/wind_dataset/validation",mode="val",frame_count=frame_count,transform=transform,patch_size=mask_patch_size)
    val_data_loader = DataLoader(val_dataset,shuffle=True,batch_size = batch_size ,num_workers=48)
    criterion = nn.MSELoss()

    lr = 0.0001
    optimizer = optim.Adam(model.parameters(), lr=lr)

    epochs = 100
    run = wandb.init(
        project="Wind-SSL-Training",
        config={
            "learning_rate": lr,
            "epochs": epochs,
            "encoder":"vit_base",
            "decoder":"CNN+LSTM",
            "criterion":"MSE",
            "frame_count":frame_count,
            "mask_patch_size":80
        },
    )
    trained_model = train_model(model,train_dataloader,val_data_loader, criterion, optimizer, num_epochs=epochs)
    wandb.finish()
