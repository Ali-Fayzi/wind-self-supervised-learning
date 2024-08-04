# infrenece
import cv2
import os
from PIL import Image
import numpy as np 
import torch 
from torch import nn 
from torchvision import transforms as T 
from model import vit_base , Decoder
from matplotlib import pyplot as plt 
import matplotlib
matplotlib.use('TkAgg')
def generate_random_patches_mask(image, patch_size=112, num_patches=1,x=0,y=0):
    image = image.resize((224, 224))
    img_array = np.array(image)
    mask = np.zeros((224, 224), dtype=np.uint8)
    masked_place = None
    for _ in range(num_patches):
        masked_place = img_array[y:y+patch_size, x:x+patch_size]
        mask[y:y+patch_size, x:x+patch_size] = 1
    masked_img_array = img_array.copy()
    masked_img_array[mask == 1] = 0
    masked_image = Image.fromarray(masked_img_array)
    white_patched_img_array = img_array.copy()
    white_patched_img_array[mask == 1] = 0
    white_patched_image = Image.fromarray(white_patched_img_array)
    mask_image = Image.fromarray((mask * 255).astype(np.uint8))
    return white_patched_image,masked_place

def get_list(images):
    out = [transform(image) for image in images]
    return out
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
        
    transform = T.Compose([
        # T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    crop_size = 224
    patch_size = 16
    num_frames = 8
    tubelet_size = 2
    uniform_power = True
    use_sdpa = True
    device = torch.device("cpu")#torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


    model.load_state_dict(torch.load("checkpoint/model_7_0.0016.pth",map_location=device))
    model.eval()

    random_index = np.random.randint(2,3000)
    images = []
    images_copy = []
    labels = []
    main_imags = []
    dataset_patch_size = 80
    x = np.random.randint(0, 224 - dataset_patch_size)
    y = np.random.randint(0, 224 - dataset_patch_size)
    
    for i in range(random_index,random_index+8):
        image_name = f"./val/({i}).png"
        im = Image.open(image_name).convert("RGB")
        main_imags.append(np.array(im))
        white_patched_image,masked_place = generate_random_patches_mask(im,patch_size=80,x=x,y=y)
        images.append(white_patched_image)
        labels.append(masked_place)

        images_copy.append(np.array(white_patched_image))

    images        = get_list(images)
    images = torch.stack(images).permute(1, 0, 2, 3)
    print(images.shape)
    images = torch.unsqueeze(images,0)
    images = images.to(device)
    output = model(images)


    output = torch.squeeze(output,0)
    # output = output.permute(1,0,2,3)
    output  = output.detach().cpu().numpy()
    for input,out,target,label in zip(images_copy,output,main_imags,labels):
        out = out.transpose(1,2,0)* 255.0 
        out = out.astype(np.uint8) 

        target_width,target_height,target_channel = target.shape
        out = cv2.resize(out,(target_height,target_width))
        # print(input.shape,out.shape,target.shape)
        plt.subplot(1,4,1)
        plt.title("Input Image")
        plt.imshow(input,cmap='gray')
        plt.subplot(1,4,2)
        plt.title("Label")
        plt.imshow(label,cmap='gray')
        plt.subplot(1,4,3)
        plt.title("Net Output")
        plt.imshow(out,cmap='gray')
        plt.subplot(1,4,4)
        plt.title("Target")
        plt.imshow(target,cmap='gray')
        plt.show()

