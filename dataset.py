import os
from torch.utils.data import DataLoader , Dataset
from PIL import Image
from torchvision import transforms as T
import torch
from glob import glob
from matplotlib import pyplot as plt
import numpy as np
import random
class Dataset:
  def __init__(self,dataset_path,frame_count,transform,patch_size=112):
    self.transform = transform
    self.dataset_path = dataset_path
    self.frame_count = frame_count
    self.patch_size = patch_size
    self.images = glob(os.path.join(dataset_path,"*.png"))

    self.counter = 0
    self.x = np.random.randint(0, 224 - self.patch_size)
    self.y = np.random.randint(0, 224 - self.patch_size)
  def __len__(self):
    return len(self.images)
  def get_list(self,images):
    out = [self.transform(image) for image in images]
    return out
  def generate_random_patches_mask(self,image, patch_size=112, num_patches=1,x=0,y=0):
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
  def __getitem__(self,index):

    self.x = np.random.randint(0, 224 - self.patch_size)
    self.y = np.random.randint(0, 224 - self.patch_size)

    if index == 0 :
        index = 1 

    if index+self.frame_count > len(self.images):
      index = index - abs(len(self.images)-(index+self.frame_count))+1
    if index == 0 :
        index = 1 
    imgs = []
    masked_patch = []
    main_images = []
    for i in range(index,index+self.frame_count):
      image_path = os.path.join(self.dataset_path,"("+str(i)+").png")
      im = Image.open(image_path).convert("RGB")
      white_patched_image,masked_place = self.generate_random_patches_mask(im,patch_size=self.patch_size,x=self.x,y=self.y)
      # plt.subplot(1,2,1)
      # plt.imshow(white_patched_image,cmap='gray')
      # plt.subplot(1,2,2)
      # plt.imshow(masked_place,cmap='gray')
      # plt.show()
      main_images.append(torch.from_numpy(np.array(im.resize((224, 224))))/255.0)
      imgs.append(white_patched_image)
      masked_patch.append(torch.from_numpy(masked_place)/255.0)
    images        = self.get_list(imgs)
    # masked_patches = self.get_list(masked_patch)
    images = torch.stack(images).permute(1, 0, 2, 3)
    masked_patches = torch.stack(masked_patch).permute(0,3,1,2)
    main_images = torch.stack(main_images).permute(0,3,1,2)
    return images, main_images
if __name__ == "__main__":
  transform = T.Compose([
      # T.Resize((224,224)),
      T.ToTensor(),
      T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
  ])
  dataset = Dataset(dataset_path="./train",frame_count=16,transform=transform,patch_size=80)
  data_loader = DataLoader(dataset,shuffle=True,batch_size = 32)
  for images,masked_patches in data_loader:
    print(images.shape , masked_patches.shape)