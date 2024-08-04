import os 
import torch
import numpy as np 
from glob import glob 
from PIL import Image
from torch.utils.data import Dataset,DataLoader
from datetime import datetime
from torchvision import transforms as T
from matplotlib import pyplot as plt 
class WindDataset(Dataset):

  def __init__(self,dataset_path,frame_count,mode,transform,patch_size=112):
    self.frame_count = frame_count
    self.mode = mode
    self.transform = transform 
    self.patch_size = patch_size
    self.data = self.process_files(dataset_path)
    print(mode," => Data Count = " , len(self.data.keys()))
    self.x = np.random.randint(0, 224 - self.patch_size)
    self.y = np.random.randint(0, 224 - self.patch_size)

  def process_files(self,dataset_path):
    data_dict = {}
    sub_folder = os.listdir(dataset_path)
    ii = 0
    for folder in sub_folder:
      if os.path.isdir(os.path.join(dataset_path,folder)):
        folders_of_sub_folder = os.listdir(os.path.join(dataset_path,folder))
        for folder2 in folders_of_sub_folder:
          images = os.listdir(os.path.join(dataset_path,folder,folder2))
          images_dict = {}
          for image_path in images:
            date  = image_path.split(".")[0].split("_")[-1]
            images_dict[date] = os.path.join(dataset_path,folder,folder2,image_path)
          sorted_images_dict = sorted(images_dict, key=lambda x: images_dict[x])
          for date in sorted_images_dict:
            if ii not in data_dict.keys():
              data_dict[ii] = []
            data_dict[ii].append(images_dict[date])
          if len(data_dict[ii]) < self.frame_count:
            # repat last image until len_dict equal to frame count 
            # while len(data_dict[ii]) < self.frame_count:
            #   data_dict[ii].append(os.path.join(dataset_path,folder,folder2,image_path))
            # or
            # ignore that folder 
            del data_dict[ii]
          else:
            ii+=1
    return data_dict
  def __len__(self):
    return len(self.data.keys())
  
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


    items = self.data[index]
    if len(items)-self.frame_count == 0:
      start_index = 0
    else:
      start_index = np.random.randint(0,len(items)-self.frame_count)
    
    selected_images = items[start_index:start_index+self.frame_count]
    imgs = []
    masked_patch = []
    main_images = []
    PLOT=False

    for image_path in selected_images:
      im = Image.open(image_path).convert("RGB")
      white_patched_image,masked_place = self.generate_random_patches_mask(im,patch_size=self.patch_size,x=self.x,y=self.y)
      if PLOT:
        plt.subplot(1,3,1)
        plt.title("Image With Mask")
        plt.imshow(white_patched_image,cmap='gray')
        plt.subplot(1,3,2)
        plt.title("Masked_Crop")
        plt.imshow(masked_place,cmap='gray')
        plt.subplot(1,3,3)
        plt.title("GrandTurth")
        plt.imshow(im.resize((224, 224)),cmap='gray')
        plt.show()
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
  train_dataset = WindDataset(dataset_path="/content/wind_dataset/test",frame_count=8,mode="train",transform=transform,patch_size=80)
  train_loader  = DataLoader(dataset=train_dataset , shuffle=True,batch_size = 16)
  for i in range(100):
    for images,targets in train_loader:
      print(f"images:{images.shape},targets:{targets.shape}")
    print(i," => Done!")