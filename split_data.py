import os

imgs = os.listdir("coco_val_subset/images/")
if not os.path.exists("coco_val_subset/train"):
    os.makedirs("coco_val_subset/train")
    os.makedirs("coco_val_subset/train/masks")
    os.makedirs("coco_val_subset/train/images")

if not os.path.exists("coco_val_subset/test"):
    os.makedirs("coco_val_subset/test")
    os.makedirs("coco_val_subset/test/masks")
    os.makedirs("coco_val_subset/test/images")

if not os.path.exists("coco_val_subset/val"):
    os.makedirs("coco_val_subset/val")
    os.makedirs("coco_val_subset/val/masks")
    os.makedirs("coco_val_subset/val/images")
# Split the dataset into train, val, and test sets


for i in range(len(imgs)):
    if i < len(imgs)*0.8:
       os.rename(f"coco_val_subset/images/{imgs[i]}", f"coco_val_subset/train/images/{imgs[i]}")
       os.rename(f"coco_val_subset/masks/{imgs[i].replace('.jpg','.png')}", f"coco_val_subset/train/masks/{imgs[i].replace('.jpg','.png')}")
    elif i < len(imgs)*0.9:
         os.rename(f"coco_val_subset/images/{imgs[i]}", f"coco_val_subset/val/images/{imgs[i]}")
         os.rename(f"coco_val_subset/masks/{imgs[i].replace('.jpg','.png')}", f"coco_val_subset/val/masks/{imgs[i].replace('.jpg','.png')}")
    else:
        os.rename(f"coco_val_subset/images/{imgs[i]}", f"coco_val_subset/test/images/{imgs[i]}")
        os.rename(f"coco_val_subset/masks/{imgs[i].replace('.jpg','.png')}", f"coco_val_subset/test/masks/{imgs[i].replace('.jpg','.png')}")
        
# Check the number of images in each split
print(f"Train images: {len(os.listdir('coco_val_subset/train/images'))}")
print(f"Val images: {len(os.listdir('coco_val_subset/val/images'))}")
print(f"Test images: {len(os.listdir('coco_val_subset/test/images'))}")