import os
from pycocotools.coco import COCO
import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings("ignore")

# === Configuration ===
annFile = 'annotations/instances_val2014.json'
coco=COCO(annFile)
SAVE_DIR = 'coco_val_subset'
max_images = 8000 

selected_cat_ids = [1, 2, 3, 4, 5]
catid_to_label = {
    1:"person",
    2:"bicycle",
    3:"car",
    4:"motorcycle",
    5:"airplane",
    
}
ann_ids = coco.getAnnIds(catIds=selected_cat_ids)

# Get image IDs corresponding to these annotations
img_ids = set()
for ann_id in ann_ids:
    ann = coco.loadAnns(ann_id)[0]
    img_ids.add(ann['image_id'])

imgs = coco.loadImgs(img_ids)


if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
if not os.path.exists(f"{SAVE_DIR}/images"):
    os.makedirs(f"{SAVE_DIR}/images")
if not os.path.exists(f"{SAVE_DIR}/masks"):
    os.makedirs(f"{SAVE_DIR}/masks")


def process_ImgAndMask(img,SAVE_DIR):
    I = io.imread("val2014/"+img['file_name'])


    #3) Exclude Crowded Annotations                                                                                                                                                                                                                                                                                                                                                                      
    ann_ids = coco.getAnnIds(imgIds=img['id'], catIds=[1,2,3,4,5], iscrowd=False)
    anns = coco.loadAnns(ann_ids)
    mask = np.zeros((img["height"], img['width']), dtype=np.uint8)

    #4 Images with No Valid Annotations
    if len(anns) == 0:
        return False  # skip image

    # --- Generate Mask ---
    for ann in anns:
        cat_id = ann['category_id']

        # 2) Incorrect/Corrupted Annotations
        if 'segmentation' not in ann or ann['segmentation'] == []:
            return False  # skip bad annotation

        if ann['category_id'] not in catid_to_label:
            return False  # skip irrelevant category

        m = coco.annToMask(ann)
        #1) Mask overlap handling --> simply last-label wins (for simplicity).
        mask[m == 1] = cat_id

    # Save Mask
    mask_filename = img['file_name'].replace('.jpg', '.png')
    mask_path = os.path.join(f"{SAVE_DIR}/masks", mask_filename)
    io.imsave(mask_path, mask)
    io.imsave(os.path.join(f"{SAVE_DIR}/images", img['file_name']), I)
    return True,[ann['category_id'] for ann in anns]
  

count=0
img_ids = []
cat_ids = []

with tqdm(total=max_images, desc="Processing Images") as pbar:
    for img in imgs:
        if count > max_images:
            break
        try:
            sucess,cat_id= process_ImgAndMask(img,SAVE_DIR)
            if sucess:
                img_ids.append(img['file_name'])
                cat_ids.append(cat_id)
                count += 1
                pbar.update(1)
        except Exception as e:
            print(e)
            continue    

print(f"Total Images: {len(img_ids)}")
df = pd.DataFrame({'image_id': img_ids, 'categories': cat_ids})
df.to_csv(f"coco_val_subset.csv", index=False)
print("Saved CSV file with image ids and categories")