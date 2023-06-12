import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import pickle
import os, os.path as osp

import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        m = m.astype(np.uint8)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        img = cv2.drawContours(img, contours, 0, (0xFF, 0x00, 0x00, 0xFF), 1, cv2.LINE_8)
    ax.imshow(img)

if __name__ == '__main__':
    image = cv2.imread('/home/tseng/workspace/project/sam/images/dog.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(20,20))
    plt.imshow(image)
    plt.axis('off')
    plt.show()

    if osp.exists('mask.pkl'):
        with open('mask.pkl', 'rb') as file:
            masks = pickle.load(file)
    else:
        sam_checkpoint = "/home/tseng/workspace/model_zoo/sam/sam_vit_b_01ec64.pth"
        model_type = "vit_b"

        device = "cuda"

        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)

        # mask_generator = SamAutomaticMaskGenerator(sam)
        # masks = mask_generator.generate(image)

        mask_generator_2 = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=32,
            pred_iou_thresh=0.78,
            stability_score_thresh=0.85,
            crop_n_layers=0,
            crop_n_points_downscale_factor=1,
            min_mask_region_area=0,  # Requires open-cv to run post-processing
        )
        masks = mask_generator_2.generate(image)

        with open('mask.pkl', 'wb') as file:
            pickle.dump(masks, file)

    plt.figure(figsize=(20,20))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    plt.show() 