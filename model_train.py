import numpy as np
import torch
import cv2
import torchvision
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.utils import make_grid, draw_bounding_boxes
from torchvision.io import read_image
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

batch_size = 2
image_size = [640,480]

device = torch.device('cpu')
model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)

data_path="data/horizontal/shoe"

def data_loader(path, num_objs):
    batch_imgs = []
    batch_data = []

    for i in range(num_objs):
        data = {}

        img = np.load(f'{path}/color_{i}.npy')
        img = cv2.resize(img, image_size, cv2.INTER_LINEAR)
        img = torch.as_tensor(img, dtype=torch.float32)
        mask = np.load(f'{path}/label/mask_{i}.npy')
        bbs = np.load(f'{path}/label/bbs_{i}.npy')

        N = bbs.shape[0]

        ## TODO: implement having multiple objects/labels in the image
        if N != 1:
            continue

        bbs = bbs.reshape((1,4))
        bbs[0,2] += bbs[0,0]
        bbs[0,3] += bbs[0,1]

        data["boxes"] = torch.as_tensor(bbs, dtype=torch.float32)
        data["labels"] = torch.ones((1,), dtype=torch.int64)
        data["masks"] = torch.as_tensor(mask, dtype=torch.uint8)
            
        batch_imgs.append(img)
        batch_data.append(data)

    batch_imgs = torch.stack([torch.as_tensor(d) for d in batch_imgs],0)
    batch_imgs = batch_imgs.swapaxes(1, 3).swapaxes(2, 3)

    return batch_imgs, batch_data


# def show(imgs):
#     if not isinstance(imgs, list):
#         imgs = [imgs]
#     fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
#     for i, img in enumerate(imgs):
#         img = img.detach()
#         img = F.to_pil_image(img)
#         axs[0, i].imshow(np.asarray(img))
#         axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    
#     plt.show()

# test1 = read_image(str(Path('data/horizontal/shoe') / 'color_0.jpeg'))

# boxes = torch.tensor([[50, 50, 100, 200], [210, 150, 350, 430]], dtype=torch.float)
# colors = ["blue", "yellow"]
# result = draw_bounding_boxes(test1, boxes, colors=colors, width=5)
 
# show(result)

in_features = model.roi_heads.box_predictor.cls_score.in_features 
model.roi_heads.box_predictor = FastRCNNPredictor(in_features,num_classes=1)

model.to(device)

optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-5)

model.train()


for i in range(200):
    print(i)
    images, targets = data_loader(data_path, 20)
    images = list(image.to(device) for image in images)
    targets=[{k: v.to(device) for k,v in t.items()} for t in targets]
    

    optimizer.zero_grad()
    loss_dict = model(images, targets)
    losses = sum(loss for loss in loss_dict.values())

    losses.backward()
    optimizer.step()

    print(i,'loss:', losses.item())
    if i%200==0:
        torch.save(model.state_dict(), str(i)+".torch")
        print("Save model to:",str(i)+".torch")