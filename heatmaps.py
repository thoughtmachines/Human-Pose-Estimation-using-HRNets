import torch
import cv2
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import numpy as np
import warnings
from vidgear.gears import CamGear
from models.hrnet import HRNet

warnings.filterwarnings("ignore",category=UserWarning)

if __name__ == "__main__":

    model = HRNet(32, 17, 0.1)

    model.load_state_dict(
        torch.load('weights/mod_pose_hrnet_w32_256x192.pth')
    )
    print('ok!!')

    if torch.cuda.is_available() and False:
        torch.backends.cudnn.deterministic = True
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    print(device)

    model = model.to(device)

    video = CamGear(0).start()

    frame = video.read()

    image = cv2.resize(
                    frame,
                    (256,192),  # (width, height)
                    interpolation=cv2.INTER_CUBIC
                )
    orig = image

    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    image = transform(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).unsqueeze(dim=0)

    image = image.to(device)

    out = model(image)
    out = out.detach()

    t1 = transforms.Compose([
                transforms.ToTensor(),
            ])

    image = t1(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)).unsqueeze(dim=0)
    image = image.to(device)

    image = image.permute(0,2,3,1)
    out = out.permute(1,0,2,3)


    fig, ax = plt.subplots(nrows=3, ncols=6)
    i = 0
    for row in ax:
        for col in row:
            if i == 0:
                col.imshow(image[0])
            else:
                res = cv2.resize(
                    out[i-1][0].numpy(),
                    (256,192),  # (width, height)
                    interpolation=cv2.INTER_CUBIC
                )
                fin = cv2.addWeighted(np.dstack((res,res,res)), 0.73, image[0].numpy(), 0.27, 0)
                fin = cv2.cvtColor(fin, cv2.COLOR_BGR2GRAY)
                #fin = cv2.applyColorMap(fin, cv2.COLORMAP_JET)
                col.imshow(fin)
            i+=1

    plt.show()

    # 17 x 1 x 256x 192

