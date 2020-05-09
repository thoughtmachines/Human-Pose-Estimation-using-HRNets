import torch
from torch import nn
import cv2
import warnings
from vidgear.gears import CamGear
from models.modules import BasicBlock, Bottleneck, TransVars, Transition, StageModule

warnings.filterwarnings("ignore",category=UserWarning)

class HRNet(nn.Module):
    def __init__(self, c=32, nof_joints=17, bn_momentum=0.1):
        super(HRNet, self).__init__()

        # Input (stem net)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(64, eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)

        # Stage 1 (layer1)      - First group of bottleneck (resnet) modules
        downsample = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True),
        )
        self.layer1 = nn.Sequential(
            Bottleneck(64, 64, downsample=downsample),
            Bottleneck(256, 64),
            Bottleneck(256, 64),
            Bottleneck(256, 64),
        )

        
        # Fusion layer 1 (transition1)      - Creation of the first two branches (one full and one half resolution)
        # setting up variables for transition
        trans1var = TransVars(out_branches=2,
                                indices=[0,1],
                                input_shapes=[(256,c),(256,c * (2 ** 1))],
                                kernels=[(3,3),(3,3)],
                                strides=[(1,1),(2,2)],
                                paddings=[(1,1),(1,1)],
                                bn_momentum=bn_momentum
                                )

        self.transition1 = Transition(trans1var)

        # Stage 2 (stage2)      - Second module with 1 group of bottleneck (resnet) modules. This has 2 branches
        self.stage2 = nn.Sequential(
            StageModule(stage=2, output_branches=2, c=c, bn_momentum=bn_momentum),
        )

        # Fusion layer 2 (transition2)      - Creation of the third branch (1/4 resolution)
        # setting up variables for transition
        trans2var = TransVars(out_branches=3,
                                indices=[2],
                                input_shapes=[(c * (2 ** 1),c * (2 ** 2))],
                                kernels=[(3,3)],
                                strides=[(2,2)],
                                paddings=[(1,1)],
                                bn_momentum=bn_momentum
                                )

        self.transition2 = Transition(trans2var)

        # Stage 3 (stage3)      - Third module with 4 groups of bottleneck (resnet) modules. This has 3 branches
        self.stage3 = nn.Sequential(
            StageModule(stage=3, output_branches=3, c=c, bn_momentum=bn_momentum),
            StageModule(stage=3, output_branches=3, c=c, bn_momentum=bn_momentum),
            StageModule(stage=3, output_branches=3, c=c, bn_momentum=bn_momentum),
            StageModule(stage=3, output_branches=3, c=c, bn_momentum=bn_momentum),
        )

        
        # Fusion layer 3 (transition3)      - Creation of the fourth branch (1/8 resolution)
        # setting up variables for transition
        trans3var = TransVars(out_branches=4,
                                indices=[3],
                                input_shapes=[(c * (2 ** 2),c * (2 ** 3))],
                                kernels=[(3,3)],
                                strides=[(2,2)],
                                paddings=[(1,1)],
                                bn_momentum=bn_momentum
                                )

        self.transition3 = Transition(trans3var)


        # Stage 4 (stage4)      - Fourth module with 3 groups of bottleneck (resnet) modules. This has 4 branches
        self.stage4 = nn.Sequential(
            StageModule(stage=4, output_branches=4, c=c, bn_momentum=bn_momentum),
            StageModule(stage=4, output_branches=4, c=c, bn_momentum=bn_momentum),
            StageModule(stage=4, output_branches=1, c=c, bn_momentum=bn_momentum),
        )

        # Final layer (final_layer)
        self.final_layer = nn.Conv2d(c, nof_joints, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.layer1(x)

        x = self.transition1([x,x])

        x = self.stage2(x)
        x = self.transition2([x[0],x[1],x[-1]])

        x = self.stage3(x)

        x = self.transition3([x[0],x[1],x[2],x[-1]])

        x = self.stage4(x)

        x = self.final_layer(x[0])
        return x


if __name__ == '__main__':
    # model = HRNet(48, 17, 0.1)
    model = HRNet(32, 17, 0.1)

    # print(model)

    model.load_state_dict(
        # torch.load('./weights/pose_hrnet_w48_384x288.pth')
        torch.load('../weights/pose_hrnet_w32_256x192.pth')
    )
    print('ok!!')

    if torch.cuda.is_available() and False:
        torch.backends.cudnn.deterministic = True
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    print(device)

    model = model.to(device)

    y = model(torch.ones(1, 3, 384, 288).to(device))
    print(y.shape)
    print(torch.min(y).item(), torch.mean(y).item(), torch.max(y).item())\

    print("####################")

