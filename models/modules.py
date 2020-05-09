import torch
from torch import nn


class Bottleneck(nn.Module):
    """bottleneck"""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, bn_momentum=0.1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=bn_momentum)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=bn_momentum)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, bn_momentum=0.1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=bn_momentum)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Fuse(nn.Module):

    def __init__(self,stage,out_branches,c,bn_momentum=0.1):
        super(Fuse,self).__init__()
        self.stage = stage
        self.out_branches = out_branches

        self.layers = nn.ModuleList()
        # for each output_branches (i.e. each branch in all cases but the very last one)
        for i in range(self.out_branches):
            self.layers.append(nn.ModuleList())
            for j in range(self.stage):  # for each branch
                if i == j:
                    self.layers[-1].append(nn.Sequential())  # Used in place of "None" because it is callable
                elif i < j:
                    self.layers[-1].append(nn.Sequential(
                        nn.Conv2d(c * (2 ** j), c * (2 ** i), kernel_size=(1, 1), stride=(1, 1), bias=False),
                        nn.BatchNorm2d(c * (2 ** i), eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True),
                        nn.Upsample(scale_factor=(2.0 ** (j - i)), mode='nearest'),
                    ))
                elif i > j:
                    ops = []
                    for k in range(i - j - 1):
                        ops.append(nn.Sequential(
                            nn.Conv2d(c * (2 ** j), c * (2 ** j), kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                                      bias=False),
                            nn.BatchNorm2d(c * (2 ** j), eps=1e-05, momentum=bn_momentum, affine=True,
                                           track_running_stats=True),
                            nn.ReLU(inplace=True),
                        ))
                    ops.append(nn.Sequential(
                        nn.Conv2d(c * (2 ** j), c * (2 ** i), kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                                  bias=False),
                        nn.BatchNorm2d(c * (2 ** i), eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True),
                    ))
                    self.layers[-1].append(nn.Sequential(*ops))
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        x_fused = []
        for i in range(len(self.layers)):
            for j in range(0, self.stage):
                if j == 0:
                    x_fused.append(self.layers[i][0](x[0]))
                else:
                    x_fused[i] = x_fused[i] + self.layers[i][j](x[j])

        for i in range(len(x_fused)):
            x_fused[i] = self.relu(x_fused[i])

        return x_fused

class StageModule(nn.Module):
    def __init__(self, stage, output_branches, c, bn_momentum):
        super(StageModule, self).__init__()
        self.stage = stage
        self.output_branches = output_branches

        self.branches = nn.ModuleList()
        for i in range(self.stage):
            w = c * (2 ** i)
            branch = nn.Sequential(
                BasicBlock(w, w, bn_momentum=bn_momentum),
                BasicBlock(w, w, bn_momentum=bn_momentum),
                BasicBlock(w, w, bn_momentum=bn_momentum),
                BasicBlock(w, w, bn_momentum=bn_momentum),
            )
            self.branches.append(branch)
        
        self.fuse_layers = Fuse(stage,output_branches,c,bn_momentum)
        

    def forward(self, x):
        assert len(self.branches) == len(x)

        x = [branch(b) for branch, b in zip(self.branches, x)]

        x_fused = self.fuse_layers(x)
        return x_fused
        
        
class TransVars(object):
    def __init__(self,out_branches,indices,input_shapes,kernels,strides,paddings,bn_momentum=0.1):
        self.out_branches = out_branches # int
        self.indices = indices # list
        self.input_shapes = input_shapes # list of tuples
        self.kernels = kernels # list of tuples
        self.strides = strides # list of tuples
        self.paddings = paddings # list of tuples
        self.bn_momentum = bn_momentum # float

class Transition(nn.Module):

    def __init__(self,transVar):
        super(Transition,self).__init__()
        self.out_branches = transVar.out_branches
        self.kernels = transVar.kernels
        input_shapes = transVar.input_shapes
        strides = transVar.strides
        paddings = transVar.paddings
        bn_momentum = transVar.bn_momentum
        kernels = transVar.kernels
        self.transitions = nn.ModuleList()

        count = 0
        for i in range(self.out_branches):
            if i in transVar.indices:
                if i == 0 :
                    module = nn.Sequential(
                        nn.Conv2d(input_shapes[count][0],
                                input_shapes[count][1],
                                kernel_size=kernels[count],
                                stride=strides[count],
                                padding=paddings[count],
                                bias=False),
                        nn.BatchNorm2d(input_shapes[count][1],eps=1e-05,momentum=bn_momentum,affine=True,track_running_stats=True),
                        nn.ReLU(inplace=True)
                    )
                    self.transitions.append(module)
                    count+=1
                else:
                    module = nn.Sequential(nn.Sequential(
                        nn.Conv2d(input_shapes[count][0],
                                input_shapes[count][1],
                                kernel_size=kernels[count],
                                stride=strides[count],
                                padding=paddings[count],
                                bias=False),
                        nn.BatchNorm2d(input_shapes[count][1],eps=1e-05,momentum=bn_momentum,affine=True,track_running_stats=True),
                        nn.ReLU(inplace=True)
                    ))
                    self.transitions.append(module)
                    count+=1
            else:
                self.transitions.append(nn.Sequential())

    def forward(self,x):
        out = []
        for i in range(self.out_branches):
            block = self.transitions[i](x[i])
            out.append(block)

        return out