import numpy as np
import torch
import scipy.io as sco
import JITBeijingGeometry as projector
from .options import *

parameters = sco.loadmat(beijingParameterRoot)
parameters = np.array(parameters["projection_matrix"]).astype(np.float32)
parameters = torch.from_numpy(parameters).contiguous()
volumeSize = torch.IntTensor(beijingVolumeSize)
detectorSize = torch.IntTensor(beijingSubDetectorSize)

class ForwardProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        device = input.device
        sino = projector.forward(input, volumeSize.to(device), detectorSize.to(device), parameters.to(device), device.index)
        return sino.reshape(beijingAngleNum, beijingPlanes, beijingSubDetectorSize[1], beijingSubDetectorSize[0]).permute(0,2,1,3).reshape(1, 1, beijingAngleNum*beijingPlanes, beijingSubDetectorSize[1], beijingSubDetectorSize[0])

    @staticmethod
    def backward(ctx, grad):
        device = grad.device
        grad = grad.reshape(beijingAngleNum, beijingSubDetectorSize[1], beijingPlanes, beijingSubDetectorSize[0]).permute(0,2,1,3).reshape(1, 1, beijingAngleNum*beijingPlanes, beijingSubDetectorSize[1], beijingSubDetectorSize[0])
        volume = projector.backward(grad, volumeSize.to(device), detectorSize.to(device), parameters.to(device), device.index)
        return volume

class BackProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        device = input.device
        input = input.reshape(beijingAngleNum, beijingSubDetectorSize[1], beijingPlanes, beijingSubDetectorSize[0]).permute(0,2,1,3).reshape(1, 1, beijingAngleNum*beijingPlanes, beijingSubDetectorSize[1], beijingSubDetectorSize[0])
        volume = projector.backward(input, volumeSize.to(device), detectorSize.to(device), parameters.to(device), device.index)
        return volume

    @staticmethod
    def backward(ctx, grad):
        device = grad.device
        sino = projector.forward(input, volumeSize.to(device), detectorSize.to(device), parameters.to(device), device.index)
        return sino.reshape(beijingAngleNum, beijingPlanes, beijingSubDetectorSize[1], beijingSubDetectorSize[0]).permute(0,2,1,3).reshape(1, 1, beijingAngleNum*beijingPlanes, beijingSubDetectorSize[1], beijingSubDetectorSize[0])

class BeijingGeometry(torch.nn.Module):
    def __init__(self):
        super(BeijingGeometry, self).__init__()
        self.lamb = 10e-3

    def forward(self, x, p):
        residual = ForwardProjection.apply(x) - p
        return -self.lamb * BackProjection.apply(residual)

class BeijingGeometryWithFBP(torch.nn.Module):
    def __init__(self):
        super(BeijingGeometryWithFBP, self).__init__()
        self.lamb = torch.nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.cosWeight = torch.nn.Parameter(self.__cosWeight__(),requires_grad=False)
        self.ramp = torch.nn.Parameter(self.__conv__(beijingSubDetectorSize[0]*beijingPlanes), requires_grad=False)

    def forward(self, x, p):
        residual = ForwardProjection.apply(x) - p
        residual = residual.view(1, 1, beijingAngleNum, beijingSubDetectorSize[1], beijingSubDetectorSize[0]*beijingPlanes)
        residual = residual * self.cosWeight
        residual = residual.view(1, 1, beijingAngleNum * beijingSubDetectorSize[1], beijingSubDetectorSize[0]*beijingPlanes)
        residual = torch.nn.functional.conv2d(residual, self.ramp, stride=1, padding=(0, int(beijingSubDetectorSize[0]*beijingPlanes/2)))
        residual = residual.view(1, 1, beijingAngleNum*beijingPlanes, beijingSubDetectorSize[1], beijingSubDetectorSize[0])
        return x - self.lamb * BackProjection.apply(residual)

    def __cosWeight__(self):
        cosine = np.zeros([1,1,beijingAngleNum,beijingSubDetectorSize[1],beijingSubDetectorSize[0]*beijingPlanes], dtype=np.float32)
        mid = np.array([beijingSubDetectorSize[1],beijingSubDetectorSize[0]*beijingPlanes]) / 2
        for i in range(beijingSubDetectorSize[1]):
            for j in range(beijingSubDetectorSize[0]*beijingPlanes):
                cosine[...,i,j] = beijingSDD / np.sqrt(beijingSDD**2 + (i-mid[1])**2 + (j-mid[0])**2)
        return torch.from_numpy(cosine)

    def __conv__(self, projWidth):
        filter = np.ones([1,1,1,projWidth+1], dtype=np.float32)
        mid = np.floor(projWidth / 2)
        for i in range(projWidth+1):
            if (i - mid) % 2 == 0:
                filter[...,i] = 0
            else:
                filter[...,i] = -0.5 / (np.pi * np.pi * (i - mid) * (i - mid))
            if i == mid:
                filter[...,i] = 1 / 8
        return torch.from_numpy(filter)
