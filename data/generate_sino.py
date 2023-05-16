import geom
import os
import numpy as np
import torch

if __name__ == '__main__':
    originPath = "/media/wyk/wyk/Data/raws/trainData"
    targetPath = "/media/wyk/wyk/Data/Inpainting/sino_with_padding"
    for item in os.listdir(originPath):
        img = np.fromfile(os.path.join(originPath, item), dtype="float32")
        img = torch.from_numpy(img.reshape(1,1,64,256,256)).cuda()
        sino = geom.ForwardProjection.apply(img)
        sino.cpu().numpy().tofile(os.path.join(targetPath, item))
        print("{} finished with size {}".format(item, list(sino.size())))