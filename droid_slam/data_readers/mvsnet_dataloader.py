import os, cv2
import numpy as np
import torch
from torchvision.transforms import Compose, ToTensor, Resize
from PIL import Image

# from droid_slam.visualization import droid_visualization


def mvs_loader(args, tstamps, poses, ref_disp):
    calib = np.loadtxt(args.calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]

    image_list = sorted(os.listdir(args.imagedir))[::args.stride]
    h1, w1 = args.depth_fusion_size
    images = []
    transform = Compose([Resize((h1, w1)), ToTensor()])
    h0, w0 = None, None
    for t in tstamps.cpu().numpy():
        imfile = image_list[int(t)]
        # image = cv2.imread(os.path.join(args.imagedir, imfile))
        # if len(calib) > 4:
        #     image = cv2.undistort(image, K, calib[4:])
        image = Image.open(os.path.join(args.imagedir, imfile))

        w0, h0 = image.size[:2]
        image = transform(image) #cv2.resize(image, (w1, h1))
        # image = image[:h1-h1%8, :w1-w1%8]
        # image = torch.as_tensor(image).permute(2, 0, 1)
        # image = image.float() / 255.
        images.append(image)
    fx, cx = fx * (w1 / w0), cx * (w1 / w0)
    fy, cy = fy * (h1 / h0), cy * (h1 / h0)
    images = torch.stack(images, dim=0)

    intr_mat = torch.tensor([[fx, 0, cx, 0], [0, fy, cy, 0],
                            [0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.float32, device=poses.device)

    intr_matrices = intr_mat.unsqueeze(0).repeat(poses.size(0), 1, 1)
    proj_stage3 = torch.stack((poses, intr_matrices), dim=1)
    proj_stage2 = proj_stage3.clone()
    proj_stage2[:, 1, :2] *= 0.5
    proj_stage1 = proj_stage2.clone()
    proj_stage1[:, 1, :2] *= 0.5
    proj_stage0 = proj_stage1.clone()
    proj_stage0[:, 1, :2] *= 0.5
    proj_matrices = {"stage1": proj_stage0.unsqueeze(0),
                     "stage2": proj_stage1.unsqueeze(0),
                     "stage3": proj_stage2.unsqueeze(0),
                     "stage4": proj_stage3.unsqueeze(0)}

    ref_depth = 1 / ref_disp
    val_depths = ref_depth[(ref_depth > 0.001) & (ref_depth < 1000)]
    min_d, max_d = torch.quantile(val_depths, torch.tensor([0.0, 0.90], device=val_depths.device)).cpu().numpy()
    d_interval = (max_d - min_d) / 512
    depth_values = torch.arange(0, 512, dtype=torch.float32).unsqueeze(0) * d_interval + min_d
    return images.unsqueeze(0).cuda(), proj_matrices, depth_values.cuda()
