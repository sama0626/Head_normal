
import torch

def index(feat, uv):

    uv = uv.transpose(1, 2)
    uv = uv.unsqueeze(2)
    samples = torch.nn.functional.grid_sample(feat, uv, align_corners=True)
    return samples[:, :, :, 0]

def orthogonal(points, calib, transform=None):

    rot = calib[:, :3, :3]
    trans = calib[:, :3, 3:4]
    pts = torch.baddbmm(trans, rot, points)
    if transform is not None:
        scale = transform[:2, :2]
        shift = transform[:2, 2:3]
        pts[:, :2, :] = torch.baddbmm(shift, scale, pts[:, :2, :])
    return pts

def perspective(points, calib, transform=None):

    rot = calib[:, :3, :3]
    trans = calib[:, :3, 3:4]
    homo = torch.baddbmm(trans, rot, points)
    xy = homo[:, :2, :] / homo[:, 2:3, :]
    if transform is not None:
        scale = transform[:2, :2]
        shift = transform[:2, 2:3]
        xy = torch.baddbmm(shift, scale, xy)
    
    xyz = torch.cat([xy, homo[:, 2:3, :]], 1)
    return xyz