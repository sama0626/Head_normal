import numpy as np
import cv2
import torch
from data.NormalNet import NormalNet
import torchvision.transforms as transforms
from PIL import Image
from lib.model.options import BaseOptions
opt = BaseOptions().parse()

to_tensor = transforms.Compose([
            transforms.Resize(opt.loadSize),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

cuda = torch.device('cuda:%d' % opt.gpu_id)

for path in np.arange(1, 9):
    render = {}
    render_list = []
    normal_list = []

    mask_path = r'./image/%s_mask.png' % (path)
    mask_data = np.round((cv2.imread(mask_path)[:, :, 0]).astype(np.float32) / 255.)  # (1536, 1024)
    mask_data_padded = np.zeros((max(mask_data.shape), max(mask_data.shape)), np.float32)  # (1536, 1536)
    mask_data_padded[:, mask_data_padded.shape[0] // 2 - min(mask_data.shape) // 2:mask_data_padded.shape[0] // 2 + min(
        mask_data.shape) // 2] = mask_data
    mask_data_padded = cv2.resize(mask_data_padded, (opt.loadSize, opt.loadSize), interpolation=cv2.INTER_NEAREST)
    mask_data_padded = Image.fromarray(mask_data_padded)
    mask_data_padded = transforms.ToTensor()(mask_data_padded).float()
    render['mask'] = mask_data_padded

    save_path = r'./image/%s_normal.png' % (path)

    image_path = r'./image/%s.png' % (path)
    image = cv2.imread(image_path)[:, :, ::-1]
    image_padded = np.zeros((max(image.shape), max(image.shape), 3), np.uint8)
    image_padded[:,
    image_padded.shape[0] // 2 - min(image.shape[:2]) // 2:image_padded.shape[0] // 2 + min(image.shape[:2]) // 2,
    :] = image

    image_padded = cv2.resize(image_padded, (opt.loadSize, opt.loadSize))
    image_padded = Image.fromarray(image_padded)
    image_padded = to_tensor(image_padded)


    image_padded = mask_data_padded.expand_as(image_padded) * image_padded

    render_list.append(image_padded)

    render['img'] = torch.stack(render_list, dim=0)
    render['img'] = render['img'].to(cuda)

    normal_list.append(image_padded)
    render['normal_F'] = torch.stack(normal_list, dim=0)
    render['normal_F'] = render['normal_F'].to(cuda)

    with torch.no_grad():
        netG = NormalNet(opt)
        netG.to(cuda)
        model_path = './checkpoints/Train_Normal_epoch_27_1984'
        netG.load_state_dict(torch.load(model_path, map_location=cuda))
        netG.eval()

        res, error = netG.forward(render)

        img = np.uint8((np.transpose(res[0].cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :,
                       ::-1] * 255.0)

        cv2.imwrite(save_path, img)









