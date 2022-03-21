import argparse
import subprocess
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader

from utils.dataset_utils import TestSpecificDataset
from utils.image_io import save_image_tensor

from net.model import AirNet

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Parameters
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--mode', type=int, default=3,
                        help='0 for denoise, 1 for derain, 2 for dehaze, 3 for all-in-one')

    parser.add_argument('--test_path', type=str, default="test/demo/", help='save path of test images')
    parser.add_argument('--output_path', type=str, default="output/demo/", help='output save path')
    parser.add_argument('--ckpt_path', type=str, default="ckpt/", help='checkpoint save path')
    opt = parser.parse_args()

    if opt.mode == 0:
        opt.batch_size = 3
        ckpt_path = opt.ckpt_path + 'Denoise.pth'
    elif opt.mode == 1:
        opt.batch_size = 1
        ckpt_path = opt.ckpt_path + 'Derain.pth'
    elif opt.mode == 2:
        opt.batch_size = 1
        ckpt_path = opt.ckpt_path + 'Dehaze.pth'
    elif opt.mode == 3:
        opt.batch_size = 5
        ckpt_path = opt.ckpt_path + 'All.pth'

    # construct the output dir
    subprocess.check_output(['mkdir', '-p', opt.output_path])

    np.random.seed(0)
    torch.manual_seed(0)

    # Make network
    torch.cuda.set_device(opt.cuda)
    net = AirNet(opt).cuda()
    net.eval()
    net.load_state_dict(torch.load(ckpt_path, map_location=torch.device(opt.cuda)))

    test_set = TestSpecificDataset(opt)
    testloader = DataLoader(test_set, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    print('Start testing...')
    with torch.no_grad():
        for ([clean_name], degrad_patch) in tqdm(testloader):
            degrad_patch = degrad_patch.cuda()

            restored = net(x_query=degrad_patch, x_key=degrad_patch)

            save_image_tensor(restored, opt.output_path + clean_name[0] + '.png')
