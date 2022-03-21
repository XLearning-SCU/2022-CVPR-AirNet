import os
import random
import copy
from PIL import Image
import numpy as np

from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor

from utils.image_utils import random_augmentation, crop_img
from utils.degradation_utils import Degradation


class TrainDataset(Dataset):
    def __init__(self, args):
        super(TrainDataset, self).__init__()
        self.args = args
        self.rs_ids = []
        self.hazy_ids = []
        self.D = Degradation(args)
        self.de_temp = 0
        self.de_type = self.args.de_type

        self.de_dict = {'denoise_15': 0, 'denoise_25': 1, 'denoise_50': 2, 'derain': 3, 'dehaze': 4}

        self._init_ids()

        self.crop_transform = Compose([
            ToPILImage(),
            RandomCrop(args.patch_size),
        ])

        self.toTensor = ToTensor()

    def _init_ids(self):
        if 'denoise_15' in self.de_type or 'denoise_25' in self.de_type or 'denoise_50' in self.de_type:
            self._init_clean_ids()
        if 'derain' in self.de_type:
            self._init_rs_ids()
        if 'dehaze' in self.de_type:
            self._init_hazy_ids()

        random.shuffle(self.de_type)

    def _init_clean_ids(self):
        clean_ids = []
        name_list = os.listdir(self.args.denoise_dir)
        clean_ids += [self.args.denoise_dir + id_ for id_ in name_list]

        if 'denoise_15' in self.de_type:
            self.s15_ids = copy.deepcopy(clean_ids)
            random.shuffle(self.s15_ids)
            self.s15_counter = 0
        if 'denoise_25' in self.de_type:
            self.s25_ids = copy.deepcopy(clean_ids)
            random.shuffle(self.s25_ids)
            self.s25_counter = 0
        if 'denoise_50' in self.de_type:
            self.s50_ids = copy.deepcopy(clean_ids)
            random.shuffle(self.s50_ids)
            self.s50_counter = 0

        self.num_clean = len(clean_ids)

    def _init_hazy_ids(self):
        hazy = self.args.data_file_dir + "hazy/hazy_outside.txt"
        self.hazy_ids += [self.args.dehaze_dir + id_.strip() for id_ in open(hazy)]

        self.hazy_counter = 0
        self.num_hazy = len(self.hazy_ids)

    def _init_rs_ids(self):
        rs = self.args.data_file_dir + "rainy/rainTrain.txt"
        self.rs_ids += [self.args.derain_dir + id_.strip() for id_ in open(rs)]

        self.rl_counter = 0
        self.num_rl = len(self.rs_ids)

    def _crop_patch(self, img_1, img_2):
        H = img_1.shape[0]
        W = img_1.shape[1]
        ind_H = random.randint(0, H - self.args.patch_size)
        ind_W = random.randint(0, W - self.args.patch_size)

        patch_1 = img_1[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]
        patch_2 = img_2[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]

        return patch_1, patch_2

    def _get_gt_name(self, rainy_name):
        gt_name = rainy_name.split("rainy")[0] + 'gt/norain-' + rainy_name.split('rain-')[-1]
        return gt_name

    def _get_nonhazy_name(self, hazy_name):
        dir_name = hazy_name.split("synthetic")[0] + 'original/'
        name = hazy_name.split('/')[-1].split('_')[0]
        suffix = '.' + hazy_name.split('.')[-1]
        nonhazy_name = dir_name + name + suffix
        return nonhazy_name

    def __getitem__(self, _):
        de_id = self.de_dict[self.de_type[self.de_temp]]

        if de_id < 3:
            if de_id == 0:
                clean_id = self.s15_ids[self.s15_counter]
                self.s15_counter = (self.s15_counter + 1) % self.num_clean
                if self.s15_counter == 0:
                    random.shuffle(self.s15_ids)
            elif de_id == 1:
                clean_id = self.s25_ids[self.s25_counter]
                self.s25_counter = (self.s25_counter + 1) % self.num_clean
                if self.s25_counter == 0:
                    random.shuffle(self.s25_ids)
            elif de_id == 2:
                clean_id = self.s50_ids[self.s50_counter]
                self.s50_counter = (self.s50_counter + 1) % self.num_clean
                if self.s50_counter == 0:
                    random.shuffle(self.s50_ids)

            # clean_id = random.randint(0, len(self.clean_ids) - 1)
            clean_img = crop_img(np.array(Image.open(clean_id).convert('RGB')), base=16)
            clean_patch_1, clean_patch_2 = self.crop_transform(clean_img), self.crop_transform(clean_img)
            clean_patch_1, clean_patch_2 = np.array(clean_patch_1), np.array(clean_patch_2)

            # clean_name = self.clean_ids[clean_id].split("/")[-1].split('.')[0]
            clean_name = clean_id.split("/")[-1].split('.')[0]

            clean_patch_1, clean_patch_2 = random_augmentation(clean_patch_1, clean_patch_2)
            degrad_patch_1, degrad_patch_2 = self.D.degrade(clean_patch_1, clean_patch_2, de_id)
        else:
            if de_id == 3:
                # Rain Streak Removal
                # rl_id = random.randint(0, len(self.rl_ids) - 1)
                degrad_img = crop_img(np.array(Image.open(self.rs_ids[self.rl_counter]).convert('RGB')), base=16)
                clean_name = self._get_gt_name(self.rs_ids[self.rl_counter])
                clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)

                self.rl_counter = (self.rl_counter + 1) % self.num_rl
                if self.rl_counter == 0:
                    random.shuffle(self.rs_ids)
            elif de_id == 4:
                # Dehazing with SOTS outdoor training set
                # hazy_id = random.randint(0, len(self.hazy_ids) - 1)
                degrad_img = crop_img(np.array(Image.open(self.hazy_ids[self.hazy_counter]).convert('RGB')), base=16)
                clean_name = self._get_nonhazy_name(self.hazy_ids[self.hazy_counter])
                clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)

                self.hazy_counter = (self.hazy_counter + 1) % self.num_hazy
                if self.hazy_counter == 0:
                    random.shuffle(self.hazy_ids)
            degrad_patch_1, clean_patch_1 = random_augmentation(*self._crop_patch(degrad_img, clean_img))
            degrad_patch_2, clean_patch_2 = random_augmentation(*self._crop_patch(degrad_img, clean_img))

        clean_patch_1, clean_patch_2 = self.toTensor(clean_patch_1), self.toTensor(clean_patch_2)
        degrad_patch_1, degrad_patch_2 = self.toTensor(degrad_patch_1), self.toTensor(degrad_patch_2)

        self.de_temp = (self.de_temp + 1) % len(self.de_type)
        if self.de_temp == 0:
            random.shuffle(self.de_type)

        return [clean_name, de_id], degrad_patch_1, degrad_patch_2, clean_patch_1, clean_patch_2

    def __len__(self):
        return 400 * len(self.args.de_type)


class DenoiseTestDataset(Dataset):
    def __init__(self, args):
        super(DenoiseTestDataset, self).__init__()
        self.args = args
        self.clean_ids = []
        self.sigma = 15

        self._init_clean_ids()

        self.toTensor = ToTensor()

    def _init_clean_ids(self):
        name_list = os.listdir(self.args.denoise_path)
        self.clean_ids += [self.args.denoise_path + id_ for id_ in name_list]

        self.num_clean = len(self.clean_ids)

    def _add_gaussian_noise(self, clean_patch):
        noise = np.random.randn(*clean_patch.shape)
        noisy_patch = np.clip(clean_patch + noise * self.sigma, 0, 255).astype(np.uint8)
        return noisy_patch, clean_patch

    def set_sigma(self, sigma):
        self.sigma = sigma

    def __getitem__(self, clean_id):
        clean_img = crop_img(np.array(Image.open(self.clean_ids[clean_id]).convert('RGB')), base=16)
        clean_name = self.clean_ids[clean_id].split("/")[-1].split('.')[0]

        noisy_img, _ = self._add_gaussian_noise(clean_img)
        clean_img, noisy_img = self.toTensor(clean_img), self.toTensor(noisy_img)

        return [clean_name], noisy_img, clean_img

    def __len__(self):
        return self.num_clean


class DerainDehazeDataset(Dataset):
    def __init__(self, args, task="derain"):
        super(DerainDehazeDataset, self).__init__()
        self.ids = []
        self.task_idx = 0
        self.args = args

        self.task_dict = {'derain': 0, 'dehaze': 1}
        self.toTensor = ToTensor()

        self.set_dataset(task)

    def _init_input_ids(self):
        if self.task_idx == 0:
            self.ids = []
            name_list = os.listdir(self.args.derain_path + 'input/')
            self.ids += [self.args.derain_path + 'input/' + id_ for id_ in name_list]
        elif self.task_idx == 1:
            self.ids = []
            name_list = os.listdir(self.args.dehaze_path + 'input/')
            self.ids += [self.args.dehaze_path + 'input/' + id_ for id_ in name_list]

        self.length = len(self.ids)

    def _get_gt_path(self, degraded_name):
        if self.task_idx == 0:
            gt_name = degraded_name.replace("input", "target")
        elif self.task_idx == 1:
            dir_name = degraded_name.split("input")[0] + 'target/'
            name = degraded_name.split('/')[-1].split('_')[0] + '.png'
            gt_name = dir_name + name
        return gt_name

    def set_dataset(self, task):
        self.task_idx = self.task_dict[task]
        self._init_input_ids()

    def __getitem__(self, idx):
        degraded_path = self.ids[idx]
        clean_path = self._get_gt_path(degraded_path)

        degraded_img = crop_img(np.array(Image.open(degraded_path).convert('RGB')), base=16)
        clean_img = crop_img(np.array(Image.open(clean_path).convert('RGB')), base=16)

        clean_img, degraded_img = self.toTensor(clean_img), self.toTensor(degraded_img)
        degraded_name = degraded_path.split('/')[-1][:-4]

        return [degraded_name], degraded_img, clean_img

    def __len__(self):
        return self.length


class TestSpecificDataset(Dataset):
    def __init__(self, args):
        super(TestSpecificDataset, self).__init__()
        self.args = args
        self.degraded_ids = []
        self._init_clean_ids(args.test_path)

        self.toTensor = ToTensor()

    def _init_clean_ids(self, root):
        name_list = os.listdir(root)
        self.degraded_ids += [root + id_ for id_ in name_list]

        self.num_img = len(self.degraded_ids)

    def __getitem__(self, idx):
        degraded_img = crop_img(np.array(Image.open(self.degraded_ids[idx]).convert('RGB')), base=16)
        name = self.degraded_ids[idx].split('/')[-1][:-4]

        degraded_img = self.toTensor(degraded_img)

        return [name], degraded_img

    def __len__(self):
        return self.num_img
