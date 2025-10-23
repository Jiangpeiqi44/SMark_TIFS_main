import os
import numpy as np
from PIL import Image
import logging
import cv2
import sys
import random
sys.path.append(".")
sys.path.append("..")
from torch.utils.data import Dataset
import torchvision.transforms as transforms
# from utils.library.bi_online_generation import random_get_hull
# import albumentations as alb
import torch

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]
# class RandomDownScale(alb.core.transforms_interface.ImageOnlyTransform):
# 	def apply(self,img,**params):
# 		return self.randomdownscale(img)

# 	def randomdownscale(self,img):
# 		keep_ratio=True
# 		keep_input_shape=True
# 		H,W,C=img.shape
# 		ratio_list=[2,4]
# 		r=ratio_list[np.random.randint(len(ratio_list))]
# 		img_ds=cv2.resize(img,(int(W/r),int(H/r)),interpolation=cv2.INTER_NEAREST)
# 		if keep_input_shape:
# 			img_ds=cv2.resize(img_ds,(W,H),interpolation=cv2.INTER_LINEAR)

# 		return img_ds
     
# def get_blend_mask(mask):
# 	H,W=mask.shape
# 	size_h=np.random.randint(192,257)
# 	size_w=np.random.randint(192,257)
# 	mask=cv2.resize(mask,(size_w,size_h))
# 	kernel_1=random.randrange(5,26,2)
# 	kernel_1=(kernel_1,kernel_1)
# 	kernel_2=random.randrange(5,26,2)
# 	kernel_2=(kernel_2,kernel_2)
	
# 	mask_blured = cv2.GaussianBlur(mask, kernel_1, 0)
# 	mask_blured = mask_blured/(mask_blured.max())
# 	mask_blured[mask_blured<1]=0
	
# 	mask_blured = cv2.GaussianBlur(mask_blured, kernel_2, np.random.randint(5,46))
# 	mask_blured = mask_blured/(mask_blured.max())
# 	mask_blured = cv2.resize(mask_blured,(W,H))
# 	return mask_blured.reshape((mask_blured.shape+(1,)))

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    image_paths = []
    assert os.path.isdir(dir), '[*]{} is not a valid directory'.format(dir)
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root,fname)
                image_paths.append(path)                
    assert len(image_paths) > 0, '[*]The number of input images should not zero'
    return image_paths
    

def select_dataset(dir, max_num, rand_select, rand_seed):
    image_paths = []
    assert os.path.isdir(dir), '[*]{} is not a valid directory'.format(dir)
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root,fname)
                image_paths.append(path)
    print("[*]Loaded {} original images, selected {} images for watermarking".format(len(image_paths), max_num))
    assert len(image_paths) >= max_num, '[*]Total loaded images number should bigger than selected images'
    if rand_select=='Yes':
        np.random.seed(rand_seed)
        selected_imgpaths = np.random.choice(image_paths, max_num)
    else:
        selected_imgpaths = image_paths[:max_num]
    return selected_imgpaths


def label_dataset(dir, label):
    data_paths = []
    assert os.path.isdir(dir), '[*]{} is not a valid directory'.format(dir)
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root,fname)
                data_paths.append((path, label))                
    assert len(data_paths) > 0, '[*]The number of input images should not zero'
    return data_paths



class TrainingDataset(Dataset):
    def __init__(self, root):
        print("[*]Loading Images from {}".format(root))
        self.image_paths = sorted(make_dataset(root))
        print('[*]{} images have been loaded'.format(len(self.image_paths)))
        self.transforms = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor()
        ])
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        try:
            image = Image.open(image_path).convert('RGB')
            return self.transforms(image)
        except:
            self.__getitem__(index + 1)


class TrainingDatasetWithPath(Dataset):
    def __init__(self, root, size = 256):
        self.image_paths = sorted(root)
        # print('[*]{} images have been loaded'.format(len(self.image_paths)))
        logging.info('[*]{} images have been loaded'.format(len(self.image_paths)))
        self.transforms = transforms.Compose([
            transforms.Resize((size,size)),
            transforms.ToTensor()
        ])
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        try:
            image = Image.open(image_path).convert('RGB')
            return self.transforms(image)
        except:
            self.__getitem__(index + 1)

class TrainingDatasetWithNorm(Dataset):
    def __init__(self, root, size = 256):
        self.image_paths = sorted(root)
        # print('[*]{} images have been loaded'.format(len(self.image_paths)))
        logging.info('[*]{} images have been loaded'.format(len(self.image_paths)))
        self.transforms = transforms.Compose([
            transforms.Resize((size,size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        try:
            image = Image.open(image_path).convert('RGB')
            return self.transforms(image)
        except:
            self.__getitem__(index + 1)

class TrainingDatasetWithStarGAN(Dataset):
    def __init__(self, root, attr_path, selected_attrs, size = 256):
        self.image_paths = sorted(root)
        self.image_bases = [os.path.basename(i) for i in self.image_paths]
        self.image_bases_np = np.array(self.image_bases)
        # print('[*]{} images have been loaded'.format(len(self.image_paths)))
        logging.info('[*]{} images have been loaded'.format(len(self.image_paths)))
        self.transforms = transforms.Compose([
            transforms.Resize((size,size)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        self.attr_path = attr_path
        self.selected_attrs = selected_attrs
        self.labels = [[0]*5]*len(self.image_paths)
        self.attr2idx = {}
        self.idx2attr = {}
        self.preprocess()
        assert(len(self.image_paths) == len(self.labels))
    
    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        # print(lines)
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            label = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(values[idx] == '1')

            if filename in self.image_bases:
                index = np.where(self.image_bases_np==filename)
                # print(index[0][0], filename)
                self.labels[index[0][0]] = label

        print('Finished preprocessing the CelebA dataset...')

    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]
        try:
            image = Image.open(image_path).convert('RGB')
            return self.transforms(image), torch.FloatTensor(label)
        except:
            self.__getitem__(index + 1)

# class TrainingDatasetWithLm(Dataset):
#     def __init__(self, root, size = 256):
#         self.size = size
#         self.image_paths = sorted(root)
#         # print('[*]{} images have been loaded'.format(len(self.image_paths)))
#         # self.lm_paths = [img_path.replace('CelebA-HQ-img','CelebA-HQ-img-lm') for img_path in self.image_paths]
#         self.image_paths = [img_path for img_path in self.image_paths if os.path.isfile(img_path.replace('CelebA-HQ-img','CelebA-HQ-img-lm').replace('.jpg','.npy'))]
#         logging.info('[*]{} images have been loaded'.format(len(self.image_paths)))
#         self.transforms = transforms.Compose([
#             transforms.Resize((size,size)),
#             transforms.ToTensor()
#         ])
#         self.source_transforms = self.get_source_transforms()
#     def __len__(self):
#         return len(self.image_paths)
#     def __getitem__(self, index):
#         image_path = self.image_paths[index]
#         try:
#             image = np.array(Image.open(image_path).convert('RGB'))
#             scale = image.shape[0]/self.size
#             image = cv2.resize(np.array(image), (self.size, self.size), interpolation=cv2.INTER_LINEAR) #.astype('float32')
#             # 这里就排除掉未提取成功的人脸
#             lm = np.load(image_path.replace('.jpg','.npy').replace('CelebA-HQ-img','CelebA-HQ-img-lm'))[0]
#             lm = self.reorder_landmark(lm)
#             if np.random.rand()<0.25:
#                 lm = lm[:68]
#             # 默认假设256分辨率下
#             lm = lm / scale
#             logging.disable(logging.FATAL)
#             mask = random_get_hull(lm, image)[:,:,0]
#             logging.disable(logging.NOTSET)
#             img_aug = self.source_transforms(image=image.astype(np.uint8))['image']
#             img_aug, mask = self.randaffine(img_aug, mask)
#             mask = get_blend_mask(mask)
#             fn = {}
#             img = image/255
#             img_aug = img_aug/255
#             # print(img.shape, mask.shape)
#             fn['img'] = img.transpose((2,0,1))
#             fn['mask'] = mask.transpose((2,0,1))
#             fn['img_aug'] = img_aug.transpose((2,0,1))
#             return fn
#         except Exception as e:
#             print(e)
#             self.__getitem__(index + 1)

#     def get_source_transforms(self):
#         return alb.Compose([
# 				alb.Compose([
# 						alb.RGBShift((-20,20),(-20,20),(-20,20),p=0.3),
# 						alb.HueSaturationValue(hue_shift_limit=(-0.3,0.3), sat_shift_limit=(-0.3,0.3), val_shift_limit=(-0.3,0.3), p=1),
# 						alb.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1,0.1), p=1),
# 					],p=1),
	
# 				alb.OneOf([
# 					RandomDownScale(p=1),
# 					alb.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1),
# 				],p=1),
				
# 			], p=1.)


#     def reorder_landmark(self, landmark):
#         landmark_add=np.zeros((13,2))
#         for idx,idx_l in enumerate([77,75,76,68,69,70,71,80,72,73,79,74,78]):
#             landmark_add[idx]=landmark[idx_l]
#             landmark[68:]=landmark_add
#             return landmark

#     def randaffine(self,img,mask):
#             f=alb.Affine(
#                     translate_percent={'x':(-0.03,0.03),'y':(-0.015,0.015)},
#                     scale=[0.95,1/0.95],
#                     fit_output=False,
#                     p=1)
                
#             g=alb.ElasticTransform(
#                     alpha=50,
#                     sigma=7,
#                     alpha_affine=0,
#                     p=1,
#                 )

#             transformed=f(image=img,mask=mask)
#             img=transformed['image']
            
#             mask=transformed['mask']
#             transformed=g(image=img,mask=mask)
#             mask=transformed['mask']
#             return img, mask
    
#     # def collate_fn(self,batch):
# 	# 	img_f,img_r=zip(*batch)
# 	# 	data={}
# 	# 	data['img']=torch.cat([torch.tensor(img_r).float(),torch.tensor(img_f).float()],0)
# 	# 	data['label']=torch.tensor([0]*len(img_r)+[1]*len(img_f))
# 	# 	return data


class InjectionDataset(Dataset):
    def __init__(self, root, max_num, rand_select, rand_seed):
        print("[*]Loading Images from {}".format(root))
        self.image_paths = sorted(select_dataset(root, max_num, rand_select, rand_seed))
        print('[*]{} images have been loaded'.format(len(self.image_paths)))
        self.transforms = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor()
        ])
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        return self.transforms(image)

class InjectionDatasetWithPath(Dataset):
    def __init__(self, root, max_num):
        self.image_paths = sorted(root)[:max_num]
        # logging.info('[*]{} images have been loaded'.format(len(self.image_paths)))
        print('[*]{} images have been loaded'.format(len(self.image_paths)))
        self.transforms = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor()
        ])
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        return self.transforms(image)

class InjectionDatasetWithStarGAN(Dataset):
    def __init__(self, root, attr_path, selected_attrs, max_num, size=256):
        self.image_paths = sorted(root)[:max_num]
        self.image_bases = [os.path.basename(i) for i in self.image_paths]
        self.image_bases_np = np.array(self.image_bases)
        # print('[*]{} images have been loaded'.format(len(self.image_paths)))
        logging.info('[*]{} images have been loaded'.format(len(self.image_paths)))
        self.transforms = transforms.Compose([
            transforms.Resize((size,size)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        self.attr_path = attr_path
        self.selected_attrs = selected_attrs
        self.labels = [[0,0,0,0,1]]*len(self.image_paths)
        self.attr2idx = {}
        self.idx2attr = {}
        self.preprocess()
        assert(len(self.image_paths) == len(self.labels))
    
    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        # print(lines)
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            label = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(values[idx] == '1')

            if filename in self.image_bases:
                index = np.where(self.image_bases_np==filename)
                # print(index[0][0], filename)
                self.labels[index[0][0]] = label

        print('Finished preprocessing the CelebA dataset...')

    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]
        try:
            image = Image.open(image_path).convert('RGB')
            return self.transforms(image), torch.FloatTensor(label)
        except:
            self.__getitem__(index + 1)

class AnalysisDataset(Dataset):
    def __init__(self, root, perturbation):
        print("[*]Loading Images from {}".format(root))
        self.image_paths = sorted(make_dataset(root))
        print('[*]{} images have been loaded'.format(len(self.image_paths)))
        if perturbation == 'Yes':
            print("[*]Loading image with perturbation")
            self.transforms = transforms.Compose([
                transforms.RandomResizedCrop(size=256, scale=(0.6, 1.0)),
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.01)
            ])
        else:
            print("[*]Loading image without perturbation")
            self.transforms = transforms.Compose([
                transforms.Resize((256,256)),
                transforms.ToTensor()
            ])
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        return self.transforms(image)



class EvaluationDataset(Dataset):
    def __init__(self, pos_root, neg_root):
        self.data_list = []

        print("[*]Loading positive images from {}".format(pos_root))
        pos_paths = sorted(label_dataset(pos_root, label=1))
        self.data_list.extend(pos_paths)

        print("[*]Loading negative images from {}".format(neg_root))
        neg_paths = sorted(label_dataset(neg_root, label=0))
        self.data_list.extend(neg_paths)

        print('[*]{} images have been loaded, and there are {} positive images and {} negative images'.format(len(self.data_list), len(pos_paths), len(neg_paths)))

        self.transforms = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor()
        ])
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, index):
        image_path, label = self.data_list[index]
        image = Image.open(image_path).convert('RGB')
        return self.transforms(image), label

class InjectionDatasetREAD(Dataset):
    def __init__(self, root, DFtype, size=256):
        self.image_paths = sorted([x.path for x in os.scandir(root) if x.name.endswith(".jpg") or x.name.endswith(".png")])
        self.image_df_paths = [i.replace('WM',DFtype) for i in self.image_paths]
        # logging.info('[*]{} images have been loaded'.format(len(self.image_paths)))
        print('[*]{} images have been loaded'.format(len(self.image_paths)))
        print('[*]{} DF images have been loaded'.format(len(self.image_df_paths)))
        self.transforms = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor()
        ])
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        image_df_path = self.image_df_paths[index]
        image_df = Image.open(image_df_path).convert('RGB')
        return self.transforms(image), self.transforms(image_df)