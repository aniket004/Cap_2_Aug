import os
import random
from scipy.io import loadmat
from collections import defaultdict

from .oxford_pets import OxfordPets
from .utils import Datum, DatasetBase, read_json


template = ['a photo of a {}, a type of flower.']


class OxfordFlowers(DatasetBase):

    dataset_dir = 'oxford_flowers'

    def __init__(self, root, num_shots, is_syn):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'jpg')
        self.label_file = os.path.join(self.dataset_dir, 'imagelabels.mat')
        self.lab2cname_file = os.path.join(self.dataset_dir, 'cat_to_name.json')
        self.split_path = os.path.join(self.dataset_dir, 'split_zhou_OxfordFlowers.json')
        self.is_syn = is_syn

        self.template = template

        train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)


        if self.is_syn:
            # add the syn data path (captions as prompt img2img contrast, 25 images per class)
            self.syn_dir = os.path.join(self.dataset_dir, f'Syn_img2img_caption_contrast_{num_shots}')
            self.syn_split_path = os.path.join(self.dataset_dir, f'Syn_img2img_caption_contrast_{num_shots}_oxford_flowers.json')

            # add syn data (captions as prompt, img2img 10 images per class)
            syn_split = read_json(self.dataset_dir + '/' + f'Syn_img2img_caption_contrast_{num_shots}_oxford_flowers.json')
            train_syn = self._convert(syn_split['train'], '/cis/home/aroy/code/Tip-Adapter/DATA/oxford_flowers/')

        else:
            train_syn = None 
        
        super().__init__(train_x=train, val=val, test=test, train_u=train_syn)
    
    def read_data(self):
        tracker = defaultdict(list)
        label_file = loadmat(self.label_file)['labels'][0]
        for i, label in enumerate(label_file):
            imname = f'image_{str(i + 1).zfill(5)}.jpg'
            impath = os.path.join(self.image_dir, imname)
            label = int(label)
            tracker[label].append(impath)
        
        print('Splitting data into 50% train, 20% val, and 30% test')

        def _collate(ims, y, c):
            items = []
            for im in ims:
                item = Datum(
                    impath=im,
                    label=y-1, # convert to 0-based label
                    classname=c
                )
                items.append(item)
            return items

        lab2cname = read_json(self.lab2cname_file)
        train, val, test = [], [], []
        for label, impaths in tracker.items():
            random.shuffle(impaths)
            n_total = len(impaths)
            n_train = round(n_total * 0.5)
            n_val = round(n_total * 0.2)
            n_test = n_total - n_train - n_val
            assert n_train > 0 and n_val > 0 and n_test > 0
            cname = lab2cname[str(label)]
            train.extend(_collate(impaths[:n_train], label, cname))
            val.extend(_collate(impaths[n_train:n_train+n_val], label, cname))
            test.extend(_collate(impaths[n_train+n_val:], label, cname))
        
        return train, val, test

    def _convert(self, items, path_prefix):
        out = []
        #breakpoint()
        for impath, label, classname in items:
            impath = os.path.join(path_prefix, impath)
            item = Datum(
                impath=impath,
                label=int(label),
                classname=classname
            )
            out.append(item)
        return out