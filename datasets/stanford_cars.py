import os
from scipy.io import loadmat

from .oxford_pets import OxfordPets
from .utils import Datum, DatasetBase, read_json


template = ['a photo of a {}.']


class StanfordCars(DatasetBase):

    dataset_dir = 'stanford_cars'

    def __init__(self, root, num_shots, is_syn):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.split_path = os.path.join(self.dataset_dir, 'split_zhou_StanfordCars.json')
        self.is_syn = is_syn

        self.template = template

        train, val, test = OxfordPets.read_split(self.split_path, self.dataset_dir)
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)

        if self.is_syn:
            # add the syn data path (captions as prompt img2img contrast, 25 images per class)
            self.syn_dir = os.path.join(self.dataset_dir, f'Syn_img2img_caption_contrast_{num_shots}')
            self.syn_split_path = os.path.join(self.dataset_dir, f'Syn_img2img_caption_contrast_{num_shots}_stanford_cars.json')

            # add syn data (captions as prompt, img2img 10 images per class)
            syn_split = read_json(self.dataset_dir + '/' + f'Syn_img2img_caption_contrast_{num_shots}_stanford_cars.json')
            train_syn = self._convert(syn_split['train'], '/cis/home/aroy/code/Tip-Adapter/DATA/stanford_cars/')

        else:
            train_syn = None 

        super().__init__(train_x=train, val=val, test=test, train_u=train_syn)
    
    def read_data(self, image_dir, anno_file, meta_file):
        anno_file = loadmat(anno_file)['annotations'][0]
        meta_file = loadmat(meta_file)['class_names'][0]
        items = []

        for i in range(len(anno_file)):
            imname = anno_file[i]['fname'][0]
            impath = os.path.join(self.dataset_dir, image_dir, imname)
            label = anno_file[i]['class'][0, 0]
            label = int(label) - 1 # convert to 0-based index
            classname = meta_file[label][0]
            names = classname.split(' ')
            year = names.pop(-1)
            names.insert(0, year)
            classname = ' '.join(names)
            item = Datum(
                impath=impath,
                label=label,
                classname=classname
            )
            items.append(item)
        
        return items

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
