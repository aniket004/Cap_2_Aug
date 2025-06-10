import os

from .utils import Datum, DatasetBase, read_json
from .oxford_pets import OxfordPets


template = ['a photo of a {}.']


class Caltech101(DatasetBase):

    dataset_dir = 'caltech-101'

    def __init__(self, root, num_shots, is_syn):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, '101_ObjectCategories')
        self.split_path = os.path.join(self.dataset_dir, 'split_zhou_Caltech101.json')
        self.is_syn = is_syn
        self.template = template

        train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)

        if self.is_syn:

            # # add the syn data path (captions as prompt img2img contrast, 25 images per class)
            # self.syn_dir = os.path.join(self.dataset_dir, 'Syn_10_img2img_caption_contrast')
            # self.syn_split_path = os.path.join(self.dataset_dir, 'Syn_10_img2img_caption_contrast_Caltech-101.json')

            # add the syn data path (captions as prompt img2img contrast, 25 images per class)
            self.syn_dir = os.path.join(self.dataset_dir, f'Syn_img2img_caption_contrast_{num_shots}')
            self.syn_split_path = os.path.join(self.dataset_dir, f'Syn_img2img_caption_contrast_{num_shots}_caltech101.json')


            # add syn data (captions as prompt, img2img 10 images per class)
            syn_split = read_json(self.dataset_dir + '/' + f'Syn_img2img_caption_contrast_{num_shots}_caltech101.json')
            train_syn = self._convert(syn_split['train'], '/cis/home/aroy/code/Tip-Adapter/DATA/caltech-101/')
        else:
            train_syn = None


        #super().__init__(train_x=train, val=val, test=test)
        super().__init__(train_x=train, val=val, test=test, train_u=train_syn)

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