import os

from .utils import Datum, DatasetBase, read_json, write_json, build_data_loader
from .oxford_pets import OxfordPets


template = ['a centered satellite photo of {}.']


NEW_CNAMES = {
    'AnnualCrop': 'Annual Crop Land',
    'Forest': 'Forest',
    'HerbaceousVegetation': 'Herbaceous Vegetation Land',
    'Highway': 'Highway or Road',
    'Industrial': 'Industrial Buildings',
    'Pasture': 'Pasture Land',
    'PermanentCrop': 'Permanent Crop Land',
    'Residential': 'Residential Buildings',
    'River': 'River',
    'SeaLake': 'Sea or Lake'
}


class EuroSAT(DatasetBase):

    dataset_dir = 'eurosat'

    def __init__(self, root, num_shots, is_syn):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, '2750')
        self.split_path = os.path.join(self.dataset_dir, 'split_zhou_EuroSAT.json')
        self.template = template
        self.is_syn = is_syn
        self.NEW_CNAMES = NEW_CNAMES
        self.inv_cmap = { v: k for k, v in self.NEW_CNAMES.items()}   

        train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)



        # # add the syn data path (classlabels as prompt, 200 images per class)
        # self.syn_dir = os.path.join(self.dataset_dir, 'Syn')
        # self.syn_split_path = os.path.join(self.dataset_dir, 'Syn_EuroSAT.json')


        # # add the syn data path (with captions, 10 images per class)
        # self.syn_dir = os.path.join(self.dataset_dir, 'Syn_10_per_class')
        # self.syn_split_path = os.path.join(self.dataset_dir, 'Syn_10_per_class_EuroSAT.json')

        # # add the syn data path (classlabels as prompt, 10 images per class)
        # self.syn_dir = os.path.join(self.dataset_dir, 'Syn_10_classname')
        # self.syn_split_path = os.path.join(self.dataset_dir, 'Syn_10_classname_EuroSAT.json')

        # # add the syn data path (captions as prompt img2img, 10 images per class)
        # self.syn_dir = os.path.join(self.dataset_dir, 'Syn_10_img2img_caption')
        # self.syn_split_path = os.path.join(self.dataset_dir, 'Syn_10_img2img_caption_EuroSAT.json')

        # # add the syn data path (captions as prompt img2img contrast, 25 images per class)
        # self.syn_dir = os.path.join(self.dataset_dir, 'Syn_10_img2img_caption_contrast')
        # self.syn_split_path = os.path.join(self.dataset_dir, 'Syn_10_img2img_caption_contrast_EuroSAT.json')

        if self.is_syn:
            # add the syn data path (captions as prompt img2img contrast, 25 images per class)
            self.syn_dir = os.path.join(self.dataset_dir, f'Syn_img2img_caption_contrast_{num_shots}')
            self.syn_split_path = os.path.join(self.dataset_dir, f'Syn_img2img_caption_contrast_{num_shots}_eurosat.json')

            # # add the syn data path (captions as prompt img2img contrast, 25 images per class)
            # self.syn_dir = os.path.join(self.dataset_dir, 'Syn_img2img_caption_contrast_2')
            # self.syn_split_path = os.path.join(self.dataset_dir, 'Syn_img2img_caption_contrast_2_eurosat.json')


            # add syn data (captions as prompt, img2img 10 images per class)
            syn_split = read_json(self.dataset_dir + '/' + f'Syn_img2img_caption_contrast_{num_shots}_eurosat.json')
            train_syn = self._convert(syn_split['train'], '/cis/home/aroy/code/Tip-Adapter/DATA/eurosat/')

            # # add syn data (captions as prompt, img2img 10 images per class)
            # syn_split = read_json(self.dataset_dir + '/Syn_img2img_caption_contrast_2_eurosat.json')
            # train_syn = self._convert(syn_split['train'], '/cis/home/aroy/code/Tip-Adapter/DATA/eurosat/')
            # #breakpoint()
        else:
            train_syn = None 
        # # add syn data (captions as prompt, img2img 10 images per class)
        # syn_split = read_json(self.dataset_dir + '/Syn_10_img2img_caption_contrast_EuroSAT.json')
        # train_syn = self._convert(syn_split['train'], '/cis/home/aroy/code/Tip-Adapter/DATA/eurosat/')
     

        #breakpoint()
        # # add syn data (classlabels as prompt, 200 images per class)
        # syn_split = read_json(self.dataset_dir + '/Syn_EuroSAT.json')

        # add syn data (with captions, 10 images per class)
        #syn_split = read_json(self.dataset_dir + '/Syn_10_per_class_EuroSAT.json')

        # # add syn data (classlabels as prompt, 10 images per class)
        # syn_split = read_json(self.dataset_dir + '/Syn_10_classname_EuroSAT.json')


        #train_syn = OxfordPets.read_split(self.syn_split_path, self.syn_dir)
        
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

    def update_classname(self, dataset_old):
        dataset_new = []
        for item_old in dataset_old:
            cname_old = item_old.classname
            cname_new = NEW_CLASSNAMES[cname_old]
            item_new = Datum(
                impath=item_old.impath,
                label=item_old.label,
                classname=cname_new
            )
            dataset_new.append(item_new)
        return dataset_new
