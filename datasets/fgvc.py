import os

from .utils import Datum, DatasetBase, read_json, write_json, build_data_loader


template = ['a photo of a {}, a type of aircraft.']


class FGVCAircraft(DatasetBase):

    dataset_dir = 'fgvc_aircraft'

    def __init__(self, root, num_shots, is_syn):
        
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'images')
        self.is_syn = is_syn

        self.template = template

        classnames = []
        with open(os.path.join(self.dataset_dir, 'variants.txt'), 'r') as f:
            lines = f.readlines()
            for line in lines:
                classnames.append(line.strip())
        cname2lab = {c: i for i, c in enumerate(classnames)}

        train = self.read_data(cname2lab, 'images_variant_train.txt')
        val = self.read_data(cname2lab, 'images_variant_val.txt')
        test = self.read_data(cname2lab, 'images_variant_test.txt')
        
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)

        if self.is_syn:
            # add the syn data path (captions as prompt img2img contrast, 25 images per class)
            self.syn_dir = os.path.join(self.dataset_dir, f'Syn_img2img_caption_contrast_{num_shots}')
            self.syn_split_path = os.path.join(self.dataset_dir, f'Syn_img2img_caption_contrast_{num_shots}_fgvc.json')

            # add syn data (captions as prompt, img2img 10 images per class)
            syn_split = read_json(self.dataset_dir + '/' + f'Syn_img2img_caption_contrast_{num_shots}_fgvc.json')
            train_syn = self._convert(syn_split['train'], '/cis/home/aroy/code/Tip-Adapter/DATA/fgvc/')

        else:
            train_syn = None 


        
        super().__init__(train_x=train, val=val, test=test, train_u=train_syn)
    
    def read_data(self, cname2lab, split_file):
        filepath = os.path.join(self.dataset_dir, split_file)
        items = []
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(' ')
                imname = line[0] + '.jpg'
                classname = ' '.join(line[1:])
                impath = os.path.join(self.image_dir, imname)
                label = cname2lab[classname]
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