import os
import random
import argparse
import yaml
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn

from datasets.imagenet import ImageNet
import clip
from utils import *

from pathlib import Path
import requests
import numpy as np
from PIL import Image
from io import BytesIO
import json
from diffusers import StableDiffusionImg2ImgPipeline
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import BlipModel, AutoProcessor
import wandb
import warnings
warnings.filterwarnings('ignore')
from tllib.modules.kernels import GaussianKernel
from tllib.alignment.dan import MultipleKernelMaximumMeanDiscrepancy
from tsne_helper import generate_tsne
import torchvision.transforms as transforms
import torchvision


def get_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', help='settings of Tip-Adapter in yaml format')
    parser.add_argument('--shots', type=int, default=2)
    #parser.add_argument('--backbone', type=str, default='RN50')
    parser.add_argument('--is_baseline', type=str, default="True")
    parser.add_argument('--strength', type=float, default=0.5)
    parser.add_argument('--train_epoch', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    #parser.add_argument('--init_alpha', type=float, default=3.0)
    #parser.add_argument('--init_beta', type=float, default=1.0)
    parser.add_argument('--cache_syn', type=str, default="False")
    parser.add_argument('--use_mmd', type=bool, default=True)
    #parser.add_argument('--use_mmd', action='store_true')
    parser.add_argument('--mmd_coeff', type=float, default=1.0)
    parser.add_argument('--train_batchsize', type=int, default=256)
    parser.add_argument('--tsne_viz', type=str, default="False")
    parser.add_argument('--mode', type=str, default="baseline")
    #parser.set_defaults(cache_syn=False)
    #parser.set_defaults(use_mmd=True)

    args = parser.parse_args()

    return args


def generate_json(cfg, dataset, syn_data_folder, dataset_folder, dataset_classname):
    '''
    Generate json for the generated dataset for training
    '''

    num_shots = cfg['shots']

    # get the synthetic images folder
    path = syn_data_folder

    all_class_images = []

    for i in range(len(dataset_classname)):
        list_dir = os.listdir(path+dataset_classname[i]+'/')
        for j in range(len(list_dir)):
            all_class_images.append( [ syn_data_folder.split('/')[-2] + '/' + dataset_classname[i] + '/' + list_dir[j], f'{i}' , dataset_classname[i] ] )


    # create a dict with keys 'train' for the syn images to match the format
    syn_split = {}
    syn_split['train'] = all_class_images

    json_data = json.dumps(syn_split)
    #jsonfile = open("/cis/home/aroy/code/Tip-Adapter/DATA/eurosat/Syn_10_img2img_caption_contrast_EuroSAT.json", "w")
    jsonfile = open(f'/cis/home/aroy/code/Tip-Adapter/DATA/{dataset_folder}/Syn_img2img_caption_contrast_{num_shots}_{dataset}.json', "w")
    jsonfile.write(json_data)
    jsonfile.close()


def get_caption(cfg, train_loader_F, template):
    '''
    Generate images with classlabels as captions
    '''

    num_shots = cfg['shots']
    ####### load large caption model ###############################
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", torch_dtype=torch.float32).to("cuda")

    #################################################################
    ######## pass the input image through the caption model #########
    # get the captions for all the images

    caption_all_class = []
    image_text_sim = []

    # get the captions for all the selected images
    index_list = []
    caption_list = []
    caption_target = []
    target_all = []

    # generate syn images 
    for ii, (images, target) in enumerate(tqdm(train_loader_F)):
        images, target = images.cuda(), target.cuda()

        # renormalize images with mean and std
        # renormalize the images 
        mean_img = torch.Tensor([0.48145466, 0.4578275, 0.40821073]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(images.shape[0], 1, images.shape[2], images.shape[3])
        std_img = torch.Tensor([0.26862954, 0.26130258, 0.27577711]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(images.shape[0], 1, images.shape[2], images.shape[3])
    
        # converted_img are re-normalized 
        conv_images = images.cpu()*std_img + mean_img        

        interval = len(conv_images)//10
        
        for j in range(interval):
            # pass bacthes of images to the model
            inputs = processor(conv_images[j*10:(j+1)*10], return_tensors="pt").to("cuda", torch.float32)
            out = model.generate(**inputs)
            # get the caption
            temp = 10*[ template[0].replace('of a {}.', '') + 'and ']
            cap = processor.batch_decode(out, skip_special_tokens=True)
            caption = [ x+y for x,y in zip(temp, cap) ]

            caption_list.append(caption)
            #index_list.append(idx[j*10:(j+1)*10])
            target_all.append(target[j*10:(j+1)*10])

    caption_lists = [item for sublist in caption_list for item in sublist]
    #index_lists = [item for sublist in index_list for item in sublist]
    target_alls = [item for sublist in target_all for item in sublist]

    #breakpoint()

    unique_target = torch.unique(torch.tensor(target_alls))

    #index_lists = torch.stack(index_lists)
    target_alls = torch.stack(target_alls)

    captions_by_class = []
    for k in range( len(unique_target) ):
        cc = [caption_lists[x] for x in torch.where(target_alls == unique_target[k])[0]]
        captions_by_class.append(cc)

    # handling corner cases
    # NOTE: captions_by_class: [Num_class x Num_shots]
    # if missing per class captions, repeat it with copying any from the rest of the class

    #breakpoint()
    
    for k in range( len(unique_target) ):
        if len(captions_by_class[k]) != num_shots:
            captions_by_class[k] += (num_shots - len(captions_by_class[k]))*[captions_by_class[k][0] ] 
            #captions_by_class[k].extend( [captions_by_class[k][0] ] )

    #breakpoint()
    return captions_by_class


def id_to_foldername( target, syn_dataset_dict ):

    foldername = []
    for i in range(len(target)):
        foldername.append(syn_dataset_dict['classes'][ target[i] ] )

    return foldername


def foldername_to_index( foldername, imagenet_dict ):

    ###breakpoint()
    # get the reverse of imagenet.__dict__['classnames']
    inv_classnames = { v: k for k, v in enumerate(imagenet_dict['classnames']) }
    #inv_classnames = { v: k for k, v in imagenet_dict['classnames'] }

    #breakpoint()

    index = []
    for i  in range(len(foldername)):
        index.append( inv_classnames[ foldername[i] ])
    
    return index


def converted_target(target, syn_dataset_dict, imagenet_dict):

    foldername = id_to_foldername(target, syn_dataset_dict)
    #breakpoint()
    converted_target = foldername_to_index(foldername, imagenet_dict)
    #breakpoint()
    return converted_target


def run_tip_adapter_F_syn(cfg, args, cache_keys, cache_values, test_features,\
     test_labels, clip_weights, clip_model, train_loader_F, classnames, \
     template, imagenet_dict):
    

    # get the captions
    captions_by_class = get_caption(cfg, train_loader_F=train_loader_F, template=template)


    # get the diffusion model generated images 
    get_syn_images(cfg, train_loader_F, template, captions_by_class, classnames, inv_cmap=None)

    train_preprocess = transforms.Compose([
                                            transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
                                            transforms.RandomHorizontalFlip(p=0.5),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
                                        ])
    #test_preprocess = preprocess

    num_shots = cfg['shots']
    #syn_root = f'/cis/home/aroy/code/Tip-Adapter/DATA/imagenet/new_datasets/Syn_img2img_caption_contrast_{num_shots}/train/'
    syn_root = f'/cis/home/aroy/code/Tip-Adapter/DATA/imagenet/Syn_img2img_caption_contrast_{num_shots}/'

    syn_dataset = torchvision.datasets.ImageFolder(root=syn_root, transform= train_preprocess)

    syn_dataset_dict = syn_dataset.__dict__

    # # check the syn_dataloader 
    #syn_loader = build_data_loader(data_source=syn_dataset.train_u, batch_size=cfg['train_batchsize'], tfm=train_tranform, is_train=True, shuffle=True)
    syn_loader = torch.utils.data.DataLoader(syn_dataset, batch_size=cfg['train_batchsize'], num_workers=8, shuffle=True)

    # get the syn loader to generate cache model
    #syn_loader_cache = build_data_loader(data_source=syn_dataset.train_u, batch_size=cfg['train_batchsize'], tfm=train_tranform, is_train=True, shuffle=False)
    syn_loader_cache = torch.utils.data.DataLoader(syn_dataset, batch_size=cfg['train_batchsize'], num_workers=8, shuffle=False)



    # Construct the cache model by few-shot training set
    print("\nConstructing cache model for syn images by few-shot visual features and labels.")
    syn_cache_keys, syn_cache_values = build_cache_model_imagenet(cfg, clip_model, syn_loader_cache)

    # combine the cache_keys and syn_cache_keys
    # cache_keys [1024, n_shot*n_class] , syn_cache_keys [1024, n_syn_sample*n_class]
    # cache_values [400, 100], syn_cache_values [1600, 100]

    if args.cache_syn:
        cache_keys = torch.cat( (cache_keys, syn_cache_keys), dim=1)
        cache_values = torch.cat(  (cache_values, syn_cache_values), dim=0 )
    else:
        cache_keys = cache_keys
        cache_values = cache_values

    # cache_keys = torch.cat( (cache_keys, syn_cache_keys), dim=1)
    # cache_values = torch.cat(  (cache_values, syn_cache_values), dim=0 )

    #breakpoint()
    # Enable the cached keys to be learnable
    adapter = nn.Linear(cache_keys.shape[0], cache_keys.shape[1], bias=False).to(clip_model.dtype).cuda()
    adapter.weight = nn.Parameter(cache_keys.t())
    
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=cfg['lr'], eps=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['train_epoch'] * len(train_loader_F))
    
    beta, alpha = cfg['init_beta'], cfg['init_alpha']
    best_acc, best_epoch = 0.0, 0


    seen = 0
    iterator = iter(syn_loader)
    #syn_batch_size=1024

    #breakpoint()
    for train_idx in range(cfg['train_epoch']):
        # Train
        adapter.train()
        correct_samples, all_samples = 0, 0
        loss_list = []
        print('Train Epoch: {:} / {:}'.format(train_idx, cfg['train_epoch']))

        for i, (real_images, real_target) in enumerate(tqdm(train_loader_F)):
            real_images, real_target = real_images.cuda(), real_target.cuda()
            
            if seen > len(syn_loader):
                iterator = iter(syn_loader)
                seen = 0

            #print('no of iterate', i)
            #breakpoint()
            syn_img, syn_target = next(iterator) #20000xD, len(20000)

            #breakpoint()

            syn_target = torch.tensor( converted_target(syn_target, syn_dataset_dict, imagenet_dict) )
            seen += syn_loader.batch_size

            #breakpoint()
            # renormalize the images 
            mean_img = torch.Tensor([0.48145466, 0.4578275, 0.40821073]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(real_images.shape[0], 1, real_images.shape[2], real_images.shape[3])
            std_img = torch.Tensor([0.26862954, 0.26130258, 0.27577711]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(real_images.shape[0], 1, real_images.shape[2], real_images.shape[3])
        
            # normalize syn_img
            mean_syn = torch.Tensor([0.48145466, 0.4578275, 0.40821073]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(syn_img.shape[0], 1, syn_img.shape[2], syn_img.shape[3])
            std_syn = torch.Tensor([0.26862954, 0.26130258, 0.27577711]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(syn_img.shape[0], 1, syn_img.shape[2], syn_img.shape[3])
        
            #breakpoint()
            # converted_img are re-normalized 
            conv_images = real_images.cpu()*std_img+mean_img
            conv_syn_img = syn_img.cpu()*std_syn+mean_syn

            # # get the synthetic images and captions for the images
            # syn_img, syn_target = get_syn_images(conv_images, target)

            # # # load syn images
            # # for j, (syn_img, syn_target) in enumerate(tqdm(syn_loader)):
            # syn_img, syn_target = syn_img.cuda(), syn_target.cuda()

            #breakpoint()
            images = torch.cat((real_images, syn_img.cuda()), dim=0)
            target = torch.cat((real_target, syn_target.cuda()), dim=0)

            with torch.no_grad():
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)

            #breakpoint()
            affinity = adapter(image_features)
            cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
            clip_logits = 100. * image_features @ clip_weights
            tip_logits = clip_logits + cache_logits * alpha


            kernels = (GaussianKernel(alpha=0.5), GaussianKernel(alpha=1.), GaussianKernel(alpha=2.))
            mmd_criterion = MultipleKernelMaximumMeanDiscrepancy(kernels).cuda()

            #breakpoint()
            if args.use_mmd:
                #breakpoint()
                loss = F.cross_entropy(tip_logits, target, label_smoothing=0.2) + \
                    args.mmd_coeff * mmd_criterion(affinity[:real_images.shape[0]], affinity[-real_images.shape[0]:])     
            else:
                loss = F.cross_entropy(tip_logits, target)

            # # get the tsne plots

            # for perp in [50]:
            #     # generate_tsne(combined_features,combined_labels,perplexity=perp,idx=idx,experiment_name='5000_1000_1000',add_text=query_accuracy)
            #     generate_tsne(affinity.cpu().detach().numpy(), target.cpu(), perplexity=perp, idx=i,
            #                     experiment_name='TSNE',
            #                     episode_number=train_idx)

            acc = cls_acc(tip_logits, target)
            correct_samples += acc / 100 * len(tip_logits)
            all_samples += len(tip_logits)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # # get the tsne plots

            # for perp in [50]:
            #     # generate_tsne(combined_features,combined_labels,perplexity=perp,idx=idx,experiment_name='5000_1000_1000',add_text=query_accuracy)
            #     generate_tsne(affinity.cpu().detach().numpy(), target.cpu(), perplexity=perp, idx=i,
            #                     experiment_name='TSNE',
            #                     episode_number=train_idx)


        current_lr = scheduler.get_last_lr()[0]
        print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples, correct_samples, all_samples, sum(loss_list)/len(loss_list)))

        # Eval
        adapter.eval()

        affinity = adapter(test_features)
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
        clip_logits = 100. * test_features @ clip_weights
        tip_logits = clip_logits + cache_logits * alpha
        acc = cls_acc(tip_logits, test_labels)

        #breakpoint()
                # get the tsne plots

        if args.tsne_viz: 
            for perp in [10, 20, 30, 40, 50, 50, 70, 80, 90, 100]:
                # generate_tsne(combined_features,combined_labels,perplexity=perp,idx=idx,experiment_name='5000_1000_1000',add_text=query_accuracy)
                generate_tsne(affinity.cpu().detach().numpy(), test_labels.cpu(), pca_comps=50, perplexity=perp, idx=train_idx,
                                experiment_name='TSNE',
                                episode_number=train_idx)

        print("**** Syn Tip-Adapter-F's test accuracy: {:.2f}. ****\n".format(acc))
        if acc > best_acc:
            best_acc = acc
            best_epoch = train_idx
            torch.save(adapter.weight, cfg['cache_dir'] + "/best_F_" + str(cfg['shots']) + "shots.pt")
    
    adapter.weight = torch.load(cfg['cache_dir'] + "/best_F_" + str(cfg['shots']) + "shots.pt")
    print(f"**** After fine-tuning with Syn images, Tip-Adapter-F's best test accuracy: {best_acc:.2f}, at epoch: {best_epoch}. ****\n")

    print("\n-------- Searching hyperparameters on the val set. --------")

    # Search Hyperparameters
    best_beta, best_alpha = search_hp(cfg, cache_keys, cache_values, test_features, test_labels, clip_weights, adapter=adapter)

    print("\n-------- Evaluating on the test set. --------")
   
    affinity = adapter(test_features)
    cache_logits = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_values
    
    tip_logits = clip_logits + cache_logits * best_alpha
    acc = cls_acc(tip_logits, test_labels)
    print("**** Syn Tip-Adapter-F's test accuracy: {:.2f}. ****\n".format(max(best_acc, acc)))

    wandb.log({
            'test_acc': max(best_acc, acc),
            'dataset': cfg['dataset'],
            'shots': cfg['shots'],
            'lr': cfg['lr'],
            'init_alpha': cfg['init_alpha'],
            'init_beta': cfg['init_beta'],
            'backbone': cfg['backbone'],
            'strength': cfg['strength'],
            'use_mmd': args.use_mmd,
            'mmd_coeff': args.mmd_coeff,
            'train_batchsize': args.train_batchsize,
            'train_epoch': args.train_epoch,
            'cache_syn': args.cache_syn
    })


def get_syn_images(cfg, train_loader_F, template, captions_by_class, classnames, inv_cmap):
    '''
    Generate images from captions by class
    '''
    num_shots = cfg['shots']
    strength = cfg['strength']

    #breakpoint()
    # returns the classname corresponding to the dataset
    # eurosat has inv_cmap corresponding to classnames
    if cfg['dataset'] == 'eurosat':
        dataset_classname = []
        for k in range(len(classnames)):
            dataset_classname.append( inv_cmap[classnames[k]])
    else:
        dataset_classname = classnames


    if cfg['dataset'] == 'caltech101':
        dataset_folder = 'caltech-101'
    elif cfg['dataset'] == 'food101':
        dataset_folder = 'food-101'
    else:
        dataset_folder = cfg['dataset']


    # get the captions for all the selected images
    index_list = []
    caption_list = []
    caption_target = []
    target_all = []


    dataset = cfg['dataset'] 
    ######## save the generated images in a folder #######################
    # #dir_path = f'/cis/home/aroy/code/Tip-Adapter/DATA/eurosat/Syn_img2img_caption_contrast_{num_shots}/' 
    # #dir_path = f'/cis/home/aroy/code/Tip-Adapter/DATA/{dataset}/Syn_img2img_caption_contrast_{num_shots}/' 
    # dir_path = f'/cis/home/aroy/code/Tip-Adapter/DATA/{dataset_folder}/Syn_img2img_caption_contrast_{num_shots}/' 
    # #dir_path = f'/cis/home/aroy/code/Tip-Adapter/DATA/{dataset_folder}/Syn_img2img_caption_contrast_{num_shots}_{strength}/' 

    dir_path = f'/cis/home/aroy/code/Tip-Adapter/DATA/{dataset_folder}/Syn_img2img_caption_contrast_{num_shots}/' 


    # create directory path
    if not os.path.exists(dir_path):

        ######## load diffusion model #######################################
        device = "cuda"
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to(
            device
        )
        ######################################################################
            # disable nsfw filter
        def dummy(images, **kwargs):

            return images, False

        pipe.safety_checker = dummy

        Path(dir_path).mkdir(parents=True, exist_ok=True)

        for i in range(len(dataset_classname)):
            os.makedirs(os.path.join(dir_path,dataset_classname[i]), exist_ok=True)

    
        flag = 0
        # generate syn images 
        for ii, (images, target) in enumerate(tqdm(train_loader_F)):
            images, target = images.cuda(), target.cuda()

            # renormalize images with mean and std
            # renormalize the images 
            mean_img = torch.Tensor([0.48145466, 0.4578275, 0.40821073]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(images.shape[0], 1, images.shape[2], images.shape[3])
            std_img = torch.Tensor([0.26862954, 0.26130258, 0.27577711]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(images.shape[0], 1, images.shape[2], images.shape[3])
        
            # converted_img are re-normalized 
            conv_images = images.cpu()*std_img + mean_img        

            interval = len(conv_images)//10
            
            #breakpoint()
            for j in range(len(conv_images)):
                class_label = target[j]
                captions = captions_by_class[class_label][0:min(num_shots, 5)]
                gen_image = pipe(prompt=captions, image=transforms.ToPILImage()\
                            (conv_images[j]).convert("RGB").resize((512,512)), \
                            strength=strength, guidance_scale=7.5).images
                            #strength=0.5, guidance_scale=7.5).images
                for k in range(min(num_shots, 5)):
                    gen_image[k].save(os.path.join(dir_path, dataset_classname[class_label], str(flag)+'.png'))
                    flag += 1


 
def run_tip_adapter(cfg, cache_keys, cache_values, test_features, test_labels, clip_weights):
    
    # Zero-shot CLIP
    clip_logits = 100. * test_features @ clip_weights
    acc = cls_acc(clip_logits, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(acc))

    # Tip-Adapter
    beta, alpha = cfg['init_beta'], cfg['init_alpha']
    
    affinity = test_features @ cache_keys
    cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
    
    tip_logits = clip_logits + cache_logits * alpha
    acc = cls_acc(tip_logits, test_labels)
    print("**** Tip-Adapter's test accuracy: {:.2f}. ****\n".format(acc))

    # Search Hyperparameters
    _ = search_hp(cfg, cache_keys, cache_values, test_features, test_labels, clip_weights)


def run_tip_adapter_F(cfg, cache_keys, cache_values, test_features, test_labels, clip_weights, clip_model, train_loader_F):
    
    # Enable the cached keys to be learnable
    adapter = nn.Linear(cache_keys.shape[0], cache_keys.shape[1], bias=False).to(clip_model.dtype).cuda()
    adapter.weight = nn.Parameter(cache_keys.t())
    
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=cfg['lr'], eps=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['train_epoch'] * len(train_loader_F))
    
    beta, alpha = cfg['init_beta'], cfg['init_alpha']
    best_acc, best_epoch = 0.0, 0

    for train_idx in range(cfg['train_epoch']):
        # Train
        adapter.train()
        correct_samples, all_samples = 0, 0
        loss_list = []
        print('Train Epoch: {:} / {:}'.format(train_idx, cfg['train_epoch']))

        for i, (images, target) in enumerate(tqdm(train_loader_F)):
            images, target = images.cuda(), target.cuda()
            with torch.no_grad():
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)

            affinity = adapter(image_features)
            cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
            clip_logits = 100. * image_features @ clip_weights
            tip_logits = clip_logits + cache_logits * alpha

            loss = F.cross_entropy(tip_logits, target)

            acc = cls_acc(tip_logits, target)
            correct_samples += acc / 100 * len(tip_logits)
            all_samples += len(tip_logits)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples, correct_samples, all_samples, sum(loss_list)/len(loss_list)))

        # Eval
        adapter.eval()

        affinity = adapter(test_features)
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
        clip_logits = 100. * test_features @ clip_weights
        tip_logits = clip_logits + cache_logits * alpha
        acc = cls_acc(tip_logits, test_labels)

        print("**** Tip-Adapter-F's test accuracy: {:.2f}. ****\n".format(acc))
        if acc > best_acc:
            best_acc = acc
            best_epoch = train_idx
            torch.save(adapter.weight, cfg['cache_dir'] + "/best_F_" + str(cfg['shots']) + "shots.pt")
    
    adapter.weight = torch.load(cfg['cache_dir'] + "/best_F_" + str(cfg['shots']) + "shots.pt")
    print(f"**** After fine-tuning, Tip-Adapter-F's best test accuracy: {best_acc:.2f}, at epoch: {best_epoch}. ****\n")

    # Search Hyperparameters
    _ = search_hp(cfg, affinity, cache_values, test_features, test_labels, clip_weights, adapter=adapter)


def main():

    # Load config file
    args = get_arguments()

    args.cache_syn = True if args.cache_syn == 'True' else False
    args.tsne_viz = True if args.tsne_viz == 'True' else False
    args.is_baseline = True if args.is_baseline == 'True' else False

    assert (os.path.exists(args.config))
    
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    cache_dir = os.path.join('./caches', cfg['dataset'])
    os.makedirs(cache_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir

    # get the params by argument
    cfg['shots'] = args.shots
    cfg['strength'] = args.strength
    cfg['train_epoch'] = args.train_epoch
    cfg['lr'] = args.lr
    #cfg['init_alpha'] = args.init_alpha 
    #cfg['init_beta'] = args.init_beta
    cfg['cache_syn'] = args.cache_syn
    cfg['use_mmd'] = args.use_mmd
    cfg['mmd_coeff'] = args.mmd_coeff
    cfg['train_batchsize'] = args.train_batchsize
    cfg['is_baseline'] = args.is_baseline

    print("\nRunning configs.")
    print(cfg, "\n")
    print(args, "\n")

    #breakpoint()
    wandb.init(project="caption_diffusion", entity="aniketroy", config=cfg)

    # CLIP
    clip_model, preprocess = clip.load(cfg['backbone'])
    clip_model.eval()

    # ImageNet dataset
    random.seed(1)
    torch.manual_seed(1)
    
    #breakpoint()
    print("Preparing ImageNet dataset.")
    imagenet = ImageNet(cfg['root_path'], cfg['shots'], preprocess, is_syn=False)

    test_loader = torch.utils.data.DataLoader(imagenet.test, batch_size=64, num_workers=8, shuffle=False)

    train_loader_cache = torch.utils.data.DataLoader(imagenet.train, batch_size=cfg['train_batchsize'], num_workers=8, shuffle=False)
    train_loader_F = torch.utils.data.DataLoader(imagenet.train, batch_size=cfg['train_batchsize'], num_workers=8, shuffle=True)


    # get the imagenet dict
    imagenet_dict = imagenet.__dict__
    # imagenet_dict have keys dict_keys(['dataset_dir', 'image_dir', 'is_syn', 'train', 'val', 'test', 'template', 'classnames'])

    #breakpoint()
    # Textual features
    print("Getting textual features as CLIP's classifier.")
    clip_weights = clip_classifier(imagenet.classnames, imagenet.template, clip_model)

    # Construct the cache model by few-shot training set
    print("\nConstructing cache model by few-shot visual features and labels.")
    cache_keys, cache_values = build_cache_model_imagenet(cfg, clip_model, train_loader_cache)

    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features_imagenet(cfg, "test", clip_model, test_loader)

    if args.mode=='baseline':
        # ------------------------------------------ Tip-Adapter-F ------------------------------------------
        run_tip_adapter_F(cfg, cache_keys, cache_values, test_features, test_labels, clip_weights, clip_model, train_loader_F)
    else:
        # ------------------------------------------ Tip-Adapter-F Syn ------------------------------------------    
        run_tip_adapter_F_syn(cfg, args, cache_keys, cache_values, test_features, test_labels, clip_weights, clip_model, train_loader_F, \
             imagenet.classnames, imagenet.template, imagenet_dict)
     

if __name__ == '__main__':
    main()