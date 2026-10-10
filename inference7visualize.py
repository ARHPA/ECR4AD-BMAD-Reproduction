#adapted from https://github.com/ljl86092297/Anomaly_Detection/blob/main/inference7visualize.py
import torch
import torch.nn as nn
import numpy as np
import wandb
import os
from torch.utils.data import DataLoader, ConcatDataset
from dataloader.camelyon16_bmad import Camelyon16BMAD
from models.conch_conditional_uad import ViTill, ViTillv2
from networks import vit_encoder
from networks.vision_transformer import Block as VitBlock, bMlp, Attention, LinearAttention2
from functools import partial
import warnings
from utils import get_gaussian_kernel,min_max_norm,cvt2heatmap,cal_anomaly_maps,show_cam_on_image, seed_everything
import cv2
from tqdm import tqdm
import yaml

def load_model(config, weight_path):

    target_layers = config.target_layers
    fuse_layer_encoder = config.fuse_layers
    fuse_layer_decoder = config.fuse_layers

    encoder_name = config.backbone
    encoder = vit_encoder.load(encoder_name)
    target_layers = config.target_layers

    if "conch" == encoder_name:
        embed_dim, num_heads = 768, 12
    elif "gigapath" == encoder_name:
        embed_dim, num_heads = 1536, 24
    elif "uni2" == encoder_name:
        #24 encoder blocks in total
        embed_dim, num_heads = 1536, 24
    elif "uni" == encoder_name or "conchv1_5" == encoder_name:
        #24 encoder blocks in total
        embed_dim, num_heads = 1024, 16
    elif 'small' in encoder_name:
        embed_dim, num_heads = 384, 6
    elif 'base' in encoder_name:
        embed_dim, num_heads = 768, 12
    elif 'large' in encoder_name:
        embed_dim, num_heads = 1024, 16
    else:
        raise ValueError("Architecture not in small, base, large.")

    num_heads = config.num_heads

    bottleneck = []
    decoder = []
    encoder_require_grad_layer = []

    bottleneck.append(bMlp(embed_dim, embed_dim * 4, embed_dim, drop = config.bottleneck_dropout))
    bottleneck = nn.ModuleList(bottleneck)

    #default attention func
    attention_func = Attention
    if config.attn_method == "linear_attn":
        attention_func = LinearAttention2
    elif config.attn_method == "qkv_attn":
        attention_func = Attention
    else:
        raise ValueError("not implemented attetnion method")

    for i in range(len(target_layers)):
        blk = VitBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=4., qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8), attn=LinearAttention2)
        decoder.append(blk)
    decoder = nn.ModuleList(decoder)

    model = ViTill(encoder=encoder, bottleneck=bottleneck, decoder=decoder, target_layers=target_layers, mask_neighbor_size=0, fuse_layer_encoder=fuse_layer_encoder, fuse_layer_decoder=fuse_layer_decoder, encoder_require_grad_layer = encoder_require_grad_layer, remove_class_token = config.remove_class_token, bottleneck_fusion = config.bottleneck_fusion)

    model = model.to(device)
    model.load_state_dict(torch.load(weight_path))

    return model


def visualize(model, dataloader, device, _class_='None', save_name='save'):
    model.eval()
    save_dir = os.path.join('./visualize', save_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

    with torch.no_grad():
        for img, label, file_path in tqdm(dataloader):
            img = img.to(device)
            output = model(img)
            en, de = output[0], output[1]
            anomaly_map, _ = cal_anomaly_maps(en, de, img.shape[-1])
            anomaly_map = gaussian_kernel(anomaly_map)

            for i in range(0, anomaly_map.shape[0]):
                heatmap = min_max_norm(anomaly_map[i, 0].cpu().numpy())
                heatmap = cvt2heatmap(heatmap * 255)
                im = img[i].permute(1, 2, 0).cpu().numpy()
                im = im * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                im = (im * 255).astype('uint8')
                im = im[:, :, ::-1]
                hm_on_img = show_cam_on_image(im, heatmap)
                #mask = (gt[i][0].numpy() * 255).astype('uint8')
                save_dir_class = os.path.join(save_dir, str(_class_))
                if not os.path.exists(save_dir_class):
                    os.mkdir(save_dir_class)
                save_name = os.path.basename(file_path[i]).replace(".png", "")
                cv2.imwrite(save_dir_class + '/' + save_name + '_img.png', im)
                cv2.imwrite(save_dir_class + '/' + save_name  + '_cam.png', hm_on_img)

    return


def load_data7visual(model, test_loader, device):
    visualize(model, test_loader, device, save_name='save')


if __name__ == '__main__':
    wandb.login(key = "your wandb key here")

    with open("conch_bmad_config.yaml", "r") as f:
        raw_config = yaml.safe_load(f)
    config_dict = {key: val["value"] for key, val in raw_config.items()}

    wandb.init(project="path_ad", config=config_dict)
    seed_everything(seed = wandb.config.seed)
    print(wandb.config)

    test_set_list = []
    batch_size = 128
    crop_size = wandb.config.crop_size
    image_size = wandb.config.image_size

    if wandb.config.dataset == "LNM_Prostate":
        test_set = LNMProstate(setname = "test", crop_size = crop_size, image_size = image_size)
    elif wandb.config.dataset == "Camelyon16BMAD":
        test_set = Camelyon16BMAD(setname = "test", crop_size = crop_size, image_size = image_size)
    elif wandb.config.dataset == "Camelyon16":
        test_set = Camelyon16(setname = "test", crop_size = crop_size, image_size = image_size)

    test_loader = DataLoader(dataset=test_set, num_workers=16, pin_memory=False, shuffle = False, batch_size = 128)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    #your weight path here
    weight_path = ""
    model = load_model(wandb.config, weight_path = weight_path)
    load_data7visual(model, test_loader, device)
