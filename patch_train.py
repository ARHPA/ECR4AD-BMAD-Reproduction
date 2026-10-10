import argparse
import os
os.environ["XFORMERS_DISABLED"] = "1"
XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import numpy
import yaml
from dataloader.camelyon16_bmad import Camelyon16BMAD
from dataloader.gleason_arvaniti import GleasonArvaniti
# Camelyon16 / LNM_Prostate need whole-slide-image libraries (openslide/h5py).
# Import them lazily so patch-level datasets that do not require those libraries
# still work on a minimal runtime (e.g. Google Colab).
try:
    from dataloader.lnm_prostate import LNMProstate
    from dataloader.camelyon16 import Camelyon16
except Exception:
    LNMProstate = None
    Camelyon16 = None
from torch.utils.data import DataLoader
from networks.vision_transformer import Block as VitBlock, bMlp, Attention, LinearAttention2, FeatureJitter
from networks import vit_encoder
from networks.dinov1.utils import trunc_normal_
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from optimizers import StableAdamW
from utils import WarmCosineScheduler
from evaluation import patch_evaluation
from functools import partial
from losses import global_cosine_hm, global_cosine_focal, global_cosine_hm_percent
from tqdm import tqdm
import torch.nn.functional as F
import wandb
from utils import seed_everything

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_learning_rate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
          lr +=[ param_group['lr'] ]
    return lr

def worker_init_fn(worker_id):
    numpy.random.seed(numpy.random.get_state()[1][0] + worker_id)

DEFAULT_CONFIG = "configs/conch/conch_bmad_config_ecr4ad.yaml"

def parse_args():
    parser = argparse.ArgumentParser(description="ECR4AD patch-level training")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG, help="Path to the YAML training config.")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Checkpoint to resume from. Defaults to <save_path>/weights/<dataset>_<backbone>_last_patch_train.pth if it exists.")
    parser.add_argument("--wandb_key", type=str, default=None,
                        help="W&B API key. Falls back to the WANDB_API_KEY environment variable. Set WANDB_MODE=offline/disabled to run without W&B.")
    return parser.parse_args()

def train(args=None):
    if args is None:
        args = parse_args()
    # W&B is optional; never hardcode keys. Use --wandb_key or WANDB_API_KEY.
    wandb.login(key = args.wandb_key or os.environ.get("WANDB_API_KEY") or None)

    with open(args.config, "r") as f:
        raw_config = yaml.safe_load(f)
    config_dict = {key: val["value"] for key, val in raw_config.items()}

    wandb.init(project="path_ad", config=config_dict)

    os.environ["CUDA_VISIBLE_DEVICES"] = wandb.config.gpu_id
    seed_everything(seed = wandb.config.seed)
    print(wandb.config)

    if "dino" in wandb.config.backbone:
        if wandb.config.enable_ecr:
            from models.dino_conditional_uad import ViTill
        else:
            from models.dino_uad import ViTill
    elif "uni" in wandb.config.backbone or "gigapath" in wandb.config.backbone:
        if wandb.config.enable_ecr:
            from models.uni_conditional_uad import ViTill
        else:
            from models.uni_uad import ViTill
    elif "conchv1_5" == wandb.config.backbone: 
        from models.conchv1_5_uad import ViTill
    else:
        if wandb.config.enable_ecr:
            from models.conch_conditional_uad import ViTill
        else:
            from models.conch_uad import ViTill

    train_set = None
    test_set = None
    image_size = wandb.config.image_size
    total_iters = wandb.config.total_iters
    data_root = wandb.config.data_root
    test_set_list = []

    if wandb.config.dataset == "Camelyon16BMAD":
        train_set = Camelyon16BMAD(data_root = data_root, setname = "train", image_size = image_size)
        #test_set = Camelyon16BMAD(data_root = data_root, setname = "valid", image_size = image_size)
        test_set = Camelyon16BMAD(data_root = data_root, setname = "test", image_size = image_size)
    elif wandb.config.dataset == "GleasonArvaniti":
        train_set = GleasonArvaniti(data_root = data_root, setname = "train", image_size = image_size)
        val_set = GleasonArvaniti(data_root = data_root, setname = "val", image_size = image_size)
        test_set = GleasonArvaniti(data_root = data_root, setname = "test", image_size = image_size)
    else:
        raise ValueError("unsupport dataset")

    train_loader = DataLoader(dataset=train_set, batch_size = wandb.config.batch_size, num_workers=16, pin_memory = True, worker_init_fn=worker_init_fn, shuffle = True, drop_last=True)
    test_loader = DataLoader(dataset=test_set, num_workers=16, pin_memory=False, shuffle = True, batch_size = 128)

    target_layers = wandb.config.target_layers
    fuse_layer_encoder = wandb.config.fuse_layers
    fuse_layer_decoder = wandb.config.fuse_layers

    encoder_name = wandb.config.backbone
    encoder = vit_encoder.load(encoder_name)
    target_layers = wandb.config.target_layers

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

    num_heads = wandb.config.num_heads

    bottleneck = []
    decoder = []

    if wandb.config.feature_jitter:
        bottleneck.append(nn.Sequential(FeatureJitter(scale=20), bMlp(embed_dim, embed_dim * 4, embed_dim, drop=wandb.config.bottleneck_dropout)))
    else:
        bottleneck.append(bMlp(embed_dim, embed_dim * 4, embed_dim, drop = wandb.config.bottleneck_dropout))

    bottleneck = nn.ModuleList(bottleneck)

    #default attention func
    attention_func = Attention
    if wandb.config.attn_method == "linear_attn":
        attention_func = LinearAttention2
    elif wandb.config.attn_method == "qkv_attn":
        attention_func = Attention
    else:
        raise ValueError("not implemented attetnion method")

    for i in range(len(target_layers)):
        blk = VitBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=4., qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8), attn=attention_func )
        decoder.append(blk)

    decoder = nn.ModuleList(decoder)
    encoder_require_grad_layer = []

    for p in encoder.parameters():
        p.requires_grad = False

    model = ViTill(encoder=encoder, bottleneck=bottleneck, decoder=decoder, target_layers=target_layers, mask_neighbor_size=wandb.config.mask_neighbor_size, fuse_layer_encoder=fuse_layer_encoder, fuse_layer_decoder=fuse_layer_decoder, encoder_require_grad_layer = encoder_require_grad_layer, remove_class_token = wandb.config.remove_class_token, bottleneck_fusion = wandb.config.bottleneck_fusion, embed_dim = embed_dim)
    model = model.cuda()

    trainable = nn.ModuleList([bottleneck, decoder])

    ce_loss = nn.CrossEntropyLoss()
    
    for m in trainable.modules():
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    optimizer = StableAdamW([{'params': trainable.parameters()}], lr=wandb.config.learning_rate, betas=(0.9, 0.999), weight_decay=1e-4, amsgrad=True, eps=1e-10)
    lr_scheduler = WarmCosineScheduler(optimizer, base_value=wandb.config.learning_rate, final_value=2e-5, total_iters=total_iters, warmup_iters=100)

    it = 0
    max_auroc = 0
    start_epoch = 0
    last_ckpt = os.path.join(wandb.config.save_path, "weights", wandb.config.dataset+'_'+wandb.config.backbone+'_last_patch_train.pth')
    resume_path = args.resume_from or (last_ckpt if os.path.exists(last_ckpt) else None)
    if resume_path is not None:
        ckpt = torch.load(resume_path, map_location='cpu', weights_only=False)
        if 'model' in ckpt:
            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            lr_scheduler.load_state_dict(ckpt['scheduler'])
            it = ckpt['it']
            max_auroc = ckpt['max_auroc']
            start_epoch = it // len(train_loader)
            print('resumed from', resume_path, 'it:', it, 'max_auroc:', max_auroc)
        else:
            # Weights-only checkpoint (e.g. *_best_model_patch_train.pth): restart optimisation from these weights.
            model.load_state_dict(ckpt)
            print('loaded model weights from', resume_path)

    for epoch in range(start_epoch, int(numpy.ceil(total_iters / len(train_loader)))):
        model.train()
        loss_list = []
        optimizer.zero_grad()
        running_loss = 0.0

        for i, batch in enumerate(tqdm(train_loader)):
            image = batch[0].cuda()

            en, de = model(image)

            p_final = 0.9
            p = min(p_final * it / 1000, p_final)
            loss = global_cosine_hm_percent(en, de, p = p, factor = 0.1)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable.parameters(), max_norm=0.1)

            optimizer.step()
            loss_list.append(loss.item())
            lr_scheduler.step()

            if (it + 1) % wandb.config.eval_period == 0:
                metrics  = patch_evaluation(model, test_loader, max_ratio = 0.01)
                if len(metrics) > 1:
                    auroc, ap, f1, fpr, tpr = metrics 
                else:
                    fpr = metrics[0]

                if auroc > max_auroc:
                    max_auroc = auroc
                    torch.save(model.state_dict(), os.path.join(wandb.config.save_path, "weights", wandb.config.dataset+'_'+wandb.config.backbone+'_best_model_patch_train.pth'))

                print('Max-Auroc:{:.4f}, I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}'.format(max_auroc, auroc, ap, f1))
                wandb.log({"max_auroc": max_auroc, "auroc": auroc, "ap": ap, "f1": f1,})
                model.train()

            it += 1
            if it == total_iters:
                break
                
        print('iter [{}/{}], loss:{:.4f}'.format(it, total_iters, numpy.mean(loss_list)))
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': lr_scheduler.state_dict(), 'it': it, 'epoch': epoch, 'max_auroc': max_auroc}, last_ckpt)

if __name__ == "__main__":
    train()
