import os
os.environ["XFORMERS_DISABLED"] = "1"
XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
import torch
import torch.nn as nn
import numpy
import yaml
from dataloader.lnm_prostate import LNMProstate
from dataloader.camelyon16 import Camelyon16
from torch.utils.data import DataLoader
from networks.vision_transformer import Block as VitBlock, bMlp, Attention, LinearAttention, LinearAttention2, ConvBlock, FeatureJitter, SwitchLinearAttention
from networks import vit_encoder
from networks.dinov1.utils import trunc_normal_
from optimizers import StableAdamW
from utils import WarmCosineScheduler
from evaluation import patch_evaluation
from functools import partial
from losses import global_cosine_hm, global_cosine_focal, global_cosine_hm_percent
from tqdm import tqdm
import torch.nn.functional as F
from utils import seed_everything
import wandb

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_learning_rate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
        lr +=[ param_group['lr'] ]
    return lr

def worker_init_fn(worker_id):
    numpy.random.seed(numpy.random.get_state()[1][0] + worker_id)

def train():
    wandb.login(key = "your wandb key here")
    # Initialize wandb with the custom configuration
    #with open("configs/conch/conch_lnm_config_ecr4ad.yaml", "r") as f:
    with open("configs/conch/conch_camelyon16_config_ecr4ad.yaml", "r") as f:
        raw_config = yaml.safe_load(f)
        config_dict = {key: val["value"] for key, val in raw_config.items()}
        wandb.init(project="path_ad", config=config_dict)

    os.environ["CUDA_VISIBLE_DEVICES"] = wandb.config.gpu_id
    seed_everything(seed = wandb.config.seed)
    print(wandb.config)

    if "dino" in wandb.config.backbone:
        from models.dino_conditional_uad import ViTill
    elif "uni" in wandb.config.backbone or "gigapath" in wandb.config.backbone:
        #from models.uni_uad import ViTill
        from models.uni_conditional_uad import ViTill
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
    test_set_list = []
    test_slide_list = []

    if wandb.config.dataset == "LNM_Prostate":
        train_set = LNMProstate(setname = "train", image_size = image_size)
        val_set = LNMProstate(setname = "val", image_size = image_size)
        test_slide_list = LNMProstate.load_test_slides()
        for slide_name in test_slide_list:
            test_set_list.append(LNMProstate(setname = "test", image_size = image_size, slide_name = slide_name))

    elif wandb.config.dataset == "Camelyon16":
        train_set = Camelyon16(setname = "train", image_size = image_size)
        val_set = Camelyon16(setname = "val", image_size = image_size)
        test_slide_list = Camelyon16.load_test_slides()
        for slide_name in test_slide_list:
            test_set_list.append(Camelyon16(setname = "test", image_size = image_size, slide_name = slide_name))
    else:
        raise ValueError("unspport dataset")

    train_loader = None
    train_loader = DataLoader(dataset=train_set, batch_size = wandb.config.batch_size, num_workers=16, pin_memory = False, worker_init_fn=worker_init_fn, shuffle = True, drop_last=True)
    val_loader = DataLoader(dataset=val_set, batch_size = 128, num_workers=16, pin_memory = False, worker_init_fn=worker_init_fn, shuffle = False, drop_last=False)
    test_loader_list = [DataLoader(dataset=test_set, batch_size = 128, num_workers = 16, pin_memory = False, shuffle = False, drop_last = False) for test_set in test_set_list]

    target_layers = wandb.config.target_layers
    fuse_layer_encoder = wandb.config.fuse_layers
    fuse_layer_decoder = wandb.config.fuse_layers

    encoder_name = wandb.config.backbone
    encoder = vit_encoder.load(encoder_name)

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
    print(count_parameters(model))
    print(count_parameters(trainable))
    print('train image number:{}'.format(len(train_loader)))

    optimizer = StableAdamW([{'params': trainable.parameters()}], lr=wandb.config.learning_rate, betas=(0.9, 0.999), weight_decay=1e-4, amsgrad=True, eps=1e-10)
    lr_scheduler = WarmCosineScheduler(optimizer, base_value=wandb.config.learning_rate, final_value=2e-5, total_iters=total_iters, warmup_iters=100)

    it = 0
    max_auroc = 0

    for epoch in range(int(numpy.ceil(total_iters / len(train_loader)))):
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
                print("************finding threshold:***********")
                max_ratio = 0.01
                threshold = patch_evaluation(model, val_loader, max_ratio = max_ratio, find_threshold = True)
                print("threshold:", threshold)
                auroc_list, ap_list, f1_list, tpr_list, positive_fpr_list, negative_fpr_list = [], [], [], [], [], []
                for item, test_loader in zip(test_slide_list, test_loader_list):
                    metrics  = patch_evaluation(model, test_loader, threshold = threshold, max_ratio = max_ratio)
                    if len(metrics) > 1:
                        auroc, ap, f1, fpr, tpr = metrics
                        print('{}: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, FPR:{:.4f}, TPR:{:.4f}'.format(item, auroc, ap, f1, fpr, tpr))
                        auroc_list.append(auroc)
                        ap_list.append(ap)
                        f1_list.append(f1)
                        positive_fpr_list.append(fpr)
                        tpr_list.append(tpr)
                    else:
                        fpr = metrics[0]
                        print('{}:FPR:{:.4f}'.format(item, fpr))
                        negative_fpr_list.append(fpr)

                if numpy.mean(auroc_list) > max_auroc:
                    max_auroc = numpy.mean(auroc_list)
                    torch.save(model.state_dict(), os.path.join(wandb.config.save_path, "weights", wandb.config.dataset+'_'+wandb.config.backbone+'_best_model_slide_train_all_tiles.pth'))

                print('Max I-Auroc:{:.4f}, Mean: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, Positive FPR:{:.4f}, Negative FRP: {:.4f},  Mean FRP: {:.4f}, TRP:{:.4f}'.format(max_auroc, numpy.mean(auroc_list), numpy.mean(ap_list), numpy.mean(f1_list), numpy.mean(positive_fpr_list), numpy.mean(negative_fpr_list), numpy.mean(positive_fpr_list + negative_fpr_list), numpy.mean(tpr_list)))

                print("learning rate:", get_learning_rate(optimizer)) 
                model.train()

            it += 1
            if it == total_iters:
                break
                
        print('iter [{}/{}], loss:{:.4f}'.format(it, total_iters, numpy.mean(loss_list)))

if __name__ == "__main__":
    train()
