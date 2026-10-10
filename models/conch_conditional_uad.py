import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .film import FiLMCondition

class ViTill(nn.Module):
    def __init__(
            self,
            encoder,
            bottleneck,
            decoder,
            target_layers=[2, 3, 4, 5, 6, 7, 8, 9],
            fuse_layer_encoder=[[0, 1, 2, 3, 4, 5, 6, 7]],
            fuse_layer_decoder=[[0, 1, 2, 3, 4, 5, 6, 7]],
            mask_neighbor_size=0,
            remove_class_token=False,
            bottleneck_fusion = True,
            encoder_require_grad_layer=[],
            embed_dim = 768
    ) -> None:
        super(ViTill, self).__init__()

        self.visual_encoder = encoder.visual
        self.num_prefix_tokens = self.visual_encoder.trunk.num_prefix_tokens

        self.bottleneck = bottleneck
        self.decoder = decoder
        self.target_layers = target_layers
        self.fuse_layer_encoder = fuse_layer_encoder
        self.fuse_layer_decoder = fuse_layer_decoder
        self.remove_class_token = remove_class_token
        self.encoder_require_grad_layer = encoder_require_grad_layer

        self.mask_neighbor_size = mask_neighbor_size
        self.bottleneck_fusion = bottleneck_fusion

        self.film_in = FiLMCondition(cond_dim=512, feat_dim=embed_dim)
        self.film_dec = nn.ModuleList([
            FiLMCondition(cond_dim=512, feat_dim=embed_dim) for _ in range(len(self.decoder))
            ])

    def forward(self, x):
        x = self.visual_encoder.trunk.patch_embed(x)
        x = self.visual_encoder.trunk._pos_embed(x)
        x = self.visual_encoder.trunk.patch_drop(x)
        x = self.visual_encoder.trunk.norm_pre(x)

        en_list = []

        for i, blk in enumerate(self.visual_encoder.trunk.blocks):
            x = blk(x)
            if i in self.target_layers:
                en_list.append(x)

        x = self.visual_encoder.trunk.norm(x)

        pooled = self.visual_encoder.attn_pool_contrast(x)[:, 0] # single query
        pooled = self.visual_encoder.ln_contrast(pooled)
        condition = pooled @ self.visual_encoder.proj_contrast

        side = int(math.sqrt(en_list[0].shape[1]))

        #prefix token is cls token + reg token
        if self.remove_class_token:
            en_list = [e[: , self.num_prefix_tokens:] for e in en_list]

        if self.bottleneck_fusion:
            x = self.fuse_feature(en_list)

        x = self.film_in(x, condition)
        for i, blk in enumerate(self.bottleneck):
            x = blk(x)

        if self.mask_neighbor_size > 0:
            attn_mask = self.generate_mask(side, x.device)
        else:
            attn_mask = None

        de_list = []
        for i, blk in enumerate(self.decoder):
            x = self.film_dec[i](x, condition)
            x = blk(x, attn_mask=attn_mask)
            de_list.append(x)

        de_list = de_list[::-1]

        en = [self.fuse_feature([en_list[idx] for idx in idxs]) for idxs in self.fuse_layer_encoder]
        de = [self.fuse_feature([de_list[idx] for idx in idxs]) for idxs in self.fuse_layer_decoder]
        if not self.remove_class_token:  # prefix tokens have not been removed above
            en = [e[:, self.num_prefix_tokens:, :] for e in en]
            de = [d[:, self.num_prefix_tokens:, :] for d in de]

        en = [e.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for e in en]

        de = [d.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for d in de]

        return en, de

    def fuse_feature(self, feat_list):
        return torch.stack(feat_list, dim=1).mean(dim=1)

    def generate_mask(self, feature_size, device='cuda'):
        """
        Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        """
        h, w = feature_size, feature_size
        hm, wm = self.mask_neighbor_size, self.mask_neighbor_size
        mask = torch.ones(h, w, h, w, device=device)
        for idx_h1 in range(h):
            for idx_w1 in range(w):
                idx_h2_start = max(idx_h1 - hm // 2, 0)
                idx_h2_end = min(idx_h1 + hm // 2 + 1, h)
                idx_w2_start = max(idx_w1 - wm // 2, 0)
                idx_w2_end = min(idx_w1 + wm // 2 + 1, w)
                mask[
                idx_h1, idx_w1, idx_h2_start:idx_h2_end, idx_w2_start:idx_w2_end
                ] = 0
        mask = mask.view(h * w, h * w)
        if self.remove_class_token:
            return mask
        mask_all = torch.ones(h * w + self.num_prefix_tokens,
                              h * w + self.num_prefix_tokens, device=device)
        mask_all[self.num_prefix_tokens:, self.num_prefix_tokens:] = mask
        return mask_all
