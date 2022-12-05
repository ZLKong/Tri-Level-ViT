

""" 
Vision Transformer
Attention Mask: 
V2: mask (N, N) * L
"""

# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math 
from collections import OrderedDict

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

__all__ = [
    'deit_tiny_s2_patch16_224', 'deit_small_s2_dst_patch16_224', 'deit_base_s2_dst_patch16_224', 'deit_base_s2_dst_patch16_384',
]


# attn_keep_ratio = 0.3


# Sparse Token >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def batched_index_select(input, dim, index):
    # input:(B, C, HW). index(B, N)
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)      # (B,C, N)



# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., token_num=196):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask=None):
        # attn_mask: (B, N+1, N+1) input-dependent

        eps = 1e-6
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # (B, N+1, C) -> (B, N, 3C) -> (B, N+1, 3, H, C/H) -> (3, B, H, N+1, C/H)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)                 # (B, H, N+1, C/H)

        attn = (q @ k.transpose(-2, -1)) * self.scale       #  (B, H, N+1, C/H) @ (B, H, C/H, N+1) -> (B, H, N+1, N+1)

        # Key pruning (attention level) >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        max_att = torch.max(attn, dim=-1, keepdim=True)[0]
        attn = attn - max_att

        attn = attn.to(torch.float32).exp_() * attn_mask.unsqueeze(1).to(torch.float32)     # (B, H, N+1, N+1)
        attn = (attn + eps/N) / (attn.sum(dim=-1, keepdim=True) + eps)          # (B, H, N+1, N+1)
        # attn = attn.softmax(dim=-1)                                           # (B, H, N+1, N+1)
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)     # (B, H, N+1, N+1) * (B, H, N+1, C/H) -> (B, H, N+1, C/H) -> (B, N+1, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # save attention map
        cls_attn = attn[:, :, 0, 1:].sum(1) / self.num_heads                      # (B, H, N) -> (B, N)
        patch_attn = attn[:, :, 1:, 1:].sum(1) / self.num_heads                   # (B, H, N, N) -> (B, N, N)
        return x, cls_attn, patch_attn
        
  



class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, token_keep_ratio=1.0, attn_keep_ratio=1.0, token_idx=None, token_prune=False,  attn_prune=False, attn_mask=None):
        
        x_att, cls_attn, patch_attn = self.attn(self.norm1(x), attn_mask)
        # x: (B, N+1, C)
        # cls_attn: (B, N)      [cls] token, sum is 1 
        # patch_attn: (B, N, N)     for each image patch
        x = x + self.drop_path(x_att)

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Token Prune >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        if token_prune:
            #print('keep_ratio',keep_ratio)
            x_cls_token = x[:, 0:1]         # (B, 1, C)
            B, N = cls_attn.shape           # N = 196
            num_keep_node = math.ceil( N * token_keep_ratio )     # 196 r

            # attentive token
            token_idx = cls_attn.topk(num_keep_node, dim=1)[1]              # (B, rN)        without gradient
            x_attentive = batched_index_select(x[:, 1:], 1, token_idx)      # (B, N, C) -> (B, rN, C)
            
            x = torch.cat([x_cls_token, x_attentive], dim=1)        # (B, 1+rN, C)
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Attention Prune >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        if attn_prune:
            # get attention map of pruned token
            patch_attn_prune = batched_index_select(patch_attn, 1, token_idx)           # (B, N, N) -> (B, rN, N)
            patch_attn_prune = batched_index_select(patch_attn_prune, 2, token_idx)     # (B, rN, N) -> (B, rN, rN)
            # 
            B, rN1, _ = x.shape
            rN = rN1 - 1
            num_keep_attn = math.ceil( rN * attn_keep_ratio)                                            #  rN * ra
            top_val, _ = patch_attn_prune.topk(dim=2, k=num_keep_attn)                           # (B, rN, rN * ra)

            attn_mask_p = (patch_attn_prune >= top_val[:, :, -1].unsqueeze(-1).expand(-1, -1, rN)) + 0        # （B, rN, rN） without gradient     0/1 mask
            # TODO: may add some random here

            attn_mask = torch.ones(B, rN1, rN1).to(x.device)                              # (B, rN+1, rN+1)
            attn_mask[:, 1:, 1:] = attn_mask_p

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        
        # x = x + self.drop_path(self.attn(self.norm1(x)))          # old form  
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, token_idx, attn_mask, cls_attn


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """
    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
                # map for all networks, the feature metadata has reliable channel and stride info, but using
                # stride to calc feature dim requires info about padding of each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))
                if isinstance(o, (list, tuple)):
                    o = o[-1]  # last feature if backbone outputs list/tuple of features
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            if hasattr(self.backbone, 'feature_info'):
                feature_dim = self.backbone.feature_info.channels()[-1]
            else:
                feature_dim = self.backbone.num_features
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Conv2d(feature_dim, embed_dim, 1)

    def forward(self, x):
        x = self.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x



class VisionTransformerTokenAttn(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., hybrid_backbone=None, norm_layer=None,
                 pruning_loc=None, token_ratio=None, distill=False):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            hybrid_backbone (nn.Module): CNN backbone to use in-place of PatchEmbed module
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()

        print('## diff vit pruning method')
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, 
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        # Token Predictor Parameters >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        self.pruning_loc = [3, 6, 9]      # [3, 6, 9]
        self.token_ratio = token_ratio      # [0.7, 0.49, 0.343]

        self.distill = distill

        # Initialization
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        _init_vit_weights(module)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, ratio, attn_ratio):
        # x: (B, C, H, W)
        # ratio: 0.7 (default)

        # Patch Token >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        B = x.shape[0]
        x = self.patch_embed(x)         # (B, N, C)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)       # (B, N+1, C)

        x = x + self.pos_embed                      # (B, N+1, C)
        x = self.pos_drop(x)                        # drop out, p=0.0   

        N1 = x.shape[1]      
        
        # >>>>>>>>>>
        num_token_all = x.shape[1]      # N + 1
        token_idx = torch.arange(0, num_token_all - 1).long().unsqueeze(0).expand(B, -1).to(x.device)      # (B, N)      initial
        attn_mask = torch.ones(B, N1, N1).to(x.device)          # (B, N+1, N+1) 1

        #stage_ratio = [ratio, ratio ** 2, ratio ** 3]
        # Transformer Block >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        for i, blk in enumerate(self.blocks):
            if i in self.pruning_loc:           # i = 3, 6, 9
                x, token_idx, attn_mask, cls_attn = blk(x, token_keep_ratio= ratio, token_idx=token_idx, token_prune=True, attn_prune=True, attn_mask=attn_mask, attn_keep_ratio=attn_ratio)
            else:
                x, token_idx, attn_mask, cls_attn = blk(x, token_keep_ratio=ratio, token_idx=token_idx, token_prune=False, attn_prune=False,  attn_mask=attn_mask, attn_keep_ratio=attn_ratio)

    
        # Out put head >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        #print('cls_attn.shape',cls_attn.shape)
        x = self.norm(x)
        x = x[:, 0] #cls
        #print('x.shape',x.shape)
        x = self.pre_logits(x)
        x = self.head(x)
        #print('head x.shape',x.shape)
        #exit()
        return x, cls_attn


def _init_vit_weights(module: nn.Module, name: str = '', head_bias: float = 0., jax_impl: bool = False):
    """ ViT weight initialization
    * When called without n, head_bias, jax_impl args it will behave exactly the same
      as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    * When called w/ valid n (module name) and jax_impl=True, will (hopefully) match JAX impl
    """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        else:
            trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)




# ***************************************** Sparse Model (224) ******************************************
@register_model
def deit_tiny_patch16_224_attn_dst(pretrained=False, **kwargs):
    model = VisionTransformerTokenAttn(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_small_patch16_224_attn_dst(pretrained=False, **kwargs):
    model = VisionTransformerTokenAttn(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_patch16_224_attn_dst(pretrained=False, **kwargs):
    model = VisionTransformerTokenAttn(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


# ***************************************** Model (384) ******************************************
@register_model
def deit_base_patch16_384_attn_dst(pretrained=False, **kwargs):
    model = VisionTransformerTokenAttn(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model



