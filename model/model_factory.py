from functools import partial

from torch import nn

from model.mae_cvt import MaskedAutoencoderCvT


def mae_cvt_patch16(**kwargs):
    model = MaskedAutoencoderCvT(
        patch_size=16, embed_dim=256, depth=3, num_heads=4,
        decoder_embed_dim=128, decoder_depth=3, decoder_num_heads=4,
        mlp_ratio=2, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_cvt_patch8(**kwargs):
    model = MaskedAutoencoderCvT(
        patch_size=8, embed_dim=256, depth=3, num_heads=4,
        decoder_embed_dim=128, decoder_depth=3, decoder_num_heads=4,
        mlp_ratio=2, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model