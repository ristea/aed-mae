import numpy as np
import torch
from einops import rearrange
from torch import nn
import torch.nn.functional as F
from model.cvt import ConvEmbed, Block
from util.morphology import Erosion2d, Dilation2d


class MaskedAutoencoderCvT(nn.Module):
    def __init__(self, img_size=(512,512), patch_size=16, in_chans=9, out_chans=4,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 use_only_masked_tokens_ab=False, abnormal_score_func='L1', masking_method="random_masking",
                 grad_weighted_loss=True, student_depth=1):
        super().__init__()
        # --------------------------------------------------------------------------
        # Abnormal specifics
        self.use_only_masked_tokens_ab = use_only_masked_tokens_ab
        self.abnormal_score_func = abnormal_score_func[0]
        self.abnormal_score_func_TS = abnormal_score_func[1]
        # --------------------------------------------------------------------------

        self.masking = getattr(self, masking_method)
        self.grad_weighted_loss=grad_weighted_loss

        assert 0 < student_depth < decoder_depth
        self.student_depth = student_depth
        self.train_TS = False
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = ConvEmbed(
            # img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            stride=patch_size,
            padding=0,
            embed_dim=embed_dim,
            norm_layer=norm_layer
        )
        self.patch_size = patch_size
        self.num_patches = img_size[0]//patch_size*img_size[1]//patch_size
        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, embed_dim)
        )

        self.blocks = nn.ModuleList([
            Block(embed_dim, embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * out_chans, bias=True)  # decoder to patch

        self.decoder_student_block = Block(decoder_embed_dim, decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
        self.decoder_student_norm = norm_layer(decoder_embed_dim)
        self.decoder_student_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * out_chans, bias=True)  # decoder to patch
        self.out_chans=out_chans
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss
        self.erosion = Erosion2d(1, 1, 2, soft_max=False)
        self.dilation = Dilation2d(1, 1, 3, soft_max=False)

        self.erosion_3 = Erosion2d(3, 3, 2, soft_max=False)
        self.dilation_3 = Dilation2d(3, 3, 3, soft_max=False)

    def freeze_backbone(self):
        self.cls_token.requires_grad = False
        self.mask_token.requires_grad = False
        for param in self.norm.parameters():
            param.requires_grad = False
        for param in self.decoder_norm.parameters():
            param.requires_grad = False
        for param in self.blocks.parameters():
            param.requires_grad = False
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        for param in self.decoder_embed.parameters():
            param.requires_grad = False
        for param in self.decoder_pred.parameters():
            param.requires_grad = False
        for i in range(0, len(self.decoder_blocks)):
            for param in self.decoder_blocks[i].parameters():
                param.requires_grad = False

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] % p == 0 and imgs.shape[3] % p == 0

        h = imgs.shape[2] // p
        w = imgs.shape[3] // p

        x = imgs.reshape(shape=(imgs.shape[0], self.out_chans, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * self.out_chans))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = 20
        w=40
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.out_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.out_chans, h * p, w * p))
        return imgs

    def random_masking(self, x, mask_ratio, grad_mask):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, D, H, W = x.shape  # batch, length, dim
        L = H*W
        x = rearrange(x, 'b c h w -> b (h w) c')
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        self.masked_H = H
        self.masked_W = int(W*(1.-mask_ratio))
        self.H = H
        self.W = W
        # x_masked = rearrange(x_masked, 'b (h w) c -> b c h w', h=self.masked_H, w=self.masked_W)
        return x_masked, mask, ids_restore

    def grad_masking_v1(self, x, mask_ratio, grad_mask):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        grad_mask = F.max_pool2d(grad_mask, self.patch_size).max(1).values
        grad_mask = rearrange(grad_mask, 'b h w -> b (h w)')

        N, D, H, W = x.shape  # batch, length, dim
        L = H*W
        x = rearrange(x, 'b c h w -> b (h w) c')
        len_keep = int(L * (1 - mask_ratio))

        # sort noise for each sample
        ids_shuffle = torch.argsort(grad_mask, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        self.masked_H = H
        self.masked_W = int(W*(1.-mask_ratio))
        self.H = H
        self.W = W
        # x_masked = rearrange(x_masked, 'b (h w) c -> b c h w', h=self.masked_H, w=self.masked_W)
        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio, grad_mask):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        # x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.masking(x, mask_ratio, grad_mask)
        # x = rearrange(x, 'b c h w -> b (h w) c')
        # append cls token
        cls_token = self.cls_token
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x, self.masked_H, self.masked_W)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x, self.H, self.W)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_decoder_TS(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # apply Student Transformer blocks
        for idx in range(0, self.student_depth):
            x = self.decoder_blocks[idx](x, self.H, self.W)
        x_student = self.decoder_student_block(x, self.H, self.W)
        x_student = self.decoder_student_norm(x_student)
        x_student = self.decoder_student_pred(x_student)
        x_student = x_student[:, 1:, :]

        for idx in range(self.student_depth, len(self.decoder_blocks)):
            x = self.decoder_blocks[idx](x, self.H, self.W)

        # predictor projection
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        # remove cls token
        x = x[:, 1:, :]

        return x_student, x

    def forward_loss(self, imgs, gradients, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        min_magnitude_anomaly = torch.ones((gradients.shape[0],1,1,1), device=imgs.device) * 128
        if self.grad_weighted_loss:
            anomaly_map = imgs[:, 3:, :, :]
            anomaly_map = torch.clip(anomaly_map, min=0, max=1)
            anomaly_map *= torch.maximum(min_magnitude_anomaly, torch.amax(gradients, dim=(1, 2, 3), keepdim=True))
            gradients += anomaly_map
            grad_weights = F.max_pool2d(gradients, self.patch_size).mean(1)
            grad_weights = rearrange(grad_weights, 'b h w -> b (h w)')
            # grad_weights = (grad_weights - torch.amin(grad_weights, keepdim=True)) / \
            #                (torch.amax(grad_weights, keepdim=True) - torch.amin(grad_weights, keepdim=True))
            grad_weights = grad_weights / grad_weights.sum(dim=1, keepdims=True)
            loss = (loss * grad_weights).sum()
        else:
            loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward_loss_TS(self, preds_stud, preds_teacher, mask):
        loss = (preds_stud - preds_teacher) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches

        return loss

    def forward(self, imgs, targets, grad_mask=None, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio, grad_mask)

        if self.train_TS is False:
            pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
            loss = self.forward_loss(targets, grad_mask, pred, mask)
            if self.training:
                return loss, pred, mask
            else:
                return loss, pred, mask, self.abnormal_score(targets, pred, mask, grad_mask)
        else:
            pred_stud, pred_teacher = self.forward_decoder_TS(latent, ids_restore)  # [N, L, p*p*3]
            loss = self.forward_loss_TS(pred_stud, pred_teacher, mask)
            if self.training:
                return loss, pred_stud, mask
            else:
                return loss, pred_teacher, mask, self.abnormal_score_TS(targets, pred_stud, pred_teacher, mask, grad_mask)

    def abnormal_score(self, imgs, pred, mask, gradients):
        imgs = self.patchify(imgs)
        if self.use_only_masked_tokens_ab:
            mask = mask.bool()
            selected_pred = []
            selected_lbl = []
            for i in range(0, imgs.shape[0]):
                selected_pred.append(pred[i][mask[i]])
                selected_lbl.append(imgs[i][mask[i]])

            pred = torch.stack(selected_pred)
            imgs = torch.stack(selected_lbl)
        return ((imgs - pred) ** 2).mean((1, 2))  # MSE

    def abnormal_score_TS(self, imgs, pred_stud, pred_teacher, mask, gradients):
        imgs = self.patchify(imgs)
        grad_weights = F.avg_pool2d(gradients, self.patch_size).mean(1)
        grad_weights = rearrange(grad_weights, 'b h w -> b (h w)')
        grad_weights = grad_weights / grad_weights.sum(dim=1, keepdims=True)
        if self.use_only_masked_tokens_ab:
            mask = mask.bool()
            selected_pred_stud = []
            selected_pred_teacher = []
            selected_lbl = []
            selected_gradients = []
            for i in range(0, imgs.shape[0]):
                selected_pred_stud.append(pred_stud[i][mask[i]])
                selected_pred_teacher.append(pred_teacher[i][mask[i]])
                selected_lbl.append(imgs[i][mask[i]])
                selected_gradients.append(grad_weights[i][mask[i]])

            pred_stud_masked = torch.stack(selected_pred_stud)
            pred_teacher_masked = torch.stack(selected_pred_teacher)
            imgs_masked = torch.stack(selected_lbl)
            grad_weights_masked = torch.stack(selected_gradients)
        output = []
        if self.abnormal_score_func_TS == "L1":
            output.append(torch.abs(pred_teacher - pred_stud).mean((2)))  # MAE
            output.append(torch.abs(imgs - pred_teacher).mean((2)))
            return [output[0].mean(1), output[1].mean(1)]
        elif self.abnormal_score_func_TS == "L2":

            output.append((((pred_teacher - pred_stud) ** 2).mean(2)))
            output.append((((imgs - pred_teacher) ** 2).mean(2)))
            return [output[0].mean(1), output[1].mean(1)]

    def process_result(self, gradients, pred_stud, pred_teacher, do_erosion=True):
        gradients = gradients.mean(dim=1,keepdim=True)
        gradients = (gradients - torch.amin(gradients, dim=(1, 2), keepdim=True)) / (
                    torch.amax(gradients, dim=(1, 2), keepdim=True)
                    - torch.amin(gradients, dim=(1, 2), keepdim=True))

        teacher_student = ((pred_teacher - pred_stud) ** 2)


        if do_erosion:
            teacher_student = self.unpatchify(teacher_student)
            teacher_student *= gradients


            teacher_student[:, -1:] = self.erosion(teacher_student[:, -1:])
            teacher_student[:, -1:] = self.dilation(teacher_student[:, -1:])
            teacher_student[:, -1:] = self.dilation(teacher_student[:, -1:])

            teacher_student[:, :-1] = self.erosion_3(teacher_student[:, :-1])
            teacher_student[:, :-1] = self.dilation_3(teacher_student[:, :-1])
            teacher_student[:, :-1] = self.dilation_3(teacher_student[:, :-1])
            #
            teacher_student = self.patchify(teacher_student)
        return teacher_student.mean(2)
