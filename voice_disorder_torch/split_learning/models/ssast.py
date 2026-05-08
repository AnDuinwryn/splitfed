
import torch
import torch.nn as nn

from voice_disorder_torch.models import ssast_ast


class SsastFinetuneAvgTokFull(nn.Module):
    def __init__(self, label_dim=527,
                 fshape=128, tshape=2, fstride=128, tstride=2,
                 input_fdim=128, input_tdim=1024, model_size='base',
                 load_pretrained_mdl_path=None):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if load_pretrained_mdl_path == None:
            raise ValueError('Please set load_pretrained_mdl_path to load a pretrained model.')
        sd = torch.load(load_pretrained_mdl_path, map_location=device)
        try:
            p_fshape, p_tshape = sd['module.v.patch_embed.proj.weight'].shape[2], sd['module.v.patch_embed.proj.weight'].shape[3]
            p_input_fdim, p_input_tdim = sd['module.p_input_fdim'].item(), sd['module.p_input_tdim'].item()
        except:
            raise ValueError('The model loaded is not from a torch.nn.Dataparallel object. Wrap it with torch.nn.Dataparallel and try again.')
        # Patch shape should be the same for pretraining and finetuning.
        if fshape != p_fshape or tshape != p_tshape:
            raise ValueError('The patch shape of pretraining and fine-tuning is not consistent, pretraining: f={:d}, t={:d}, finetuning: f={:d}, t={:d}.'.format(p_fshape, p_tshape, fshape, tshape))
        # print(f'Now loading a pretrained model from {load_pretrained_mdl_path}.')

        # During pretraining stage, fshape = fstride and tshape = tstride given that no patch embedding is used.
        # input_fdim and input_tdim control positional embedding cut/interpolation.
        # input_fdim may better be the same during pretraining stage and finetuning stage,
        # while input_tdim holds the opposite.
        audio_mdl = ssast_ast.ASTModel(fstride=p_fshape, tstride=p_tshape, 
                                        fshape=p_fshape, tshape=p_tshape, 
                                        input_fdim=p_input_fdim, input_tdim=p_input_tdim, 
                                        pretrain_stage=True, model_size=model_size)
        audio_mdl = torch.nn.DataParallel(audio_mdl)
        audio_mdl.load_state_dict(sd, strict=False)

        self.v = audio_mdl.module.v
        self.orig_embedding_dim = self.v.pos_embed.shape[2]
        self.cls_token_num = audio_mdl.module.cls_token_num

        # mlp head for finetuning
        self.mlp_head = nn.Sequential(nn.LayerNorm(self.orig_embedding_dim),
                                      nn.Linear(self.orig_embedding_dim, label_dim))
        f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim, fshape, tshape)
        p_f_dim, p_t_dim = audio_mdl.module.p_f_dim, audio_mdl.module.p_t_dim

        num_patches = f_dim * t_dim
        p_num_patches = p_f_dim * p_t_dim
        self.v.patch_embed.num_patches = num_patches
        # print('fine-tuning patch split stride: frequncey={:d}, time={:d}'.format(fstride, tstride))
        # print('fine-tuning number of patches={:d}'.format(num_patches))
        if fstride != p_fshape or tstride != p_tshape:
            new_proj = nn.Conv2d(1, self.orig_embedding_dim, kernel_size=(fshape, tshape), stride=(fstride, tstride))
            # but the weights of patch embedding layer is still got from the pretrained models
            new_proj.weight = torch.nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
            new_proj.bias = self.v.patch_embed.proj.bias
            self.v.patch_embed.proj = new_proj
        
        new_pos_embed = self.v.pos_embed[:, self.cls_token_num:, :].detach().reshape(1, p_num_patches, self.orig_embedding_dim).transpose(1, 2).reshape(1, self.orig_embedding_dim, p_f_dim, p_t_dim)
        
        # Cut or interpolate the positional embedding.
        if t_dim < p_t_dim:
            new_pos_embed = new_pos_embed[:, :, :, int(p_t_dim/2) - int(t_dim / 2): int(p_t_dim/2) - int(t_dim / 2) + t_dim]
        else:
            new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(8, t_dim), mode='bilinear')
        if f_dim < p_f_dim:
            new_pos_embed = new_pos_embed[:, :, int(p_f_dim/2) - int(f_dim / 2): int(p_f_dim/2) - int(f_dim / 2) + t_dim, :]
        else:
            new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
        new_pos_embed = new_pos_embed.reshape(1, self.orig_embedding_dim, num_patches).transpose(1, 2)
        self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :self.cls_token_num, :].detach(), new_pos_embed], dim=1))


    # Get the shape of intermediate representation.
    def get_shape(self, fstride, tstride, input_fdim, input_tdim, fshape, tshape):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.orig_embedding_dim, kernel_size=(fshape, tshape), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)
        B = x.shape[0]
        x = self.v.patch_embed(x)
        if self.cls_token_num == 2:
            cls_tokens = self.v.cls_token.expand(B, -1, -1)
            dist_token = self.v.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_token, x), dim=1)
        else:
            cls_tokens = self.v.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)

        for _, blk in enumerate(self.v.blocks):
            x = blk(x)
        x = self.v.norm(x)

        # average output of all tokens except cls token(s)
        x = torch.mean(x[:, self.cls_token_num:, :], dim=1)
        x = self.mlp_head(x)
        return x


class SsastClientEncoder(nn.Module):
    def __init__(self, label_dim=527, f_shape=16, t_shape=16, f_stride=10, t_stride=10,
                 input_fdim=128, input_tdim=259, model_size='base', load_pretrained_mdl_path=None, n_client_blocks=2):
        super().__init__()
        self.full_mdl = SsastFinetuneAvgTokFull(label_dim=label_dim,
                                         fshape=f_shape, tshape=t_shape, fstride=f_stride, tstride=t_stride,
                                         input_fdim=input_fdim, input_tdim=input_tdim, model_size=model_size,
                                         load_pretrained_mdl_path=load_pretrained_mdl_path)
        self.v = self.full_mdl.v
        self.cls_token_num = self.full_mdl.cls_token_num
        self.client_blocks = nn.ModuleList(self.v.blocks[:n_client_blocks])
        del self.v.blocks
        del self.v.norm
        del self.full_mdl.mlp_head
        del self.full_mdl

    def forward(self, x):
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)
        B = x.shape[0]
        x = self.v.patch_embed(x)
        if self.cls_token_num == 2:
            cls_tokens = self.v.cls_token.expand(B, -1, -1)
            dist_token = self.v.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_token, x), dim=1)
        else:
            cls_tokens = self.v.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for _, blk in enumerate(self.client_blocks):
            x = blk(x)
        return x


class SsastServerHead(nn.Module):
    def __init__(self, label_dim=527, f_shape=16, t_shape=16, f_stride=10, t_stride=10,
                 input_fdim=128, input_tdim=259, model_size='base', load_pretrained_mdl_path=None, n_client_blocks=2):
        super().__init__()
        self.full_mdl = SsastFinetuneAvgTokFull(label_dim=label_dim,
                                         fshape=f_shape, tshape=t_shape, fstride=f_stride, tstride=t_stride,
                                         input_fdim=input_fdim, input_tdim=input_tdim, model_size=model_size,
                                         load_pretrained_mdl_path=load_pretrained_mdl_path)
        self.server_blks = nn.ModuleList(self.full_mdl.v.blocks[n_client_blocks:])
        self.v_norm = self.full_mdl.v.norm
        self.mlp_head = self.full_mdl.mlp_head
        self.cls_token_num = self.full_mdl.cls_token_num
        del self.full_mdl.v.patch_embed
        del self.full_mdl.v.pos_embed
        del self.full_mdl.v.pos_drop
        del self.full_mdl.v.cls_token
        del self.full_mdl.v.dist_token
        del self.full_mdl.v.blocks
        del self.full_mdl

    def forward(self, x):
        for _, blk in enumerate(self.server_blks):
            x = blk(x)
        x = self.v_norm(x)
        x = torch.mean(x[:, self.cls_token_num:, :], dim=1)
        x = self.mlp_head(x)
        return x
