import torch
from torch import nn, einsum
from einops import rearrange
from einops.layers.torch import Rearrange


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d



class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        
        self.residual_block = nn.Sequential(nn.Conv2d(channels, channels, 3, stride=1, padding=1),
                                            nn.LeakyReLU(0.2, inplace=True),
                                            nn.InstanceNorm2d(channels)
                                            )

    def forward(self, x):
        return x + self.residual_block(x)



class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        '''
        process and downscale the image feature maps
        '''
        self.model = nn.Sequential(*[ResidualBlock(in_channels), 
                                     nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
                                     nn.LeakyReLU(0.2, inplace=True),
                                     nn.GroupNorm(8,out_channels)])

    def forward(self, x):
        return self.model(x)



class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()
        '''
        process and upscale the image feature maps
        '''
        self.model = nn.Sequential(*[ResidualBlock(in_channels),
                                     nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2),
                                     nn.LeakyReLU(0.2, inplace=True),
                                     nn.GroupNorm(8,out_channels),
                                     ])

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x
    
    
class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        generic one layer FC NN for embedding things  
        '''
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)
    


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context = None):
        b, n, _, h = *x.shape, self.heads
        context = default(context, x)

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x, t):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x + t
        return x


class UViT(nn.Module):
    def __init__(self, *, n_feat = 128, patch_size = 7, dim = 128, depth = 4, heads = 8, dim_head = 64, dropout = 0.1, emb_dropout = 0.):
        super().__init__()

        self.in_channels = 1
        self.n_feat = n_feat

        self.down1 = UnetDown(self.in_channels, n_feat)
        self.down2 = UnetDown(n_feat, 2*n_feat)

        num_patches = 4 * n_feat
        patch_dim = patch_size**2

        self.patch_to_embedding = nn.Sequential(
            Rearrange('b c h w -> b c (h w)', h = patch_size, w = patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.time_embedding = nn.Sequential(
                                            EmbedFC(1, n_feat*2),
                                            Rearrange('b d -> b d ()')
                                        )
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, dim*4, dropout)

        self.embedding_to_patch = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, patch_dim),
            Rearrange('b c (h w) -> b c h w', h = patch_size, w = patch_size)
        )
        
        self.timeembed1 = EmbedFC(1, 2*n_feat)
        self.timeembed2 = EmbedFC(1, 1*n_feat)

        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)

        self.final_block = nn.Sequential(nn.Conv2d((n_feat)+1, n_feat//4, 3, 1, 1), 
                                        nn.LeakyReLU(0.2, inplace=True),
                                        nn.Conv2d(n_feat//4, 1, 3, 1, 1)
                                        )


    def forward(self, x, t):
        # downsample unet
        down1 = self.down1(x)
        down2 = self.down2(down1)
        
        # begin ViT backbone
        latent = self.patch_to_embedding(down2)
        b, n, _ = latent.shape

        time_embedding = self.time_embedding(t)
        latent += self.pos_embedding[:, :(n)]
        latent = self.dropout(latent)

        latent = self.transformer(latent, time_embedding)
        
        latent = self.embedding_to_patch(latent)
        # end ViT backbone
        
        # compute other time embeddings
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)
        
        # upsample unet
        up1 = self.up1(latent + temb1, down2)
        up2 = self.up2(up1 + temb2, down1)
        out = self.final_block(torch.cat((up2, x), 1))
        return out