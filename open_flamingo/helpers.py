"""
Based on: https://github.com/lucidrains/flamingo-pytorch
"""

import torch
from einops import rearrange, repeat
from einops_exts import rearrange_many
from torch import einsum, nn

def chunk2mask(input_tensor:torch.Tensor):
    INT_MAX=10000
    # 创建输出张量并进行元素替换
    output_tensor = input_tensor.clone()

    for row in range(output_tensor.size(0)):
        nonzero_indices = torch.nonzero(input_tensor[row], as_tuple=False).flatten()
        if len(nonzero_indices) == 0:
            continue
        nonzero_values = input_tensor[row, nonzero_indices]
        for i in range(1, len(nonzero_indices)):
            output_tensor[row, nonzero_indices[i-1]+1:nonzero_indices[i]+1] = nonzero_values[i]

    # 对每一行处理最后一组零值
    for row in range(output_tensor.size(0)):
        nonzero_indices = torch.nonzero(output_tensor[row], as_tuple=False).flatten()
        if len(nonzero_indices) > 0:
            last_nonzero_index = nonzero_indices[-1]
            
            output_tensor[row, last_nonzero_index+1:] = INT_MAX

    # 遍历每一行
    for i in range(output_tensor.shape[0]):
        row = output_tensor[i]
        nonzero_indices = torch.nonzero(row)
        if len(nonzero_indices) > 0:
            first_nonzero_idx = nonzero_indices[0]
            first_nonzero_val = row[first_nonzero_idx]
            # 设置连续的零值为第一个非零值
            for j in range(first_nonzero_idx, 0,-1):
                if row[j] == 0:
                    row[j] = first_nonzero_val

    reverse_tensor=output_tensor.clone()
    for i in range(reverse_tensor.shape[0]):
        row=reverse_tensor[i]
        # 使用unique函数获取不同的值
        unique_values = torch.unique(row)
        # row[torch.where(row==0)]=INT_MAX
        for val in unique_values:
            if val==0:
                continue
                # row[torch.where(row==val)]=INT_MAX
            else:
                row[torch.where(row==val)]=unique_values[unique_values < val].max()
    reverse_tensor[:,0]=INT_MAX
    
    return output_tensor,reverse_tensor



def exists(val):
    return val is not None


def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, T, n1, D)
            latent (torch.Tensor): latent features
                shape (b, T, n2, D)
        """
        x = self.norm_media(x)
        latents = self.norm_latents(latents)

        h = self.heads

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        q, k, v = rearrange_many((q, k, v), "b t n (h d) -> b h t n d", h=h)
        q = q * self.scale

        # attention
        sim = einsum("... i d, ... j d  -> ... i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h t n d -> b t n (h d)", h=h)
        return self.to_out(out)


class PerceiverResampler(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=6,
        dim_head=64,
        heads=8,
        num_latents=64,
        max_num_media=None,
        max_num_frames=None,
        ff_mult=4,
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.frame_embs = (
            nn.Parameter(torch.randn(max_num_frames, dim))
            if exists(max_num_frames)
            else None
        )
        self.media_time_embs = (
            nn.Parameter(torch.randn(max_num_media, 1, dim))
            if exists(max_num_media)
            else None
        )

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, T, F, v, D)
        Returns:
            shape (b, T, n, D) where n is self.num_latents
        """
        b, T, F, v = x.shape[:4]

        # frame and media time embeddings
        if exists(self.frame_embs):
            frame_embs = repeat(self.frame_embs[:F], "F d -> b T F v d", b=b, T=T, v=v)
            x = x + frame_embs
        x = rearrange(
            x, "b T F v d -> b T (F v) d"
        )  # flatten the frame and spatial dimensions
        if exists(self.media_time_embs):
            x = x + self.media_time_embs[:T]

        # blocks
        latents = repeat(self.latents, "n d -> b T n d", b=b, T=T)
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
        return self.norm(latents)


# gated cross attention
class MaskedCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_visual,
        dim_head=64,
        heads=8,
        only_attend_immediate_media=False,
    ):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim_visual, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        # whether for text to only attend to immediate preceding image, or all previous images
        self.only_attend_immediate_media = only_attend_immediate_media

    def forward(self, x, media, media_locations=None, chunk_locations=None, use_cached_media=False):
        """
        Args:
            x (torch.Tensor): text features
                shape (B, T_txt, D_txt)
            media (torch.Tensor): image features (actually the perceiver latents)
                shape (B, T_img, n, D_img) where n is the dim of the latents
            media_locations: boolean mask identifying the media tokens in x
                shape (B, T_txt)
            use_cached_media: bool
                If true, treat all of x as if they occur after the last media
                registered in media_locations. T_txt does not need to exactly
                equal media_locations.shape[1] in this case
        """

        if not use_cached_media:
            assert (
                media_locations.shape[1] == x.shape[1]
            ), f"media_location.shape is {media_locations.shape} but x.shape is {x.shape}"
        # num_beam = 3 during inference, so the batch size =3 during inference
        # x.shape=[batchsize, text_length, hidden_dim]=[3,22,1024]
        # num_latents derives from the Perceiver module, where num_latents embeddings are perceived from each media 
        # media.shape=[batchsize, num_images, num_latents, hidden_dim]=[3,3,64,1024]
        T_txt = x.shape[1]
        print(f"Length of Text {T_txt}")
        _, T_img, n = media.shape[:3]
        h = self.heads

        x = self.norm(x)
        # x.shape=[3,22,512]
        # q.shape=[3,22,1024]
        q = self.to_q(x)
        # num_images*num_latents=192
        # media.shape=[3,192,1024]
        media = rearrange(media, "b t n d -> b (t n) d")
        # k/v.shape=[3, 192, 512]
        k, v = self.to_kv(media).chunk(2, dim=-1)
        # q.shape=[3, 8, 22, 64]
        # k/v.shape=[3, 8, 192, 64]
        # multi-head attention
        q, k, v = rearrange_many((q, k, v), "b n (h d) -> b h n d", h=h)
        
        q = q * self.scale
        # sim.shape=[3,8,22,192]=[3, num_head, num_text_token, channel*num_image_patch]
        # num_head=8
        sim = einsum("... i d, ... j d -> ... i j", q, k)

        if exists(media_locations):
            # T_img=3, media_time=[1,2,3]
            media_time = torch.arange(T_img, device=x.device) + 1

            if use_cached_media:
                # text time is set to the last cached media location
                text_time = repeat(
                    torch.count_nonzero(media_locations, dim=1),
                    "b -> b i",
                    i=T_txt,
                )
                mask_op = torch.eq
                text_to_media_mask = mask_op(
                    rearrange(text_time, "b i -> b 1 i 1"),
                    repeat(media_time, "j -> 1 1 1 (j n)", n=n),
                )
                # chunk_tensor=torch.mul(chunk_locations,text_time)
                # chunk_mask, inverse_mask=chunk2mask(chunk_tensor)
            else:
                # at each boolean of True, increment the time counter (relative to media time)
                # text_time.shape=[3, 22]
                # 生成了3行一模一样的[0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3]
                
                # x_chunk,y_chunk=torch.where(chunk_locations==True)
                text_time = media_locations.cumsum(dim=-1)
                # chunk_time=chunk_locations.cumsum(dim=-1)
                chunk_tensor=torch.mul(chunk_locations,text_time)
                chunk_mask, inverse_mask=chunk2mask(chunk_tensor)

                # text time must equal media time if only attending to most immediate image
                # otherwise, as long as text time is greater than media time (if attending to all previous images / media)
                text_to_media_mask = torch.ge(
                    rearrange(chunk_mask, "b i -> b 1 i 1"),
                    rearrange(media_time, "j -> 1 1 1 j "),
                )
                text_to_media_inverse_mask = torch.lt(
                    rearrange(inverse_mask, "b i -> b 1 i 1"),
                    rearrange(media_time, "j -> 1 1 1 j "),
                )
                text_to_media_mask=torch.logical_and(text_to_media_mask,text_to_media_inverse_mask)
                text_to_media_mask=repeat(text_to_media_mask,"b 1 i j -> b 1 i (j n)", n=n)
            # # n: the number of perceiver latents
            # # text_to_media_mask.shape=[3,1,22,192]
            # mask_op = torch.eq if self.only_attend_immediate_media else torch.ge
            # text_to_media_mask = mask_op(
            #     rearrange(text_time, "b i -> b 1 i 1"),
            #     repeat(media_time, "j -> 1 1 1 (j n)", n=n),
            # )
            # sim.shape=[3,8,22,192]
            sim = sim.masked_fill(~text_to_media_mask, -torch.finfo(sim.dtype).max)

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        # if exists(media_locations) and self.only_attend_immediate_media:
        #     # any text without a preceding media needs to have attention zeroed out
        #     text_without_media_mask = text_time == 0
        #     text_without_media_mask = rearrange(
        #         text_without_media_mask, "b i -> b 1 i 1"
        #     )
        #     attn = attn.masked_fill(text_without_media_mask, 0.0)

        out = einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class GatedCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_visual,
        dim_head=64,
        heads=8,
        ff_mult=4,
        only_attend_immediate_media=True,
    ):
        super().__init__()
        self.attn = MaskedCrossAttention(
            dim=dim,
            dim_visual=dim_visual,
            dim_head=dim_head,
            heads=heads,
            only_attend_immediate_media=only_attend_immediate_media,
        )
        self.attn_gate = nn.Parameter(torch.tensor([0.0]))

        self.ff = FeedForward(dim, mult=ff_mult)
        self.ff_gate = nn.Parameter(torch.tensor([0.0]))

    def forward(
        self,
        x,
        media,
        media_locations=None,
        chunk_locations=None,
        use_cached_media=False,
    ):
        x = (
            self.attn(
                x,
                media,
                media_locations=media_locations,
                chunk_locations=chunk_locations,
                use_cached_media=use_cached_media,
            )
            * self.attn_gate.tanh()
            + x
        )
        x = self.ff(x) * self.ff_gate.tanh() + x

        return x