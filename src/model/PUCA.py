import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .NAFNet import NAFBlock
from ..util.util import pixel_shuffle_down_sampling, pixel_shuffle_up_sampling


class CentralMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH//2, kH//2] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)

class Downsample(nn.Module):
    def __init__(self, dilation):
        super().__init__()
        self.dilation=dilation

    def forward(self, x):
        B,C,H,W = x.shape
        x = rearrange(x, 'b c (hd h) (wd w) -> b (c hd wd) h w', h=self.dilation**2, w=self.dilation**2)
        x = rearrange(x, 'b c (hn hh) (wn ww) -> b c (hn wn) hh ww', hh=self.dilation, ww=self.dilation)
        x = rearrange(x, 'b (c hd wd) cc hh ww-> b (c cc) (hd hh) (wd ww)', hd=H//(self.dilation**2), wd=W//(self.dilation**2))
        return x
    
class Upsample(nn.Module):
    def __init__(self, dilation):
        super().__init__()
        self.dilation=dilation

    def forward(self, x):
        B,C,H,W = x.shape
        x = rearrange(x, 'b (c cc) (hd hh) (wd ww) -> b (c hd wd) cc hh ww', cc = self.dilation**2, hh=self.dilation, ww=self.dilation)
        x = rearrange(x, 'b c (hn wn) hh ww -> b c (hn hh) (wn ww)', hn=self.dilation, wn=self.dilation)
        x = rearrange(x, 'b (c hd wd) h w -> b c (hd h) (wd w)', hd=H//self.dilation, wd=W//self.dilation)
        return x

class PUCA(nn.Module):
    def __init__(self, img_channel, pd, dilation, width, 
                 enc_blk_nums, middle_blk_nums, dec_blk_nums):
        super().__init__()
        self.pd = pd
        self.dilation = dilation
        
        self.intro = nn.Conv2d(img_channel, width, kernel_size=1, stride=1)
        self.tail = nn.Sequential(nn.Conv2d(width,  width,    kernel_size=1),
                                  nn.Conv2d(width,    width//2, kernel_size=1),
                                  nn.Conv2d(width//2, width//2, kernel_size=1),
                                  nn.Conv2d(width//2, img_channel, kernel_size=1))
        
    
        self.masked_conv = nn.Sequential(CentralMaskedConv2d(width,width, kernel_size=2*dilation-1, stride=1, padding=dilation-1),
                                  nn.Conv2d(width, width, kernel_size=1, stride=1),
                                  nn.Conv2d(width, width, kernel_size=1, stride=1))
        self.final = nn.Conv2d(width, width, kernel_size=1, stride=1)
    
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan, dilation)for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Sequential(nn.Conv2d(chan, chan//2, kernel_size=1, stride=1),
                                Downsample(dilation)
                )
                )
            
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan, dilation) for _ in range(middle_blk_nums)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1),
                    Upsample(dilation)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan, dilation)for _ in range(num)]
                )
            )

    def forward(self, x, refine=False):
        x = self.intro(x)
        
        if self.training:
            pd=self.pd[0]
        elif refine:
            pd = self.pd[2]
        else:
            pd = self.pd[1]
        
        b, c, h, w = x.shape
        if pd>1:
            p = 0
            x = pixel_shuffle_down_sampling(x,pd,self.dilation)
        else:
            p= 2*self.dilation
            x = F.pad(x, (p,p,p,p), 'reflect')
        
        
        x = self.masked_conv(x)

        encs = []

        for i, (encoder, down) in enumerate(zip(self.encoders, self.downs)):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for i, (decoder, up, enc_skip) in enumerate(zip(self.decoders, self.ups, encs[::-1])):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)
            
        x = self.final(x)
        
        if pd>1:
            x = pixel_shuffle_up_sampling(x,pd, self.dilation)
       
        if p ==0:
            x = x
        else:
            x = x[:,:,p:-p,p:-p]

        x = self.tail(x)
        return x