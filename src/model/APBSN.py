import torch
import torch.nn as nn

from . import regist_model
from .PUCA import PUCA


@regist_model
class APBSN(nn.Module):
    '''
    Asymmetric PD Blind-Spot Network (AP-BSN)
    '''
    def __init__(self, bsn, pd, dilation, width, 
                 enc_blk_nums, middle_blk_nums, dec_blk_nums,
                 R3=True, R3_T=8, R3_p=0.16):
        '''
        Args:
            bsn            : blind-spot network type
            pd             : 'PD stride factor' during training and inference
            dilation       : stride factor of bsn's dilated conv
            R3             : flag of 'Random Replacing Refinement'
            R3_T           : number of masks for R3
            R3_p           : probability of R3
            enc_blk_nums   : number of bsn encoder module
            middle_blk_nums: number of bsn mid-level module
            dec_blk_nums   : number of bsn decoder module
        '''
        super().__init__()

        # network hyper-parameters
        self.R3 = R3
        self.R3_T = R3_T
        self.R3_p = R3_p

        # define network
        if bsn == 'PUCA':
            self.bsn = PUCA(3, pd, dilation, width, 
                 enc_blk_nums, middle_blk_nums, dec_blk_nums)
        else:
            raise NotImplementedError('bsn %s is not implemented'%bsn)

    def forward(self, x):
        return self.bsn(x)
    
    def denoise(self, x):
        '''
        Denoising process for inference.
        '''
        # forward PD-BSN process with inference pd factor
        img_pd_bsn = self.forward(x)

        # Random Replacing Refinement
        if not self.R3:
            ''' Directly return the result (w/o R3) '''
            return img_pd_bsn
        else:
            denoised = torch.empty(*(x.shape), self.R3_T, device=x.device)
            for t in range(self.R3_T):
                indice = torch.rand_like(x)
                mask = indice < self.R3_p

                tmp_input = torch.clone(img_pd_bsn).detach()
                tmp_input[mask] = x[mask]
                denoised[..., t] = self.bsn(tmp_input, refine=True)
                
            return torch.mean(denoised, dim=-1)