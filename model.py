import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, Bernoulli, kl_divergence as kl
import torch.nn.functional as F


Downsample = 1000
Downsample2 = 500
MID_LAYER = 100
FIRST_LAYER = 1000

def dice_loss(pred, target):
    pred = torch.sigmoid(pred)
    smooth = 1e-13
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()
    A_sum = torch.sum(iflat * iflat)
    B_sum = torch.sum(tflat * tflat)
    return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth) )


class DesnseEncoder(nn.Module):
    def __init__(self, input_dim, z_dim):
        super(DesnseEncoder, self).__init__()
        self.Encoder = nn.Sequential(
            nn.Linear(input_dim, FIRST_LAYER),
            nn.BatchNorm1d(num_features=FIRST_LAYER),
            nn.LeakyReLU(0.2, False),
            nn.Linear(FIRST_LAYER, MID_LAYER),
            nn.BatchNorm1d(num_features=MID_LAYER),
            nn.LeakyReLU(0.2, False),
        )
        self.mean_enc = nn.Linear(MID_LAYER, z_dim)
        self.var_enc = nn.Linear(MID_LAYER, z_dim)

    def forward(self, x):
        out = self.Encoder(x)
        mean = self.mean_enc(out)
        log_var = self.var_enc(out)
        return mean, log_var


class VAE2(nn.Module):
    '''Conditional VAE
    '''
    def __init__(self, input_dim, z_dim, batch=False):
        super(VAE2, self).__init__()
        if batch:
            c = 2
        else:
            c = 1
        self.Encoder = DesnseEncoder(input_dim, z_dim)
        self.Decoder = nn.Sequential(
            nn.Linear(z_dim+c, MID_LAYER),
            nn.BatchNorm1d(num_features=MID_LAYER),
            nn.LeakyReLU(0.2, True),
            nn.Linear(MID_LAYER, FIRST_LAYER),
            nn.BatchNorm1d(num_features=FIRST_LAYER),
            nn.LeakyReLU(0.2, True),
            nn.Linear(FIRST_LAYER, input_dim),
        )
    
    def forward(self, x, l, b=None, no_rec=False):
        if no_rec:
            mean, log_var = self.Encoder(x)
            return mean, log_var
        else:
            if b is not None:
                mean, log_var = self.Encoder(x)
                z = Normal(mean, torch.exp(log_var)).rsample()
                z_c = torch.cat((z, l, b), 1)
                rec = self.Decoder(z_c)
                return mean, log_var, z, rec
            else:
                mean, log_var = self.Encoder(x)
                z = Normal(mean, torch.exp(log_var)).rsample()
                z_c = torch.cat((z, l), 1)
                rec = self.Decoder(z_c)
                return mean, log_var, z, rec


class VAEInv(nn.Module):
    def __init__(self, vae):
        super(VAEInv, self).__init__()
        self.vae = vae

    def forward(self, x, l, b=None, no_rec=False):
        if no_rec:
            mean, log_var = self.vae.forward(x, l, no_rec=True)
            return mean, log_var
        else:
            if b is not None:
                mean, log_var, z, rec = self.vae(x, l, b)
            else:
                mean, log_var, z, rec = self.vae(x, l)
            return mean, log_var, z, rec


class VAE(nn.Module):
    '''Standard VAE class
    '''
    def __init__(self, input_dim, z_dim):
        super(VAE, self).__init__()
        self.Encoder = nn.Sequential(
            nn.Linear(input_dim, 1000),
            nn.BatchNorm1d(num_features=1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 100),
            nn.BatchNorm1d(num_features=100),
            nn.LeakyReLU(0.2, True),
        )
        self.mean_enc = nn.Linear(100, z_dim)
        self.var_enc = nn.Linear(100, z_dim)
        self.Decoder = nn.Sequential(
            nn.Linear(z_dim, 100),
            nn.BatchNorm1d(num_features=100),
            nn.LeakyReLU(0.2, True),
            nn.Linear(100, 1000),
            nn.BatchNorm1d(num_features=1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, input_dim),
        )
    
    def forward(self, x, no_rec=False):
        if no_rec:
            out = self.Encoder(x)
            mean = self.mean_enc(out)
            log_var = self.var_enc(out)
            return mean, log_var
        else:
            out = self.Encoder(x)
            mean = self.mean_enc(out)
            log_var = self.var_enc(out)
            z = Normal(mean, torch.exp(log_var)).rsample()
            rec = self.Decoder(z)
            return mean, log_var, z, rec


class VAEconv2(nn.Module):
    '''Standard VAE class
    '''
    def __init__(self, input_dim, z_dim):
        super(VAEconv2, self).__init__()
        self.ConvEncoder = nn.Sequential(
            ConvBlock(1, 64, 201),
            nn.MaxPool1d(25, 25),
            ConvBlock(64, 128, 17),
            nn.MaxPool1d(5, 5),
            ConvBlock(128, 256, 7),
            nn.MaxPool1d(4, 4),
            nn.Conv1d(256, 1, kernel_size=1, bias=True),
            nn.BatchNorm1d(1),
            nn.LeakyReLU(inplace=True)
        )
        self.mean_enc = nn.Sequential(
            nn.Linear(int(input_dim/Downsample2), z_dim)
        )
        self.var_enc = nn.Sequential(
            nn.Linear(int(input_dim/Downsample2), z_dim)
        )
        self.LinearDec = nn.Sequential(
            nn.Linear(z_dim, int(input_dim/Downsample2)),
            nn.BatchNorm1d(num_features=int(input_dim/Downsample2)),
            nn.LeakyReLU(0.2, True)
        )
        self.ConvDecoder = nn.Sequential(
            TPConv(1, 256, 4, 7),
            TPConv(256, 128, 5, 17),
            TPConv(128, 64, 25, 201),
            nn.Conv1d(64, 1, kernel_size=201, padding=100, bias=True)
        )
    
    def forward(self, x, no_rec=False):
        out = x.unsqueeze(1)
        out = self.ConvEncoder(out)
        out = out.squeeze(1)
        # out = self.LinearEncoder(out)
        mean = self.mean_enc(out)
        log_var = self.var_enc(out)
        if no_rec:
            return mean, log_var
        else:
            z = Normal(mean, torch.exp(log_var).sqrt()).rsample()
            rec = self.LinearDec(z)
            rec = rec.unsqueeze(1)
            rec = self.ConvDecoder(rec)
            rec = rec.squeeze(1)
            return mean, log_var, z, rec


SUB_1 = 64
SUB_2 = 32

class SplitEnc(nn.Module):
    ''' Addapted from https://github.com/wukevin/babel/blob/main/babel/models/autoencoders.py
    '''
    def __init__(self, input_dim, z_dim):
        super(SplitEnc, self).__init__()
        self.input_dim = input_dim
        self.split_layer = nn.ModuleList()
        for n in self.input_dim:
            # assert isinstance(n, int)
            layer1 = nn.Linear(n, SUB_1)
            nn.init.xavier_uniform_(layer1.weight)
            bn1 = nn.BatchNorm1d(SUB_1)
            act1 = nn.PReLU()
            layer2 = nn.Linear(SUB_1, SUB_2)
            nn.init.xavier_uniform_(layer2.weight)
            bn2 = nn.BatchNorm1d(SUB_2)
            act2 = nn.PReLU()
            self.split_layer.append(
                nn.ModuleList([layer1, bn1, act1, layer2, bn2, act2])
            )
        self.mean_enc = nn.Linear(SUB_2*len(input_dim), z_dim)
        self.var_enc = nn.Linear(SUB_2*len(input_dim), z_dim)
        nn.init.xavier_uniform_(self.mean_enc.weight)
        nn.init.xavier_uniform_(self.var_enc.weight)

    def forward(self, x):
        xs = torch.split(x, self.input_dim, dim=1)
        assert len(xs) == len(self.input_dim)
        enc_chroms = []
        for init_mod, chrom_input in zip(self.split_layer, xs):
            for f in init_mod:
                chrom_input = f(chrom_input)
            enc_chroms.append(chrom_input)
        enc1 = torch.cat(enc_chroms, dim=1)
        mean = self.mean_enc(enc1)
        log_var = self.var_enc(enc1)
        return mean, log_var


class SplitDec(nn.Module):
    ''' Addapted from https://github.com/wukevin/babel/blob/main/babel/models/autoencoders.py
    '''
    def __init__(self, input_dim, z_dim):
        super(SplitDec, self).__init__()
        self.input_dim = input_dim
        self.dec1 = nn.Linear(z_dim, len(self.input_dim) * SUB_2)
        self.bn1 = nn.BatchNorm1d(len(self.input_dim) * SUB_2)
        self.act1 = nn.PReLU()
        self.split_layer = nn.ModuleList()
        for n in self.input_dim:
            # assert isinstance(n, int)
            layer1 = nn.Linear(SUB_2, SUB_1)
            nn.init.xavier_uniform_(layer1.weight)
            bn1 = nn.BatchNorm1d(SUB_1)
            act1 = nn.PReLU()
            layer2 = nn.Linear(SUB_1, n)
            nn.init.xavier_uniform_(layer2.weight)
            self.split_layer.append(
                nn.ModuleList([layer1, bn1, act1, layer2])
            )

    def forward(self, x):
        x = self.act1(self.bn1(self.dec1(x)))
        xs = torch.chunk(x, chunks=len(self.input_dim), dim=1)
        rec_chroms = []
        for init_mod, chrom_input in zip(self.split_layer, xs):
            for f in init_mod:
                chrom_input = f(chrom_input)
            rec_chroms.append(chrom_input)
        rec = torch.cat(rec_chroms, dim=1)
        return rec


class VAESplit(nn.Module):
    '''Conditional VAE
    '''
    def __init__(self, input_dim, z_dim, batch=False):
        super(VAESplit, self).__init__()
        if batch:
            c = 2
        else:
            c = 1
        self.input_dim = input_dim
        self.Encoder = SplitEnc(input_dim, z_dim)
        self.Decoder = SplitDec(input_dim, z_dim+c)
    
    def forward(self, x, l, b=None, no_rec=False):
        if no_rec:
            mean, log_var = self.Encoder(x)
            return mean, log_var
        else:
            if b is not None:
                mean, log_var = self.Encoder(x)
                z = Normal(mean, torch.exp(log_var)).rsample()
                z_c = torch.cat((z, l, b), 1)
                rec = self.Decoder(z_c)
                return mean, log_var, z, rec
            else:
                mean, log_var = self.Encoder(x)
                z = Normal(mean, torch.exp(log_var)).rsample()
                z_c = torch.cat((z, l), 1)
                rec = self.Decoder(z_c)
                return mean, log_var, z, rec