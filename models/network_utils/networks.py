import functools
from torch.optim import lr_scheduler
from torch import nn


class Encoder(nn.Module):
    def __init__(self, input_nc, nf=64, n_layers=8, nz=200, dropout=False):
        super(Encoder, self).__init__()
        norm_layer = get_norm_layer(norm_type='batch')
        nl_layer = get_non_linearity(layer_type='lrelu')
        kw, padw = 4, 1

        enc = [nn.Conv2d(input_nc, nf, kernel_size=kw, stride=2, padding=padw), nl_layer()]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 4)
            enc += [
                nn.Conv2d(nf * nf_mult_prev, nf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw)]
            if norm_layer is not None and n < n_layers - 1:
                enc += [norm_layer(nf * nf_mult)]
                enc += [nl_layer()]

        self.enc = nn.Sequential(*enc)

        fc = [nn.Linear(nf * nf_mult, nz)]
        if dropout: fc += [nn.Dropout(0.3)]
        self.fc = nn.Sequential(*fc)

    def forward(self, x):
        return self.fc(self.enc(x).view(x.size(0),-1) )


class Decoder(nn.Module):
    def __init__(self, output_nc, nf=64, n_layers=8, n_bilinear_layers=6, nz=200,dropout=False):
        super(Decoder, self).__init__()

        norm_layer = get_norm_layer(norm_type='batch')
        nl_layer = get_non_linearity(layer_type='lrelu')

        nf_mult = 4
        fc = [nn.Linear(nz, nf * nf_mult)]
        if dropout: fc += [nn.Dropout(0.3)]
        fc += [nl_layer()]
        self.fc = nn.Sequential(*fc)

        deconv = []

        for n in range(1, n_layers+1):

            nf_mult_prev = nf_mult
            nf_mult = min(2 ** (n_layers - n - 1), 4)

            upsample = 'bilinear' if n_layers - n < n_bilinear_layers else 'basic'
            deconv += upsampleLayer( int(nf * nf_mult_prev), int(nf * nf_mult), upsample=upsample)
            deconv += [norm_layer( int(nf * nf_mult) ),nl_layer()]

        deconv += [nn.Conv2d( int(nf * nf_mult), output_nc, kernel_size=3, padding=1)]

        self.nf = nf

        self.deconv = nn.Sequential(*deconv)

    def forward(self, z):
        feat_out = self.fc(z).view(z.size(0),4*self.nf,1,1)
        out = self.deconv(feat_out)
        return out

def transform_code(z, nz, RT,object_centric=False):
    z_tf = z.view(-1,nz,3).bmm(RT[:,:3,:3])
    if not object_centric:
        z_tf = z_tf + RT[:,:3,3].unsqueeze(1).expand((-1,nz,3))
    return z_tf.view(-1, nz * 3)



def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, gain)
            nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def get_non_linearity(layer_type='relu'):
    if layer_type == 'relu':
        nl_layer = functools.partial(nn.ReLU, inplace=True)
    elif layer_type == 'lrelu':
        nl_layer = functools.partial(nn.LeakyReLU, negative_slope=0.2, inplace=True)
    elif layer_type == 'elu':
        nl_layer = functools.partial(nn.ELU, inplace=True)
    else:
        raise NotImplementedError('nonlinearity activitation [%s] is not found' % layer_type)
    return nl_layer

def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def upsampleLayer(inplanes, outplanes, upsample='basic'):
    if upsample == 'basic':
        upconv = [nn.ConvTranspose2d(inplanes, outplanes, kernel_size=4, stride=2, padding=1)]
    elif upsample == 'bilinear':
        upconv = [nn.Upsample(scale_factor=2, mode='bilinear'),
                  nn.ReflectionPad2d((1,1,1,1)),
                  nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=0)]
    else:
        raise NotImplementedError('upsample layer [%s] not implemented' % upsample)
    return upconv

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)