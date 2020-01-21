from torch.autograd import Variable
import torch.nn.functional as F
import torch
from models.network_utils.projection_layer import inverse_warp
from models.network_utils import networks
from models.network_utils.losses import ssim
from util.util import tensor2im
from collections import OrderedDict
import numpy as np
import itertools
import cv2
import os


class BaseModel():
    def name(self):
        return 'BaseModel'

    def __init__(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.load_dir = os.path.join(opt.checkpoints_dir, opt.name)

        if len(opt.gpu_ids) == 0:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:%d' % opt.gpu_ids[0])

        self.enc = networks.Encoder(input_nc=3, nz=opt.nz_geo * 3).to(self.device)
        self.dec = networks.Decoder(output_nc=1, nz=opt.nz_geo * 3).to(self.device)

        self.net_dict = {'enc': self.enc, 'dec': self.dec}
        param_list = []
        for name, model in self.net_dict.items():
            if not self.isTrain or opt.continue_train:
                self.load_network(model, name, opt.which_epoch)
            else:
                networks.init_weights(model, init_type=opt.init_type)

            if self.isTrain:  # and not name == 'net_exp':
                model.train()
                param_list.append(model.parameters())
            else:
                model.eval()
            networks.print_network(model)

        if self.isTrain:
            # define loss functions
            self.old_lr = opt.lr
            # initialize optimizers
            self.schedulers,self.optimizers = [],[]
            self.optimizer_G = torch.optim.Adam(itertools.chain(*param_list), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        if opt.category == 'kitti':
            intrinsics = np.array(
                [718.9, 0., 128, \
                 0., 718.9, 128, \
                 0., 0., 1.]).reshape((3, 3))
            self.depth_bias = 0
            self.depth_scale = 1
            self.depth_scale_vis = 250. / self.depth_scale
            self.depth_bias_vis = 0.
            self.intrinsics = torch.Tensor(intrinsics.astype(np.float32)).cuda().unsqueeze(0)

        elif opt.category in ['car','chair']:
            intrinsics = np.array([480, 0, 128,
                                   0, 480, 128,
                                   0, 0, 1]).reshape((3, 3))
            self.depth_bias, self.depth_scale = 2, 1.
            self.depth_scale_vis = 125. / self.depth_scale
            self.depth_bias_vis = self.depth_bias - self.depth_scale
            self.intrinsics = torch.Tensor(intrinsics).float().to(self.device).unsqueeze(0)


    def set_input(self, input):
        self.real_A = Variable(input['A'].to(self.device))
        self.real_B = Variable(input['B'].to(self.device))
        self.real_RT = Variable(input['RT'].to(self.device))

    def forward(self):
        self.z = self.encode(self.real_A)
        self.z_tf = self.transform(self.z,self.real_RT)
        self.depth = self.decode(self.z_tf)
        self.image = self.warp(self.real_A, self.depth, self.real_RT)

    def encode(self, image_tensor):
        return self.enc(image_tensor)

    def transform(self,z,RT):
        return networks.transform_code(z, self.opt.nz_geo, RT.inverse(), object_centric=self.opt.category in ['car', 'chair'])

    def decode(self,z):
        output = self.dec(z)
        if self.opt.category in ['kitti']:
            return 1 / (10 * torch.sigmoid(output) + 0.01) # predict disparity instead of depth for natural scenes
        elif self.opt.category in ['car', 'chair']:
            return torch.tanh(output) * self.depth_scale + self.depth_bias

    def warp(self, image, depth, RT):
        self.fake_B, self.flow, self.mask = inverse_warp(image, depth,RT,self.intrinsics)

    def backward_G(self):
        self.loss_G = F.l1_loss(self.fake_B,self.real_B) * self.opt.lambda_recon
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_visuals(self):
        dict = {}
        dict['real_A'] = tensor2im(self.real_A.data)
        dict['real_B'] = tensor2im(self.real_B.data)
        dict['fake_B'] = tensor2im(self.fake_B.data)
        dict['diff'] = tensor2im(0.5*(self.real_B.data[0,:,:,:] - self.fake_B.data[0,:,:,:]).abs())
        return OrderedDict(dict)

    def get_current_errors(self):
        return OrderedDict([('loss_G', self.loss_G.item())])

    def eval(self):
        self.forward()
        dict = {}
        dict['L1'] = F.l1_loss(self.fake_B, self.real_B).item()
        dict['SSIM'] = ssim((self.fake_B + 1) / 2, (self.real_B + 1) / 2).item()
        return OrderedDict(dict)

    def get_current_anim(self):
        self.switch_mode('eval')

        self.anim_dict = {'vis':[]}
        self.real_A = self.real_A[:1]
        self.real_B = self.real_B[:1]

        NV = 60
        for i in range(NV):
            pose = np.array([0, -(i-NV/2)*np.pi/180, 0, 0, 0, 0]) if self.opt.category in ['car','chair'] \
                else np.array([0, 0, 0,0, 0, i / 1000])
            self.real_RT = self.get_RT(pose)
            self.forward()
            self.anim_dict['vis'].append(tensor2im(self.fake_B.data))
        self.switch_mode('train')
        return self.anim_dict

    def get_high_res(self, image, pose, z=None):
        image_small = cv2.resize(image,(256,256))
        image_small = torch.from_numpy(image_small / 128. - 1).permute((2, 0, 1)).contiguous().unsqueeze(0)
        image_small = Variable(image_small).to(self.device).float()
        RT = self.get_RT(pose)

        z = self.enc(image_small) if z is None else z
        z_tf = self.transform(z, RT)
        depth = self.decode(z_tf)
        depth = F.upsample(depth, scale_factor=4, mode='bilinear')

        intrinsics = self.intrinsics[:1,:,:]*4
        intrinsics[0,2,2] = 1
        image = image / 128. - 1
        image = torch.from_numpy(image).permute((2, 0, 1)).contiguous().unsqueeze(0).to(self.device).float()
        image, _, _ = inverse_warp(image, depth, RT,intrinsics)
        image = tensor2im(image.data.detach())
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        return image

    def get_RT(self,pose1, pose2=np.zeros(6)):
        from scipy.spatial.transform import Rotation as ROT
        if self.opt.category in ['car','chair']:
            T = np.array([0, 0, 2]).reshape((3, 1))
            R = ROT.from_euler('xyz', pose1[:3]).as_dcm()
            T = -R.dot(T) + T
        else:
            R = ROT.from_euler('xyz', pose1[0:3]).as_dcm()
            T = pose1[3:].reshape((3, 1))
        mat = np.block(
            [[R, T],
             [np.zeros((1, 3)), 1]])

        return torch.Tensor(mat).float().to(self.device).unsqueeze(0)

    def switch_mode(self,mode):
        for name, model in self.net_dict.items():
            if mode == 'eval': model.eval()
            if mode == 'train': model.train()

    def save(self, label):
        for name,model in self.net_dict.items():
            self.save_network(model,name, label, self.gpu_ids)

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_ids, save_dir=None):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        if save_dir is None: save_dir = self.save_dir
        save_path = os.path.join(save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda(gpu_ids[0])

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label, load_dir=None):
        load_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        if load_dir is None: load_dir = self.load_dir
        load_path = os.path.join(self.load_dir, load_filename)
        network.load_state_dict(torch.load(load_path))

    # update learning rate (called once every epoch)
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)
