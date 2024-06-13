from lib.model.BasePIFuNet import BasePIFuNet
from lib.model.HGFilters import *
from lib.net_util import init_net
from lib.net_util import VGGLoss
from lib.model.FBNet import define_G

class NormalNet(BasePIFuNet):
    def __init__(self, opt, error_term=nn.SmoothL1Loss()):

        super(NormalNet, self).__init__(error_term=error_term)
        self.l1_loss = nn.SmoothL1Loss()  # define a L1 LOSS ;
        self.opt = opt
        self.name= "pix2pixHD"
        if self.training:
            print('self.training is true ')
            self.vgg_loss = [VGGLoss()]

        self.netF = None
        if True:
            self.in_nmlF_dim = 3
            self.netF = define_G(self.in_nmlF_dim, 3, 64, "global", 4, 9, 1, 3,
                                 "instance")
        init_net(self)
    def filter(self, images):
        if self.netF is not None:
            self.nmlF = self.netF.forward(images)

        mask = (images.abs().sum(dim=1, keepdim=True) !=
                0.0).detach().float()
        self.nmlF = self.nmlF * mask

    def forward(self, input_data):

        self.filter(input_data['img'])
        error = self.get_norm_error(input_data['normal_F'])
        return self.nmlF, error


    def get_norm_error(self, tgt_F):

        l1_F_loss = self.l1_loss(self.nmlF, tgt_F)
        with torch.no_grad():
            vgg_F_loss = self.vgg_loss[0](self.nmlF, tgt_F)

        total_loss = 5.0 * l1_F_loss + vgg_F_loss
        return total_loss
