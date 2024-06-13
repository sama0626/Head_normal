

import torch.nn as nn
import pytorch_lightning as pl

from ..geometry import index, orthogonal, perspective
class BasePIFuNet(pl.LightningModule):
    def __init__(
            self,
            projection_mode='orthogonal',
            error_term=nn.MSELoss(),
    ):

        super(BasePIFuNet, self).__init__()
        self.name = 'base'

        self.error_term = error_term

        self.index = index
        self.projection = orthogonal if projection_mode == 'orthogonal' else perspective

    def forward(self, points, images, calibs, transforms=None):

        features = self.filter(images)
        preds = self.query(features, points, calibs, transforms)
        return preds

    def filter(self, images):

        return None

    def query(self, features, points, calibs, transforms=None):

        return None

    def get_error(self, preds, labels):

        return self.error_term(preds, labels)
