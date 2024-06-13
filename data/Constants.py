from __future__ import division, absolute_import, print_function


class Constants:
    def __init__(self):
        self.dim_w = 128
        self.dim_h = 192
        self.hb_ratio = self.dim_h/self.dim_w
        self.real_h = 1.0
        self.real_w = self.real_h /self.dim_h * self.dim_w
        self.voxel_size = self.real_h/self.dim_h # 1./192.
        self.tau = 0.5
        self.K = 100
        self.fill = True
        self.constBackground = 4294967295

        self.h_normalize_half = self.real_h / 2.
        self.meshNormMargin = 0.15
        self.threshH = self.h_normalize_half * (1-self.meshNormMargin)
        self.threshWD = self.h_normalize_half * self.dim_w/self.dim_h * (1-self.meshNormMargin)


        self.lamb_sil = 0.02
        self.lamb_dis = 0.001
        self.lamb_nml_rf = 0.1

        self.black_list_images = ["092908", "092909", "092910", "092911",
                                  "090844", "090845", "090846", "090847"
                                 ]

        self.voxelizer_path = "./voxelizer/build/bin"

consts = Constants()