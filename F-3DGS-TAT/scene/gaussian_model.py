#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from pytorch3d.ops import knn_points
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.sh_utils import RGB2SH, SH2RGB, eval_sh
import torch.nn.functional as F

import tinycudann as tcnn

def create_mask():
    # 创建一个 5x5x5 的全 False mask
    mask = torch.zeros(5, 5, 5, dtype=torch.bool)

    # 通过遍历设置 True 和 False 使其交替出现
    for i in range(5):
        for j in range(5):
            for k in range(5):
                mask[i, j, k] = (i + j + k) % 2 == 0

    return mask

def compute_appfeature_VM(app_line, app_plane, xyz_sampled):

    # plane + line basis
    matMode = [[0,1], [0,2], [1,2]]
    vecMode = [2, 1, 0]
    coordinate_plane = torch.stack((xyz_sampled[..., matMode[0]], xyz_sampled[..., matMode[1]], xyz_sampled[..., matMode[2]])).detach().view(3, -1, 1, 2)
    coordinate_line = torch.stack((xyz_sampled[..., vecMode[0]], xyz_sampled[..., vecMode[1]], xyz_sampled[..., vecMode[2]]))
    coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

    plane_coef_point,line_coef_point = [],[]
    for idx_plane in range(len(app_plane)):
        plane_coef_point.append(F.grid_sample(app_plane[idx_plane], coordinate_plane[[idx_plane]],
                                            align_corners=True).view(-1, *xyz_sampled.shape[:1]))
        line_coef_point.append(F.grid_sample(app_line[idx_plane], coordinate_line[[idx_plane]],
                                        align_corners=True).view(-1, *xyz_sampled.shape[:1]))
    plane_coef_point, line_coef_point = torch.cat(plane_coef_point), torch.cat(line_coef_point)


    return (plane_coef_point * line_coef_point).T

def compute_appfeature_CP(app_line_0, app_line_1, app_line_2 , xyz_sampled):

    vecMode =  [2, 1, 0]
    coordinate_line = torch.stack(
        (xyz_sampled[..., vecMode[0]], xyz_sampled[..., vecMode[1]], xyz_sampled[..., vecMode[2]]))
    coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)


    line_coef_point = F.grid_sample(app_line_0, coordinate_line[[0]],
                                        align_corners=True).view(-1, *xyz_sampled.shape[:1])
    line_coef_point = line_coef_point * F.grid_sample(app_line_1, coordinate_line[[1]],
                                                        align_corners=True).view(-1, *xyz_sampled.shape[:1])
    line_coef_point = line_coef_point * F.grid_sample(app_line_2, coordinate_line[[2]],
                                                        align_corners=True).view(-1, *xyz_sampled.shape[:1])

    return line_coef_point.T

def sample_along_axis(vox_interval, hist, hist_int, edges, size):
    sample_x = np.empty(0)
    sample_y = np.empty(0)
    sample_z = np.empty(0)
    for i in range(vox_interval):
        for j in range(vox_interval):
            for k in range(vox_interval):
                if hist[i][j][k] > 5: # np.power(size - 1, 3): # > 64 and hist[i][j][k] < 125:
                    sample_x = np.concatenate((sample_x, np.random.uniform(edges[0][i], edges[0][i+1], size * (hist_int[i][j][k] + 1))))
                    sample_y = np.concatenate((sample_y, np.random.uniform(edges[1][j], edges[1][j+1], size * (hist_int[i][j][k] + 1))))
                    sample_z = np.concatenate((sample_z, np.random.uniform(edges[2][k], edges[2][k+1], size * (hist_int[i][j][k] + 1))))
    return sample_x, sample_y, sample_z

def sample_along_axis_small(vox_interval, hist, edges, size):
    sample_x = np.empty(0)
    sample_y = np.empty(0)
    sample_z = np.empty(0)
    for i in range(vox_interval):
        for j in range(vox_interval):
            for k in range(vox_interval):
                if hist[i][j][k] < 0 : #> 5 and hist[i][j][k] <= 15: # np.power(size - 1, 3): # > 64 and hist[i][j][k] < 125:
                    sample_x = np.concatenate((sample_x, np.random.uniform(edges[0][i], edges[0][i+1], size)))
                    sample_y = np.concatenate((sample_y, np.random.uniform(edges[1][j], edges[1][j+1], size)))
                    sample_z = np.concatenate((sample_z, np.random.uniform(edges[2][k], edges[2][k+1], size)))
    return sample_x, sample_y, sample_z

def generate_point_cloud(x, y, z, densify, size, mask):
    gen_point_cloud = torch.stack((x.view(densify, size, 1, 1).repeat(1, 1, size, size), 
                               y.view(densify, 1, size, 1).repeat(1, size, 1, size), 
                               z.view(densify, 1, 1, size).repeat(1, size, size, 1)), dim=-1).view(-1, 3)
    return gen_point_cloud

def generate_point_cloud_3(x, y, z, densify, size):
    gen_point_cloud = torch.stack((x.view(densify, size, 1, 1).repeat(1, 1, size, size), 
                               y.view(densify, 1, size, 1).repeat(1, size, 1, size), 
                               z.view(densify, 1, 1, size).repeat(1, size, size, 1)), dim=-1).view(-1, 3)
    return gen_point_cloud

def factorize_CP(x, y, z, densify, size, comp):
    gen_features_x = x.view(densify, comp, size, 1, 1).repeat(1, 1, 1, size, size)
    gen_features_y = y.view(densify, comp, 1, size, 1).repeat(1, 1, size, 1, size)
    gen_features_z = z.view(densify, comp, 1, 1, size).repeat(1, 1, size, size, 1)
    gen_features = gen_features_x * gen_features_y * gen_features_z
    return gen_features

def factorize_VM(vec_x, vec_y, vec_z, plane_x, plane_y, plane_z, densify, size, comp):
    vec_x = vec_x.view(densify, comp, size, 1, 1).repeat(1, 1, 1, size, size)
    vec_y = vec_y.view(densify, comp, 1, size, 1).repeat(1, 1, size, 1, size)
    vec_z = vec_z.view(densify, comp, 1, 1, size).repeat(1, 1, size, size, 1)
    plane_x = plane_x.view(densify, comp, 1, size, size).repeat(1, 1, size, 1, 1)
    plane_y = plane_y.view(densify, comp, size, 1, size).repeat(1, 1, 1, size, 1)
    plane_z = plane_z.view(densify, comp, size, size, 1).repeat(1, 1, 1, 1, size)
    gen_features = torch.concat((vec_x * plane_x, vec_y * plane_y, vec_z * plane_z), dim = 1)
    return gen_features


class GaussianModel:

    def setup_functions(self, training_args):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

        self.comp = training_args.comp
        # self.resolution = 50
        self.scale = 0.1
        self.size_3 = 3
        self.size = training_args.size
        self.vox_interval = training_args.vox_interval

        # self.feas_layer_1 = nn.Linear(self.comp, 128).cuda() # 288
        # self.feas_layer_2 = nn.Linear(128, 49).cuda()

        self.feas_layer = tcnn.Network(
                n_input_dims=self.comp,
                n_output_dims=49,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 128,
                    "n_hidden_layers": 1,
                },
            )

    def __init__(self, sh_degree : int, training_args):
        self.active_sh_degree = 3
        self.max_sh_degree = sh_degree
        self._mask_fac = torch.empty(0)
        self._app_line_0 = torch.empty(0)
        self._app_line_1 = torch.empty(0)
        self._app_line_2 = torch.empty(0)
        self._app_plane_0 = torch.empty(0)
        self._app_plane_1 = torch.empty(0)
        self._app_plane_2 = torch.empty(0)
        self._app3_line_0 = torch.empty(0)
        self._app3_line_1 = torch.empty(0)
        self._app3_line_2 = torch.empty(0)
        self._app3_plane_0 = torch.empty(0)
        self._app3_plane_1 = torch.empty(0)
        self._app3_plane_2 = torch.empty(0)
        self._shs_dc_line_0 = torch.empty(0)
        self._shs_dc_line_1 = torch.empty(0)
        self._shs_dc_line_2 = torch.empty(0)
        self._shs_dc3_line_0 = torch.empty(0)
        self._shs_dc3_line_1 = torch.empty(0)
        self._shs_dc3_line_2 = torch.empty(0)
        self._rotat_line_0 = torch.empty(0)
        self._rotat_line_1 = torch.empty(0)
        self._rotat_line_2 = torch.empty(0)
        self._scale_line_0 = torch.empty(0)
        self._scale_line_1 = torch.empty(0)
        self._scale_line_2 = torch.empty(0)
        self.comp = torch.empty(0)
        self.scale = 0
        self.fixnum = 0
        self._x = torch.empty(0)
        self._y = torch.empty(0)
        self._z = torch.empty(0)
        self.size = torch.empty(0)
        self.densify = torch.empty(0)
        self._x3 = torch.empty(0)
        self._y3 = torch.empty(0)
        self._z3 = torch.empty(0)
        self.size_3 = torch.empty(0)
        self.densify_3 = torch.empty(0)
        self.fac_mask = torch.empty(0)
        self.vox_interval = torch.empty(0)
        self._features_dc_fac = torch.empty(0)
        self._features_rest_fac = torch.empty(0)
        self._scaling_fac = torch.empty(0)
        self._rotation_fac = torch.empty(0)
        self._opacity_fac = torch.empty(0)
        self._mask = torch.empty(0)
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions(training_args=training_args)

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        # # CP decomposition
        scale_CP = factorize_CP(self._scale_line_0, self._scale_line_1, self._scale_line_2, self.densify, self.size, 3)
        # print('rotat_CP', rotat_CP.shape)
        scale_CP = scale_CP.permute(0, 2, 3, 4, 1).reshape(self.densify * self.size * self.size * self.size, 3)
        return torch.concat((scale_CP, self.scaling_activation(self._scaling)), dim = 0)

        # return self.scaling_activation(torch.concat((self._scaling_fac, self._scaling), dim = 0))

    @property
    def get_rotation(self):
        rotat_CP = factorize_CP(self._rotat_line_0, self._rotat_line_1, self._rotat_line_2, self.densify, self.size, 4)
        # print('rotat_CP', rotat_CP.shape)
        rotat_CP = rotat_CP.permute(0, 2, 3, 4, 1).reshape(self.densify * self.size * self.size * self.size, 4)
        return self.rotation_activation(torch.concat((rotat_CP, self._rotation), dim = 0))

        # # CP decomposition
        # feas_CP = factorize_CP(self._shs_dc_line_0, self._shs_dc_line_1, self._shs_dc_line_2, self.densify, self.size, self.comp)
        # feas_CP_3 = factorize_CP(self._shs_dc3_line_0, self._shs_dc3_line_1, self._shs_dc3_line_2, self.densify_3, self.size_3, self.comp)
        # feas_CP = feas_CP.permute(0, 2, 3, 4, 1).reshape(self.densify*self.size*self.size*self.size, self.comp)
        # feas_CP_3 = feas_CP_3.permute(0, 2, 3, 4, 1).reshape(self.densify_3*self.size_3*self.size_3*self.size_3, self.comp)
        # feas = torch.concat((feas_CP, feas_CP_3))
        # feas = self.feas_layer_2(torch.relu(self.feas_layer_1(feas)))
        # return self.rotation_activation(torch.concat((feas[:, 48:52], self._rotation), dim = 0))
        
        # return self.rotation_activation(torch.concat((self._rotation_fac, self._rotation), dim = 0))

    @property
    def get_xyz(self):
        point_cloud = generate_point_cloud(self._x, self._y, self._z, self.densify, self.size, self.voxel_mask)
        point_cloud_3 = generate_point_cloud_3(self._x3, self._y3, self._z3, self.densify_3, self.size_3)
        return torch.concat((point_cloud, point_cloud_3, self._xyz), dim = 0)
        # return new_point_cloud
        # return self._xyz

    @property
    def get_features(self):
        # features_dc_fac = self._features_dc_fac
        # features_rest_fac = self._features_rest_fac
        features_dc = self._features_dc
        features_rest = self._features_rest
        # return torch.concat((torch.cat((features_dc_fac, features_rest_fac), dim=1),torch.cat((features_dc, features_rest), dim=1)), dim = 0)

        # CP decomposition
        feas_CP = factorize_CP(self._shs_dc_line_0, self._shs_dc_line_1, self._shs_dc_line_2, self.densify, self.size, self.comp)
        feas_CP_3 = factorize_CP(self._shs_dc3_line_0, self._shs_dc3_line_1, self._shs_dc3_line_2, self.densify_3, self.size_3, self.comp)
        feas_CP = feas_CP.permute(0, 2, 3, 4, 1).reshape(self.densify*self.size*self.size*self.size, self.comp)
        feas_CP_3 = feas_CP_3.permute(0, 2, 3, 4, 1).reshape(self.densify_3*self.size_3*self.size_3*self.size_3, self.comp)
        feas = torch.concat((feas_CP, feas_CP_3))
        # feas = self.feas_layer_2(torch.relu(self.feas_layer_1(feas)))
        feas = self.feas_layer(feas)
        feas = feas[:, :48].view(feas.shape[0], 16, 3)
        return torch.concat((feas, torch.cat((features_dc, features_rest), dim=1)), dim = 0)

        # # VM decomposition
        # feas_VM = factorize_VM(self._app_line_0, self._app_line_1, self._app_line_2, self._app_plane_0, self._app_plane_1, self._app_plane_2, self.densify, self.size, self.comp)
        # feas_VM_3 = factorize_VM(self._app3_line_0, self._app3_line_1, self._app3_line_2, self._app3_plane_0, self._app3_plane_1, self._app3_plane_2, self.densify_3, self.size_3, self.comp)
        # feas_VM = feas_VM.permute(0, 2, 3, 4, 1).reshape(self.densify*self.size*self.size*self.size, self.comp*3)
        # feas_VM_3 = feas_VM_3.permute(0, 2, 3, 4, 1).reshape(self.densify_3*self.size_3*self.size_3*self.size_3, self.comp*3)
        # feas = torch.concat((feas_VM, feas_VM_3))
        # feas = self.feas_layer_2(torch.relu(self.feas_layer_1(feas)))
        # feas = feas[:, :48].view(feas.shape[0], 16, 3)
        # return torch.concat((feas, torch.cat((features_dc, features_rest), dim=1)), dim = 0)

        # CP decompostion with interpolation
        # point_cloud = generate_point_cloud(self._x, self._y, self._z, self.densify, self.size)
        # point_cloud_norm = point_cloud / 1.3
        # feas = compute_appfeature_CP(self._app_line_0, self._app_line_1, self._app_line_2 , point_cloud_norm).view(point_cloud.shape[0], 16, 3)

        # # VM decomposition with interpolation
        # point_cloud = generate_point_cloud(self._x, self._y, self._z, self.densify, self.size)
        # point_cloud_norm = point_cloud / 1.3
        # app_line = []
        # app_line.append(self._app_line_0)
        # app_line.append(self._app_line_1)
        # app_line.append(self._app_line_2)
        # app_plane = []
        # app_plane.append(self._app_plane_0)
        # app_plane.append(self._app_plane_1)
        # app_plane.append(self._app_plane_2)
        # feas = compute_appfeature_VM(app_line, app_plane, point_cloud_norm) # view(point_cloud.shape[0], 16, 3)
        # feas = self.feas_layer_2(torch.relu(self.feas_layer_1(feas)))
        # feas = feas[:,:48].view(point_cloud.shape[0], 16, 3)
        # # print('feas', feas.type())
        # # print('feas', torch.concat((feas, torch.cat((features_dc, features_rest), dim=1)), dim = 0).shape)
        # # print('s', torch.concat((torch.cat((features_dc_fac, features_rest_fac), dim=1),torch.cat((features_dc, features_rest), dim=1)), dim = 0).shape)
        # return torch.concat((feas, torch.cat((features_dc, features_rest), dim=1)), dim = 0)
        # # return torch.cat((features_dc_fac, features_rest_fac), dim=1)
    
    @property
    def get_opacity(self):

        # return self.opacity_activation(torch.concat((self._opacity_fac, self._opacity), dim = 0))

        # CP decomposition
        feas_CP = factorize_CP(self._shs_dc_line_0, self._shs_dc_line_1, self._shs_dc_line_2, self.densify, self.size, self.comp)
        feas_CP_3 = factorize_CP(self._shs_dc3_line_0, self._shs_dc3_line_1, self._shs_dc3_line_2, self.densify_3, self.size_3, self.comp)
        feas_CP = feas_CP.permute(0, 2, 3, 4, 1).reshape(self.densify*self.size*self.size*self.size, self.comp)
        feas_CP_3 = feas_CP_3.permute(0, 2, 3, 4, 1).reshape(self.densify_3*self.size_3*self.size_3*self.size_3, self.comp)
        feas = torch.concat((feas_CP, feas_CP_3))
        # feas = self.feas_layer_2(torch.relu(self.feas_layer_1(feas)))
        feas = self.feas_layer(feas)
        opacity = feas[:, -1].squeeze().unsqueeze(1)
        return self.opacity_activation(torch.concat((opacity, self._opacity), dim = 0))

        # # VM decomposition
        # feas_VM = factorize_VM(self._app_line_0, self._app_line_1, self._app_line_2, self._app_plane_0, self._app_plane_1, self._app_plane_2, self.densify, self.size, self.comp)
        # feas_VM_3 = factorize_VM(self._app3_line_0, self._app3_line_1, self._app3_line_2, self._app3_plane_0, self._app3_plane_1, self._app3_plane_2, self.densify_3, self.size_3, self.comp)
        # feas_VM = feas_VM.permute(0, 2, 3, 4, 1).reshape(self.densify*self.size*self.size*self.size, self.comp*3)
        # feas_VM_3 = feas_VM_3.permute(0, 2, 3, 4, 1).reshape(self.densify_3*self.size_3*self.size_3*self.size_3, self.comp*3)
        # feas = torch.concat((feas_VM, feas_VM_3))
        # feas = self.feas_layer_2(torch.relu(self.feas_layer_1(feas)))
        # opacity = feas[:, -1].squeeze().unsqueeze(1)
        # # print(opacity.shape)
        # # print(self._opacity.shape)
        # return self.opacity_activation(torch.concat((opacity, self._opacity), dim = 0))

        # # VM decomposition with interpolation
        # point_cloud = generate_point_cloud(self._x, self._y, self._z, self.densify, self.size)
        # point_cloud_norm = point_cloud / 1.3
        # app_line = []
        # app_line.append(self._app_line_0)
        # app_line.append(self._app_line_1)
        # app_line.append(self._app_line_2)
        # app_plane = []
        # app_plane.append(self._app_plane_0)
        # app_plane.append(self._app_plane_1)
        # app_plane.append(self._app_plane_2)
        # feas = compute_appfeature_VM(app_line, app_plane, point_cloud_norm) # view(point_cloud.shape[0], 16, 3)
        # feas = self.feas_layer_2(torch.relu(self.feas_layer_1(feas)))
        # feas = feas[:, -1].unsqueeze(1)
        # return self.opacity_activation(torch.concat((feas, self._opacity), dim = 0))

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, hist_path):

        # 该处进行点云voxel化然后渲染
        path = hist_path
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        print('number of original 3d gaussian points :', xyz.shape)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])


        # make histogram and generate new coordinates
        point_cloud = xyz

        # min_value = (-1.3, -1.3, - 1.3)
        # max_value = (1.3, 1.3, 1.3)

        min_value = np.min(xyz, axis=0) # * 1.2
        max_value = np.max(xyz, axis=0) # * 1.2

        # 定义直方图参数
        x_bins = np.linspace(min_value[0], max_value[0], self.vox_interval + 1)
        y_bins = np.linspace(min_value[1], max_value[1], self.vox_interval + 1)
        z_bins = np.linspace(min_value[2], max_value[2], self.vox_interval + 1)

        # 使用 numpy.histogramdd 函数计算直方图
        hist, edges = np.histogramdd(point_cloud, bins=(x_bins, y_bins, z_bins))
        # print('hist', hist)
        # print('edges', edges)
        hist_int = (hist/(self.size*self.size*self.size)).astype(int) # (self.size*self.size*self.size)).astype(int)
        # hist_int_5 = (hist/(self.size_5*self.size_5*self.size_5)).astype(int)

        # # 找到每个点落在哪个箱子里
        # inds = np.vstack([np.digitize(point_cloud[:, i], edges[i]) - 1 for i in range(3)]).T

        # print('inds', inds.shape)

        # # 创建一个空的列表来保存直方图计数小于5的点
        # points_with_low_hist = []
        # points_index = []

        # # 遍历每个点，检查对应的直方图计数
        # for i, ind in enumerate(inds):
        #     if hist[tuple(ind)] <= 27:
        #         points_with_low_hist.append(point_cloud[i])
        #         points_index.append(i)

        # # 转换成NumPy数组
        # points_with_low_hist = np.array(points_with_low_hist)
        # points_index = torch.tensor(np.array(points_index))

        # print('normalshape', points_with_low_hist.shape)

        sample_x = np.empty(0)
        sample_y = np.empty(0)
        sample_z = np.empty(0)
        sample_x3 = np.empty(0)
        sample_y3 = np.empty(0)
        sample_z3 = np.empty(0)
        # 打印前几个体素的边界坐标
        self.voxel_mask = create_mask()
        sample_x, sample_y, sample_z = sample_along_axis(self.vox_interval, hist, hist_int, edges, self.size)
        sample_x3, sample_y3, sample_z3 = sample_along_axis_small(self.vox_interval, hist, edges, self.size_3)

        print('number of points along each axis : ', sample_x.shape[0])

        self.densify = int(sample_x.shape[0]/self.size)
        self.densify_3 = int(sample_x3.shape[0]/self.size_3)

        learn_point_cloud = generate_point_cloud(torch.tensor(sample_x, dtype=torch.float), torch.tensor(sample_y, dtype=torch.float), torch.tensor(sample_z, dtype=torch.float), self.densify, self.size, self.voxel_mask)
        learn_point_cloud_3 = generate_point_cloud_3(torch.tensor(sample_x3, dtype=torch.float), torch.tensor(sample_y3, dtype=torch.float), torch.tensor(sample_z3, dtype=torch.float), self.densify_3, self.size_3)
        learn_point_cloud = torch.concat((learn_point_cloud, learn_point_cloud_3))
        # self.fac_mask = torch.ones(learn_point_cloud.shape[0], dtype=torch.bool)
        # print('mask', self.fac_mask, self.fac_mask.shape)

        # 初始化histogram参数
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud_fac = learn_point_cloud.float().cuda()
        self.fixnum = learn_point_cloud.shape[0]
        hist_shs = np.random.random((learn_point_cloud.shape[0], 3)) / 255.0
        fused_color_fac = torch.tensor(np.asarray(hist_shs)).float().cuda()
        features_fac = torch.zeros((fused_color_fac.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features_fac[:, :3, 0 ] = fused_color_fac
        features_fac[:, 3:, 1:] = 0.0

        print("Number of coarse points at initialisation : ", fused_point_cloud_fac.shape[0])
        # dist2_fac = torch.clamp_min(distCUDA2(fused_point_cloud_fac), 0.0000001)
        # scales_fac = torch.log(torch.sqrt(dist2_fac))[...,None].repeat(1, 3)
        # print('scales', scales_fac.shape)
        rots_fac = torch.zeros((fused_point_cloud_fac.shape[0], 4), device="cuda")
        rots_fac[:, 0] = 1
        # print('rots', rots_fac.shape)

        opacities_fac = inverse_sigmoid(0.01 * torch.ones((fused_point_cloud_fac.shape[0], 1), dtype=torch.float, device="cuda"))
        # print('opacities', opacities_fac.shape)

        # 初始化random points的参数
        fused_point_cloud_xyz = np.empty((0, 3)) # pcd.points # np.empty((0, 3)) # points_with_low_hist
        # fused_point_cloud_xyz = point_cloud
        fused_point_cloud_shs = SH2RGB(np.random.random((fused_point_cloud_xyz.shape[0], 3)) / 255.0)
        fused_point_cloud = torch.tensor(np.asarray(fused_point_cloud_xyz)).float().cuda()
        # print('fused_point_cloud', fused_point_cloud.shape)
        fused_color = RGB2SH(torch.tensor(np.asarray(fused_point_cloud_shs)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of fine points at initialisation : ", fused_point_cloud.shape[0])
        # dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(fused_point_cloud_xyz)).float().cuda()), 0.0000001)
        # scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        # print('scales', scales.shape)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        # print('rots', rots.shape)

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        # print('opacities', opacities.shape)

        # scale initialization all
        fused_point_cloud_all = torch.concat((fused_point_cloud_fac, fused_point_cloud), dim=0)
        dist2_all = torch.clamp_min(distCUDA2(fused_point_cloud_all), 0.0000001)
        scales_all = torch.log(torch.sqrt(dist2_all))[...,None].repeat(1, 3)
        scales_fac = scales_all[:self.fixnum, :]
        scales = scales_all[self.fixnum:, :]
        print('scalesmin', torch.min(scales_fac))
        print('scalesmax', torch.max(scales_fac))

        # 创建一个张量来存储最近邻点的索引
        # print('features_dc', features_dc.shape)
        # print('features_extra', features_dc.shape)
        print('point_cloud', point_cloud.shape)
        features_dc = torch.tensor(features_dc, dtype=torch.float, device="cuda")
        features_extra = torch.tensor(features_extra, dtype=torch.float, device="cuda")
        _, indices, _ = knn_points(fused_point_cloud_fac.unsqueeze(0), torch.tensor(np.asarray(point_cloud)).float().cuda().unsqueeze(0))
        indices = indices.squeeze(0).squeeze(1)

        # factorize CP decomposition
        # size 3
        self._shs_dc3_line_0 = nn.Parameter(self.scale * 2 * torch.rand((self.densify_3, self.comp, self.size_3)).float().cuda().requires_grad_(True))
        self._shs_dc3_line_1 = nn.Parameter(self.scale * 2 * torch.rand((self.densify_3, self.comp, self.size_3)).float().cuda().requires_grad_(True))
        self._shs_dc3_line_2 = nn.Parameter(self.scale * 2 * torch.rand((self.densify_3, self.comp, self.size_3)).float().cuda().requires_grad_(True))
        # size 5
        self._shs_dc_line_0 = nn.Parameter(self.scale * 2 * torch.rand((self.densify, self.comp, self.size)).float().cuda().requires_grad_(True))
        self._shs_dc_line_1 = nn.Parameter(self.scale * 2 * torch.rand((self.densify, self.comp, self.size)).float().cuda().requires_grad_(True))
        self._shs_dc_line_2 = nn.Parameter(self.scale * 2 * torch.rand((self.densify, self.comp, self.size)).float().cuda().requires_grad_(True))

        # rotation
        self._rotat_line_0 = nn.Parameter(torch.ones((self.densify, 4, self.size)).float().cuda().requires_grad_(True))
        self._rotat_line_1 = nn.Parameter(torch.ones((self.densify, 4, self.size)).float().cuda().requires_grad_(True))
        self._rotat_line_2 = nn.Parameter(torch.ones((self.densify, 4, self.size)).float().cuda().requires_grad_(True))

        # scale
        self._scale_line_0 = nn.Parameter(self.scale * torch.ones((self.densify, 3, self.size)).float().cuda().requires_grad_(True))
        self._scale_line_1 = nn.Parameter(self.scale * torch.ones((self.densify, 3, self.size)).float().cuda().requires_grad_(True))
        self._scale_line_2 = nn.Parameter(self.scale * torch.ones((self.densify, 3, self.size)).float().cuda().requires_grad_(True))

        # factorize VM decomposition
        # size 3
        self._app3_line_0 = nn.Parameter(self.scale * 2 * torch.rand((self.densify_3, self.comp, self.size_3)).float().cuda().requires_grad_(True))
        self._app3_line_1 = nn.Parameter(self.scale * 2 * torch.rand((self.densify_3, self.comp, self.size_3)).float().cuda().requires_grad_(True))
        self._app3_line_2 = nn.Parameter(self.scale * 2 * torch.rand((self.densify_3, self.comp, self.size_3)).float().cuda().requires_grad_(True))
        self._app3_plane_0 = nn.Parameter(self.scale * 2 * torch.rand((self.densify_3, self.comp, self.size_3, self.size_3)).float().cuda().requires_grad_(True))
        self._app3_plane_1 = nn.Parameter(self.scale * 2 * torch.rand((self.densify_3, self.comp, self.size_3, self.size_3)).float().cuda().requires_grad_(True))
        self._app3_plane_2 = nn.Parameter(self.scale * 2 * torch.rand((self.densify_3, self.comp, self.size_3, self.size_3)).float().cuda().requires_grad_(True))

        # size 5
        self._app_line_0 = nn.Parameter(self.scale * 2 * torch.rand((self.densify, self.comp, self.size)).float().cuda().requires_grad_(True))
        self._app_line_1 = nn.Parameter(self.scale * 2 * torch.rand((self.densify, self.comp, self.size)).float().cuda().requires_grad_(True))
        self._app_line_2 = nn.Parameter(self.scale * 2 *  torch.rand((self.densify, self.comp, self.size)).float().cuda().requires_grad_(True))
        self._app_plane_0 = nn.Parameter(self.scale * 2 * torch.rand((self.densify, self.comp, self.size, self.size)).float().cuda().requires_grad_(True))
        self._app_plane_1 = nn.Parameter(self.scale * 2 * torch.rand((self.densify, self.comp, self.size, self.size)).float().cuda().requires_grad_(True))
        self._app_plane_2 = nn.Parameter(self.scale * 2 * torch.rand((self.densify, self.comp, self.size, self.size)).float().cuda().requires_grad_(True))

        self._x = nn.Parameter(torch.tensor(sample_x, dtype=torch.float, device="cuda").requires_grad_(True))
        self._y = nn.Parameter(torch.tensor(sample_y, dtype=torch.float, device="cuda").requires_grad_(True))
        self._z = nn.Parameter(torch.tensor(sample_z, dtype=torch.float, device="cuda").requires_grad_(True))
        self._x3 = nn.Parameter(torch.tensor(sample_x3, dtype=torch.float, device="cuda").requires_grad_(True))
        self._y3 = nn.Parameter(torch.tensor(sample_y3, dtype=torch.float, device="cuda").requires_grad_(True))
        self._z3 = nn.Parameter(torch.tensor(sample_z3, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc_fac = nn.Parameter(features_fac[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest_fac = nn.Parameter(features_fac[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        # self._features_dc_fac = nn.Parameter(features_dc[indices].transpose(1, 2).contiguous().requires_grad_(True))
        # self._features_rest_fac = nn.Parameter(features_extra[indices].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling_fac = nn.Parameter(scales_fac.requires_grad_(True))
        self._rotation_fac = nn.Parameter(rots_fac.requires_grad_(True))
        self._opacity_fac = nn.Parameter(opacities_fac.requires_grad_(True))
        self._mask_fac = nn.Parameter(torch.ones((scales_fac.shape[0], 1), device="cuda").requires_grad_(True))
        # self.max_radii2D_fac = torch.zeros((fused_point_cloud_fac.shape[0]), device="cuda")

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        # self._features_dc = nn.Parameter(features_dc[points_index].transpose(1, 2).contiguous().requires_grad_(True))
        # self._features_rest = nn.Parameter(features_extra[points_index].transpose(1, 2).contiguous().requires_grad_(True))
        # self._scaling = nn.Parameter(scales[points_index].requires_grad_(True))
        # self._rotation = nn.Parameter(rots[points_index].requires_grad_(True))
        # self._opacity = nn.Parameter(opacities[points_index].requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._x], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "x"},
            {'params': [self._y], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "y"},
            {'params': [self._z], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "z"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._features_dc_fac], 'lr': training_args.feature_lr, "name": "f_dc_fac"},
            {'params': [self._features_rest_fac], 'lr': training_args.feature_lr / 20.0, "name": "f_rest_fac"},
            {'params': [self._opacity_fac], 'lr': training_args.opacity_lr, "name": "opacity_fac"},
            {'params': [self._scaling_fac], 'lr': training_args.scaling_lr, "name": "scaling_fac"},
            {'params': [self._rotation_fac], 'lr': training_args.rotation_lr, "name": "rotation_fac"},
            {'params': [self._app_line_0], 'lr': training_args.tensor_lr_init, "name": "color_line0"},
            {'params': [self._app_line_1], 'lr': training_args.tensor_lr_init, "name": "color_line1"},
            {'params': [self._app_line_2], 'lr': training_args.tensor_lr_init, "name": "color_line2"},
            {'params': [self._app_plane_0], 'lr': training_args.tensor_lr_init, "name": "color_plane0"},
            {'params': [self._app_plane_1], 'lr': training_args.tensor_lr_init, "name": "color_plane1"},
            {'params': [self._app_plane_2], 'lr': training_args.tensor_lr_init, "name": "color_plane2"},
            {'params': [self._shs_dc_line_0], 'lr': training_args.tensor_lr_init, "name": "shs_dc_line0"},
            {'params': [self._shs_dc_line_1], 'lr': training_args.tensor_lr_init, "name": "shs_dc_line1"},
            {'params': [self._shs_dc_line_2], 'lr': training_args.tensor_lr_init, "name": "shs_dc_line2"},
            {'params': [self._shs_dc3_line_0], 'lr': training_args.tensor_lr_init, "name": "shs_dc3_line0"},
            {'params': [self._shs_dc3_line_1], 'lr': training_args.tensor_lr_init, "name": "shs_dc3_line1"},
            {'params': [self._shs_dc3_line_2], 'lr': training_args.tensor_lr_init, "name": "shs_dc3_line2"},
            {'params': [self._scale_line_0], 'lr': training_args.scaling_lr, "name": "scale_line0"},
            {'params': [self._scale_line_1], 'lr': training_args.scaling_lr, "name": "scale_line1"},
            {'params': [self._scale_line_2], 'lr': training_args.scaling_lr, "name": "scale_line2"},
            {'params': [self._rotat_line_0], 'lr': training_args.rotation_lr, "name": "rotat_line0"},
            {'params': [self._rotat_line_1], 'lr': training_args.rotation_lr, "name": "rotat_line1"},
            {'params': [self._rotat_line_2], 'lr': training_args.rotation_lr, "name": "rotat_line2"},
            # {'params': self.feas_layer_1.parameters(), 'lr': training_args.mlp_lr_init, "name": "fully1"},
            # {'params': self.feas_layer_2.parameters(), 'lr': training_args.mlp_lr_init, "name": "fully2"},
            {'params': self.feas_layer.parameters(), 'lr': training_args.mlp_lr_init, "name": "fully"},
            {'params': [self._mask_fac], 'lr': 0.01, "name": "mask"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.tensor_scheduler_args = get_expon_lr_func(lr_init=training_args.tensor_lr_init,
                                            lr_final=training_args.tensor_lr_final,
                                            lr_delay_mult=training_args.position_lr_delay_mult,
                                            max_steps=training_args.iterations)
        self.mlp_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_lr_init,
                                            lr_final=training_args.mlp_lr_final,
                                            lr_delay_mult=training_args.position_lr_delay_mult,
                                            max_steps=training_args.iterations)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "color_line0" or "color_line1" or "color_line2" or "color_plane0" or "color_plane1" or "color_plane2"\
                                        or "shs_dc_line0" or "shs_dc_line1" or "shs_dc_line3" or "shs_dc3_line0" or "shs_dc3_line1" or "shs_dc3_line2":
                lr_tensor = self.tensor_scheduler_args(iteration)
                param_group['lr'] = lr_tensor
            if param_group["name"] == "fully1" or "fully2":
                lr_mlp = self.mlp_scheduler_args(iteration)
                param_group['lr'] = lr_mlp

    def update_sample_learning_rate(self):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "x" or param_group["name"] == "y" or param_group["name"] == "z" :
                param_group['lr'] = 0

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path, viewdir):
        mkdir_p(os.path.dirname(path))
        
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

        if self._xyz.shape[0] != 0:
            shs_view = (torch.cat((self._features_dc, self._features_rest), dim=1)).transpose(1, 2).view(-1, 3, (self.max_sh_degree+1)**2)
            sh2rgb = eval_sh(self.active_sh_degree, shs_view, viewdir[self.fixnum:, :])
            colors = torch.clamp_min(sh2rgb + 0.5, 0.0)
            mins = colors.amin()
            maxs = colors.amax()
            colors = (colors - mins) / (maxs - mins) * 255
            pos_only_dtype = [('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('red', '<f4'), ('green', '<f4'), ('blue', '<f4')]
            pos_only = np.empty(xyz.shape[0], dtype=pos_only_dtype)
            pos_only[:] = list(map(tuple, np.concatenate([xyz, colors.cpu().detach().numpy()], axis=1)))
            pos_only_el = PlyElement.describe(pos_only, 'vertex')
            PlyData([pos_only_el]).write(path[:-4] + "_norm_pos.ply")

        # mask_fac = ((torch.sigmoid(self._mask_fac) > 0.1).float() - torch.sigmoid(self._mask_fac)).detach() + torch.sigmoid(self._mask_fac)
        # mask_fac = mask_fac.bool().squeeze()
        # pts_mask = self.get_opacity[:self.fixnum, :] > 0.005
        # pts_mask = pts_mask.squeeze(dim = 1)
        # xyz = self.get_xyz[:self.fixnum, :][mask_fac].detach().cpu().numpy()
        # shs_view = (self.get_features[:self.fixnum, :][mask_fac]).transpose(1, 2).view(-1, 3, (self.max_sh_degree+1)**2)
        # # shs_view = (torch.cat((self._features_dc_fac[pts_mask], self._features_rest_fac[pts_mask]), dim=1)).transpose(1, 2).view(-1, 3, (self.max_sh_degree+1)**2)
        # sh2rgb = eval_sh(self.active_sh_degree, shs_view, viewdir[:self.fixnum, :][mask_fac])
        # colors = torch.clamp_min(sh2rgb + 0.5, 0.0)
        # mins = colors.amin()
        # maxs = colors.amax()
        # colors = (colors - mins) / (maxs - mins) * 255
        # pos_only_dtype = [('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('red', '<f4'), ('green', '<f4'), ('blue', '<f4')]
        # pos_only = np.empty(xyz.shape[0], dtype=pos_only_dtype)
        # pos_only[:] = list(map(tuple, np.concatenate([xyz, colors.cpu().detach().numpy()], axis=1)))
        # pos_only_el = PlyElement.describe(pos_only, 'vertex')
        # PlyData([pos_only_el]).write(path[:-4] + "_fac_pos.ply")

        shs_view = self.get_features.transpose(1, 2).view(-1, 3, (self.max_sh_degree+1)**2)
        xyz = self.get_xyz.detach().cpu().numpy()
        sh2rgb = eval_sh(self.active_sh_degree, shs_view, viewdir)
        colors = torch.clamp_min(sh2rgb + 0.5, 0.0)
        mins = colors.amin()
        maxs = colors.amax()
        colors = (colors - mins) / (maxs - mins) * 255
        pos_only_dtype = [('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('red', '<f4'), ('green', '<f4'), ('blue', '<f4')]
        pos_only = np.empty(xyz.shape[0], dtype=pos_only_dtype)
        pos_only[:] = list(map(tuple, np.concatenate([xyz, colors.cpu().detach().numpy()], axis=1)))
        pos_only_el = PlyElement.describe(pos_only, 'vertex')
        PlyData([pos_only_el]).write(path[:-4] + "_all_pos.ply")

        # store VM features, CP features and points parameters, not including scales and rotations
        tensors_VM_5 = {'tensor1': self._app_line_0, 'tensor2': self._app_line_1, 'tensor2': self._app_line_2,\
                   'tensor4': self._app_plane_0, 'tensor5': self._app_plane_1, 'tensor6': self._app_plane_2}
        # torch.save(tensors_VM_5, path[:-4] + 'tensors_VM_5.pth')
        tensors_VM_3 = {'tensor1': self._app3_line_0, 'tensor2': self._app3_line_1, 'tensor2': self._app3_line_2,\
                   'tensor4': self._app3_plane_0, 'tensor5': self._app3_plane_1, 'tensor6': self._app3_plane_2}
        # torch.save(tensors_VM_3, path[:-4] + 'tensors_VM_3.pth')
        tensors_CP_5 = {'tensor1': self._shs_dc_line_0, 'tensor2': self._shs_dc_line_1, 'tensor3': self._shs_dc_line_2}
        # torch.save(tensors_CP_5, path[:-4] + 'tensors_CP_5.pth')
        tensors_CP_3 = {'tensor1': self._shs_dc3_line_0, 'tensor2': self._shs_dc3_line_1, 'tensor3': self._shs_dc3_line_2}
        # covariance
        tensors_cov = {'scale1': self._scale_line_0, 'scale2': self._scale_line_1, 'scale3': self._scale_line_2,\
                       'rotation1': self._rotat_line_0, 'rotation2': self._rotat_line_1, 'rotataion3': self._rotat_line_2}
        # torch.save(tensors_CP_3, path[:-4] + 'tensors_CP_3.pth')
        points = {'tensor7': self._x, 'tensor8': self._y, 'tensor9': self._z}
        # torch.save(points, path[:-4] + 'points.pth')
        mask_bool = (((torch.sigmoid(self._mask_fac) > 0.1).float() - torch.sigmoid(self._mask_fac)).detach() + torch.sigmoid(self._mask_fac)).bool()
        mask = {'tensor1': mask_bool}
        mlp = {'tensor1': self.feas_layer}

        VM_store = [tensors_VM_5, points, tensors_cov, mask, mlp]
        # torch.save (VM_store, path[:-4] + 'tensors_VM.pth')
        CP_store = [tensors_CP_5, points, tensors_cov, mask, mlp]
        torch.save (CP_store, path[:-4] + 'tensors_CP.pth')

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity[self.fixnum:, :], torch.ones_like(self.get_opacity[self.fixnum:, :])*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        # mod = torch.load("exp/chair_fix/point_cloud/iteration_40000/point_cloudtensors_CP.pth")
        # self._shs_dc_line_0 = mod[0]['tensor1']
        # self._shs_dc_line_1 = mod[0]['tensor2']
        # self._shs_dc_line_2 = mod[0]['tensor3']
        # self._x = mod[1]['tensor7']
        # self._y = mod[1]['tensor8']
        # self._z = mod[1]['tensor9']
        # self._scale_line_0 = mod[2]['scale1']
        # self._scale_line_1 = mod[2]['scale2']
        # self._scale_line_2 = mod[2]['scale3']
        # self._rotat_line_0 = mod[2]['rotation1']
        # self._rotat_line_1 = mod[2]['rotation2']
        # self._rotat_line_2 = mod[2]['rotation3']

        plydata = PlyData.read(path)

        # import pdb; pdb.set_trace()
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree
        

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] != "xyz" and group["name"] != "f_dc" and group["name"] != "f_rest" and group["name"] != "opacity" and group["name"] != "scaling" and group["name"] != "rotation":
                continue
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] != "xyz" and group["name"] != "f_dc" and group["name"] != "f_rest" and group["name"] != "opacity" and group["name"] != "scaling" and group["name"] != "rotation":
                continue
            # print('name', group["name"])
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.concat((self.xyz_gradient_accum[:self.fixnum],self.xyz_gradient_accum[self.fixnum:][valid_points_mask]), dim = 0)
        self.max_radii2D = torch.concat((self.max_radii2D[:self.fixnum],self.max_radii2D[self.fixnum:][valid_points_mask]), dim = 0)
        self.denom = torch.concat((self.denom[:self.fixnum],self.denom[self.fixnum:][valid_points_mask]), dim = 0)
        # self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        # self.denom = self.denom[valid_points_mask]
        # self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] != "xyz" and group["name"] != "f_dc" and group["name"] != "f_rest" and group["name"] != "opacity" and group["name"] != "scaling" and group["name"] != "rotation":
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # n_init_points = self._xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self.get_rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8 * N))
        new_rotation = self.get_rotation[selected_pts_mask].repeat(N,1)
        # print('new_features_dc', self.get_features.shape)
        new_features_dc = self.get_features[:, 0:1, :][selected_pts_mask].repeat(N,1,1)
        new_features_rest = self.get_features[:, 1:, :][selected_pts_mask].repeat(N,1,1)
        new_opacity = self.get_opacity[selected_pts_mask].repeat(N,1)

        self.get_scaling[selected_pts_mask] = self.get_scaling[selected_pts_mask] / (2.0 * N)
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        # prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        # self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        # print('mask', selected_pts_mask.shape)

        new_xyz = self.get_xyz[selected_pts_mask]
        # print('shape', self.get_features.shape)
        # print('shape', self.get_features[:, 0, :].shape)
        # print('shape',torch.concat((self._features_dc_fac, self._features_dc), dim = 0).shape)
        new_features_dc = self.get_features[:, 0:1, :][selected_pts_mask]
        new_features_rest = self.get_features[:, 1:, :][selected_pts_mask]
        new_opacities = self.get_opacity[selected_pts_mask]
        new_scaling = self.get_scaling[selected_pts_mask]
        new_rotation = self.get_rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        # print('shape', prune_mask.shape)
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        prune_mask = prune_mask[self.fixnum:]
        # print('shape', prune_mask.shape)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def normal_points_prune(self, min_opacity, extent, max_screen_size):
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        # print('shape', prune_mask.shape)
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        prune_mask = prune_mask[self.fixnum:]
        # print('shape', prune_mask.shape)
        self.prune_points(prune_mask)

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        point_grad = viewspace_point_tensor.grad
        # print('point', point_grad.shape)
        self.xyz_gradient_accum[update_filter] += torch.norm(point_grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
        # print('denom', self.denom)
    
    def mask_prune(self):
        prune_mask = (torch.sigmoid(self._mask_fac)<=0.1).squeeze()
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()

    # def point_mask(self):
    #     self.fac_mask = self._opacity_fac > 0.005
    #     self.fac_mask = self.fac_mask.squeeze(dim = 1)
    #     self._opacity_fac = self._opacity_fac[self.fac_mask]
    #     self._rotation_fac = self._rotation_fac[self.fac_mask]
    #     self._scaling_fac = self._scaling_fac[self.fac_mask]
    #     self._features_dc_fac = self._features_dc_fac[self.fac_mask]
    #     self._features_rest_fac = self._features_rest_fac[self.fac_mask]
    #     # print('fac_mask', self.fac_mask)
    #     self.xyz_gradient_accum = torch.concat((self.xyz_gradient_accum[:self.fixnum, :][self.fac_mask], self.xyz_gradient_accum[self.fixnum:, :]), dim = 0)
    #     self.denom = torch.concat((self.denom[:self.fixnum, :][self.fac_mask], self.denom[self.fixnum:, :]), dim = 0)
    #     self.max_radii2D = torch.concat((self.max_radii2D[:self.fixnum][self.fac_mask], self.max_radii2D[self.fixnum:]), dim = 0)
    #     # print('shape', self.xyz_gradient_accum.shape)
    #     self.fixnum = self.get_xyz.shape[0] - self._xyz.shape[0]
    #     # 有待检查
    #     # self.max_radii2D = self.max_radii2D[:self.get_xyz.shape[0]]
    #     # print('shape', self.max_radii2D.shape)