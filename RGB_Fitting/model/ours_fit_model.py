import os
import torch
import numpy as np
import torch.nn.functional as F
import torchvision

from .hifi3dpp import ParametricFaceModel
from .renderer_nvdiffrast import MeshRenderer
from . import uvtex_spherical_fixshape_fitter, uvtex_wspace_shape_joint_fitter, spherical_shape_fixuvtex_fitter
from RGB_Fitting.network import texgan
from RGB_Fitting.network.recog import define_net_recog
from RGB_Fitting.network.recon_deep3d import define_net_recon_deep3d
from RGB_Fitting.network.stylegan2 import dnnlib
from RGB_Fitting.utils.data_utils import setup_seed, tensor2np, draw_mask, draw_landmarks, img3channel
from RGB_Fitting.utils.mesh_utils import unwrap_vertex_to_uv, write_mesh_obj, write_mtl

import trimesh
from PIL import Image

class FitModel:

    def __init__(self, cpk_dir, topo_dir, texgan_model_name, loose_tex=False, lm86=False, device='cuda'):
        self.args_model = {
            # face model and renderer
            'fm_model_file': os.path.join(topo_dir, 'hifi3dpp_model_info.mat'),
            'unwrap_info_file': os.path.join(topo_dir, 'unwrap_1024_info.mat'),
            'camera_distance': 10.,
            'focal': 1015.,
            'center': 112.,
            'znear': 5.,
            'zfar': 15.,
            # texture gan
            'texgan_model_file': os.path.join(cpk_dir, f'texgan_model/{texgan_model_name}'),
            # deep3d nn inference model
            'net_recon': 'resnet50',
            'net_recon_path': os.path.join(cpk_dir, 'deep3d_model/epoch_latest.pth'),
            # recognition model
            'net_recog': 'r50',
            'net_recog_path': os.path.join(cpk_dir, 'arcface_model/ms1mv3_arcface_r50_fp16_backbone.pth'),
            # vgg model
            'net_vgg_path': os.path.join(cpk_dir, 'vgg_model/vgg16.pt'),
        }
        self.args_s2_search_uvtex_spherical_fixshape = {
            'w_feat': 10.0,
            'w_color': 10.0,
            'w_vgg': 100.0,
            'w_reg_latent': 0.05,
            'initial_lr': 0.1,
            'lr_rampdown_length': 0.25,
            'total_step': 100,
            'print_freq': 5,
            'visual_freq': 10,
        }
        self.args_s3_optimize_uvtex_shape_joint = {
            'w_feat': 0.2,
            'w_color': 1.6,
            'w_reg_id': 2e-4,
            'w_reg_exp': 1.6e-3,
            'w_reg_gamma': 10.0,
            'w_reg_latent': 0.05,
            'w_lm': 2e-3,
            'initial_lr': 0.01,
            'tex_lr_scale': 1.0 if loose_tex else 0.05,
            'lr_rampdown_length': 0.4,
            'total_step': 200,
            'print_freq': 10,
            'visual_freq': 20,
        }

        self.args_names = ['model', 's2_search_uvtex_spherical_fixshape', 's3_optimize_uvtex_shape_joint']

        # parametric face model
        self.facemodel = ParametricFaceModel(fm_model_file=self.args_model['fm_model_file'],
                                             unwrap_info_file=self.args_model['unwrap_info_file'],
                                             camera_distance=self.args_model['camera_distance'],
                                             focal=self.args_model['focal'],
                                             center=self.args_model['center'],
                                             lm86=lm86,
                                             device=device)

        # texture gan
        self.tex_gan = texgan.TextureGAN(model_path=self.args_model['texgan_model_file'], device=device)

        # deep3d nn reconstruction model
        fc_info = {
            'id_dims': self.facemodel.id_dims,
            'exp_dims': self.facemodel.exp_dims,
            'tex_dims': self.facemodel.tex_dims
        }
        self.net_recon_deep3d = define_net_recon_deep3d(net_recon=self.args_model['net_recon'],
                                                        use_last_fc=False,
                                                        fc_dim_dict=fc_info,
                                                        pretrained_path=self.args_model['net_recon_path'])
        self.net_recon_deep3d = self.net_recon_deep3d.eval().requires_grad_(False)

        # renderer
        fov = 2 * np.arctan(self.args_model['center'] / self.args_model['focal']) * 180 / np.pi
        self.renderer = MeshRenderer(fov=fov,
                                     znear=self.args_model['znear'],
                                     zfar=self.args_model['zfar'],
                                     rasterize_size=int(2 * self.args_model['center']))

        # the recognition model
        self.net_recog = define_net_recog(net_recog=self.args_model['net_recog'],
                                          pretrained_path=self.args_model['net_recog_path'])
        self.net_recog = self.net_recog.eval().requires_grad_(False)

        # the vgg model
        with dnnlib.util.open_url(self.args_model['net_vgg_path']) as f:
            self.net_vgg = torch.jit.load(f).eval()

        # coeffs and latents
        self.pred_coeffs = None
        self.pred_latents_w = None
        self.pred_latents_z = None

        self.to(device)
        self.device = device

    def to(self, device):
        self.device = device
        self.facemodel.to(device)
        self.tex_gan.to(device)
        self.net_recon_deep3d.to(device)
        self.renderer.to(device)
        self.net_recog.to(device)
        self.net_vgg.to(device)

    def infer_render(self, is_uv_tex=True):
        # forward face model
        self.pred_coeffs_dict = self.facemodel.split_coeff(self.pred_coeffs)
        self.pred_vertex, self.pred_tex, self.pred_shading, self.pred_color, self.pred_lm = \
            self.facemodel.compute_for_render(self.pred_coeffs_dict)
        if is_uv_tex:
            # forward texture gan
            self.pred_uv_map = self.tex_gan.synth_uv_map(self.pred_latents_w)
            # render front face
            vertex_uv_coord = self.facemodel.vtx_vt.unsqueeze(0).repeat(self.pred_coeffs.size()[0], 1, 1)
            render_feat = torch.cat([vertex_uv_coord, self.pred_shading], axis=2)
            self.render_face_mask, _, self.render_face = \
                self.renderer(self.pred_vertex, self.facemodel.face_buf, feat=render_feat, uv_map=self.pred_uv_map)
        else:
            # render front face
            self.render_face_mask, _, self.render_face = \
                self.renderer(self.pred_vertex, self.facemodel.face_buf, feat=self.pred_color)

    def visualize(self, input_data, is_uv_tex=True):
        # input data
        input_img = tensor2np(input_data['img'][:1, :, :, :])
        skin_img = img3channel(tensor2np(input_data['skin_mask'][:1, :, :, :]))
        parse_mask = tensor2np(input_data['parse_mask'][:1, :, :, :], dst_range=1.0)
        gt_lm = input_data['lm'][0, :, :].detach().cpu().numpy()
        # predict data
        pred_face_img = self.render_face * self.render_face_mask + (1 - self.render_face_mask) * input_data['img']
        pred_face_img = tensor2np(pred_face_img[:1, :, :, :])
        pred_lm = self.pred_lm[0, :, :].detach().cpu().numpy()
        # draw mask and landmarks
        parse_img = draw_mask(input_img, parse_mask)
        gt_lm[..., 1] = pred_face_img.shape[0] - 1 - gt_lm[..., 1]
        pred_lm[..., 1] = pred_face_img.shape[0] - 1 - pred_lm[..., 1]
        lm_img = draw_landmarks(pred_face_img, gt_lm, color='b')
        lm_img = draw_landmarks(lm_img, pred_lm, color='r')
        # combine visual images
        combine_img = np.concatenate([input_img, skin_img, parse_img, lm_img, pred_face_img], axis=1)
        if is_uv_tex:
            pre_uv_img = tensor2np(F.interpolate(self.pred_uv_map, size=input_img.shape[:2], mode='area')[:1, :, :, :])
            combine_img = np.concatenate([combine_img, pre_uv_img], axis=1)
        return combine_img

    def visualize_3dmmtex_as_uv(self):
        tex_vertex = self.pred_tex[0, :, :].detach().cpu().numpy()
        unwrap_uv_idx_v_idx = self.facemodel.unwrap_uv_idx_v_idx.detach().cpu().numpy()
        unwrap_uv_idx_bw = self.facemodel.unwrap_uv_idx_bw.detach().cpu().numpy()
        tex_uv = unwrap_vertex_to_uv(tex_vertex, unwrap_uv_idx_v_idx, unwrap_uv_idx_bw) * 255.
        return tex_uv

    def save_mesh(self, path, mesh_name, mlt_name=None, uv_name=None, is_uv_tex=True):
        pred_coeffs_dict = self.facemodel.split_coeff(self.pred_coeffs)
        pred_id_vertex, pred_exp_vertex, pred_alb_tex = self.facemodel.compute_for_mesh(pred_coeffs_dict)
        if is_uv_tex:
            assert mlt_name is not None and uv_name is not None
            write_mtl(os.path.join(path, mlt_name), uv_name)
            id_mesh_info = {
                'v': pred_id_vertex.detach()[0].cpu().numpy(),
                'vt': self.facemodel.vt_list.cpu().numpy(),
                'fv': self.facemodel.head_buf.cpu().numpy(),
                'fvt': self.facemodel.head_tri_vt.cpu().numpy(),
                'mtl_name': mlt_name
            }
            exp_mesh_info = {
                'v': pred_exp_vertex.detach()[0].cpu().numpy(),
                'vt': self.facemodel.vt_list.cpu().numpy(),
                'fv': self.facemodel.head_buf.cpu().numpy(),
                'fvt': self.facemodel.head_tri_vt.cpu().numpy(),
                'mtl_name': mlt_name
            }
        else:
            id_mesh_info = {
                'v': pred_id_vertex.detach()[0].cpu().numpy(),
                'vt': pred_alb_tex.detach()[0].cpu().numpy(),
                'fv': self.facemodel.head_buf.cpu().numpy()
            }
            exp_mesh_info = {
                'v': pred_exp_vertex.detach()[0].cpu().numpy(),
                'vt': pred_alb_tex.detach()[0].cpu().numpy(),
                'fv': self.facemodel.head_buf.cpu().numpy()
            }
        write_mesh_obj(mesh_info=id_mesh_info, file_path=os.path.join(path, f'{mesh_name[:-4]}_id{mesh_name[-4:]}'))
        write_mesh_obj(mesh_info=exp_mesh_info, file_path=os.path.join(path, f'{mesh_name[:-4]}_exp{mesh_name[-4:]}'))

    def save_coeffs(self, path, coeffs_name, is_uv_tex=True):
        # coeffs & landmarks
        coeffs_info = {'coeffs': self.pred_coeffs, 'lm68': self.pred_lm}
        if is_uv_tex:
            coeffs_info['latents_w'] = self.pred_latents_w
            coeffs_info['latents_z'] = self.pred_latents_z
        torch.save(coeffs_info, os.path.join(path, coeffs_name))

    def gather_args_str(self):
        args_str = '\n'
        for name in self.args_names:
            args_dict = getattr(self, 'args_' + name)
            args_str += f'----------------- Args-{name} ---------------\n'
            for k, v in args_dict.items():
                args_str += '{:>30}: {:<30}\n'.format(str(k), str(v))
        args_str += '----------------- End -------------------'
        return args_str


    def fitting(self, input_data, logger):
        # fix random seed
        setup_seed(123)

        # print args
        logger.write_txt_log(self.gather_args_str())

        # save the input data
        torch.save(input_data, os.path.join(logger.vis_dir, f'input_data.pt'))

        #--------- Stage 1 - getting initial coeffs by Deep3D NN inference ---------

        logger.write_txt_log('Stage 1 getting initial coeffs by Deep3D NN inference.')
        with torch.no_grad():
            self.pred_coeffs = self.net_recon_deep3d(input_data['img'].to(self.device))
        self.infer_render(is_uv_tex=False)
        vis_img = self.visualize(input_data, is_uv_tex=False)
        vis_tex_uv = self.visualize_3dmmtex_as_uv()
        logger.write_disk_images([vis_img], ['stage1_vis'])
        logger.write_disk_images([vis_tex_uv], ['stage1_vis_3dmmtex_as_uv'])
        self.save_mesh(path=logger.vis_dir, mesh_name='stage1_mesh.obj', is_uv_tex=False)
        self.save_coeffs(path=logger.vis_dir, coeffs_name='stage1_coeffs.pt', is_uv_tex=False)
        
        #--------- Stage 2 - search UV tex on a spherical surface with fixed shape ---------

        logger.write_txt_log('Start stage 2 searching UV tex on a spherical surface with fixed shape.')
        logger.reset_prefix(prefix='s2_search_uvtex_spherical_fixshape')
        fitter = uvtex_spherical_fixshape_fitter.Fitter(facemodel=self.facemodel,
                                                        tex_gan=self.tex_gan,
                                                        renderer=self.renderer,
                                                        net_recog=self.net_recog,
                                                        net_vgg=self.net_vgg,
                                                        logger=logger,
                                                        input_data=input_data,
                                                        init_coeffs=self.pred_coeffs,
                                                        init_latents_z=None,
                                                        **self.args_s2_search_uvtex_spherical_fixshape)
        self.pred_coeffs, self.pred_latents_z, self.pred_latents_w = fitter.iterate()
        logger.reset_prefix()
        logger.write_txt_log('End stage 2 searching UV tex on a spherical surface with fixed shape.')

        self.infer_render()
        vis_img = self.visualize(input_data)
        logger.write_disk_images([vis_img], ['stage2_vis'])
        logger.write_disk_images([tensor2np(self.pred_uv_map[:1, :, :, :])], ['stage2_uv'])
        self.save_mesh(path=logger.vis_dir,
                       mesh_name='stage2_mesh.obj',
                       mlt_name='stage2_mesh.mlt',
                       uv_name='stage2_uv.png')
        self.save_coeffs(path=logger.vis_dir, coeffs_name='stage2_coeffs.pt')

        #--------- Stage 3 - jointly optimize UV tex and shape ---------

        logger.write_txt_log('Start stage 3 jointly optimize UV tex and shape.')
        logger.reset_prefix(prefix='s3_optimize_uvtex_shape_joint')
        fitter = uvtex_wspace_shape_joint_fitter.Fitter(facemodel=self.facemodel,
                                                        tex_gan=self.tex_gan,
                                                        renderer=self.renderer,
                                                        net_recog=self.net_recog,
                                                        net_vgg=self.net_vgg,
                                                        logger=logger,
                                                        input_data=input_data,
                                                        init_coeffs=self.pred_coeffs,
                                                        init_latents_z=self.pred_latents_z,
                                                        **self.args_s3_optimize_uvtex_shape_joint)
        self.pred_coeffs, self.pred_latents_z, self.pred_latents_w = fitter.iterate()
        logger.reset_prefix()
        logger.write_txt_log('End stage 3 jointly optimize UV tex and shape.')

        self.infer_render()
        vis_img = self.visualize(input_data)
        logger.write_disk_images([vis_img], ['stage3_vis'])
        logger.write_disk_images([tensor2np(self.pred_uv_map[:1, :, :, :])], ['stage3_uv'])
        self.save_mesh(path=logger.vis_dir,
                       mesh_name='stage3_mesh.obj',
                       mlt_name='stage3_mesh.mlt',
                       uv_name='stage3_uv.png')
        self.save_coeffs(path=logger.vis_dir, coeffs_name='stage3_coeffs.pt')

    def rec_fitting(self, input_data, rec_dir):
        #--------- Stage 1 - getting initial coeffs by Deep3D NN inference ---------

        with torch.no_grad():
            self.pred_coeffs = self.net_recon_deep3d(input_data['img'].to(self.device))
        self.infer_render(is_uv_tex=False)
        vis_img = self.visualize(input_data, is_uv_tex=False)
        vis_tex_uv = self.visualize_3dmmtex_as_uv()
        torchvision.utils.save_image(torch.tensor(vis_img).permute(2, 0, 1).unsqueeze(0)/255.,
                                        os.path.join(rec_dir, 'stage1_vis.png'))
        torchvision.utils.save_image(torch.tensor(vis_tex_uv).permute(2, 0, 1).unsqueeze(0)/255.,
                                        os.path.join(rec_dir, 'stage1_vis_3dmmtex_as_uv.png'))
        self.save_mesh(path=rec_dir, mesh_name='stage1_mesh.obj', is_uv_tex=False)
        self.save_coeffs(path=rec_dir, coeffs_name='stage1_coeffs.pt', is_uv_tex=False)

    def fitting1011(self, input_data, logger):
        # fix random seed
        setup_seed(123)

        # print args
        logger.write_txt_log(self.gather_args_str())

        # save the input data
        torch.save(input_data, os.path.join(logger.vis_dir, f'input_data.pt'))

        #--------- Stage 1 - getting initial coeffs by Deep3D NN inference ---------

        logger.write_txt_log('Stage 1 getting initial coeffs by Deep3D NN inference.')
        with torch.no_grad():
            self.pred_coeffs = self.net_recon_deep3d(input_data['img'].to(self.device))
        self.infer_render(is_uv_tex=False)
        vis_img = self.visualize(input_data, is_uv_tex=False)
        vis_tex_uv = self.visualize_3dmmtex_as_uv()
        logger.write_disk_images([vis_img], ['stage1_vis'])
        logger.write_disk_images([vis_tex_uv], ['stage1_vis_3dmmtex_as_uv'])
        self.save_mesh(path=logger.vis_dir, mesh_name='stage1_mesh.obj', is_uv_tex=False)
        self.save_coeffs(path=logger.vis_dir, coeffs_name='stage1_coeffs.pt', is_uv_tex=False)

        #--------- Stage 2 - search UV tex on a spherical surface with fixed shape ---------

        logger.write_txt_log('Start stage 2 searching UV tex on a spherical surface with fixed shape.')
        logger.reset_prefix(prefix='s2_search_uvtex_spherical_fixshape')
        fitter = uvtex_spherical_fixshape_fitter.Fitter(facemodel=self.facemodel,
                                                        tex_gan=self.tex_gan,
                                                        renderer=self.renderer,
                                                        net_recog=self.net_recog,
                                                        net_vgg=self.net_vgg,
                                                        logger=logger,
                                                        input_data=input_data,
                                                        init_coeffs=self.pred_coeffs,
                                                        init_latents_z=None,
                                                        **self.args_s2_search_uvtex_spherical_fixshape)
        self.pred_coeffs, self.pred_latents_z, self.pred_latents_w = fitter.iterate()
        logger.reset_prefix()
        logger.write_txt_log('End stage 2 searching UV tex on a spherical surface with fixed shape.')

        self.infer_render()
        vis_img = self.visualize(input_data)
        logger.write_disk_images([vis_img], ['stage2_vis'])
        logger.write_disk_images([tensor2np(self.pred_uv_map[:1, :, :, :])], ['stage2_uv'])
        self.save_mesh(path=logger.vis_dir,
                       mesh_name='stage2_mesh.obj',
                       mlt_name='stage2_mesh.mlt',
                       uv_name='stage2_uv.png')
        self.save_coeffs(path=logger.vis_dir, coeffs_name='stage2_coeffs.pt')

        #--------- Stage 3 - jointly optimize UV tex and shape ---------

        logger.write_txt_log('Start stage 3 jointly optimize UV tex and shape.')
        logger.reset_prefix(prefix='s3_optimize_uvtex_shape_joint')
        fitter = uvtex_wspace_shape_joint_fitter.Fitter(facemodel=self.facemodel,
                                                        tex_gan=self.tex_gan,
                                                        renderer=self.renderer,
                                                        net_recog=self.net_recog,
                                                        net_vgg=self.net_vgg,
                                                        logger=logger,
                                                        input_data=input_data,
                                                        init_coeffs=self.pred_coeffs,
                                                        init_latents_z=self.pred_latents_z,
                                                        **self.args_s3_optimize_uvtex_shape_joint)
        self.pred_coeffs, self.pred_latents_z, self.pred_latents_w = fitter.iterate()
        logger.reset_prefix()
        logger.write_txt_log('End stage 3 jointly optimize UV tex and shape.')

        self.infer_render()
        vis_img = self.visualize(input_data)
        logger.write_disk_images([vis_img], ['stage3_vis'])
        logger.write_disk_images([tensor2np(self.pred_uv_map[:1, :, :, :])], ['stage3_uv'])
        self.save_mesh(path=logger.vis_dir,
                       mesh_name='stage3_mesh.obj',
                       mlt_name='stage3_mesh.mlt',
                       uv_name='stage3_uv.png')
        self.save_coeffs(path=logger.vis_dir, coeffs_name='stage3_coeffs.pt')

class RecModel:

    def __init__(self, cpk_dir, topo_dir, loose_tex=False, lm86=False, device='cuda'):
        self.args_model = {
            # face model and renderer
            'fm_model_file': os.path.join(topo_dir, 'hifi3dpp_model_info.mat'),
            'unwrap_info_file': os.path.join(topo_dir, 'unwrap_1024_info.mat'),
            'camera_distance': 10.,
            'focal': 1015.,
            'center': 112.,
            'znear': 5.,
            'zfar': 15.,
            # deep3d nn inference model
            'net_recon': 'resnet50',
            'net_recon_path': os.path.join(cpk_dir, 'deep3d_model/epoch_latest.pth'),
            # recognition model
            'net_recog': 'r50',
            'net_recog_path': os.path.join(cpk_dir, 'arcface_model/ms1mv3_arcface_r50_fp16_backbone.pth'),
            # vgg model
            'net_vgg_path': os.path.join(cpk_dir, 'vgg_model/vgg16.pt'),
        }
        self.args_s2_search_uvtex_spherical_fixshape = {
            'w_feat': 10.0,
            'w_color': 10.0,
            'w_vgg': 100.0,
            'w_reg_latent': 0.05,
            'initial_lr': 0.1,
            'lr_rampdown_length': 0.25,
            'total_step': 100,
            'print_freq': 5,
            'visual_freq': 10,
        }
        self.args_s3_optimize_uvtex_shape_joint = {
            'w_feat': 0.2,
            'w_color': 1.6,
            'w_reg_id': 2e-3,
            'w_reg_exp': 1.6e-3,
            'w_reg_gamma': 10.0,
            # 'w_reg_latent': 0.05,
            'w_lm': 2e-3,
            'initial_lr': 0.01,
            'tex_lr_scale': 1.0 if loose_tex else 0.05,
            'lr_rampdown_length': 0.4,
            'total_step': 200,
            'print_freq': 10,
            'visual_freq': 20,
        }

        self.args_names = ['model', 's2_search_uvtex_spherical_fixshape', 's3_optimize_uvtex_shape_joint']

        # parametric face model
        self.facemodel = ParametricFaceModel(fm_model_file=self.args_model['fm_model_file'],
                                             unwrap_info_file=self.args_model['unwrap_info_file'],
                                             camera_distance=self.args_model['camera_distance'],
                                             focal=self.args_model['focal'],
                                             center=self.args_model['center'],
                                             lm86=lm86,
                                             device=device)

        # deep3d nn reconstruction model
        fc_info = {
            'id_dims': self.facemodel.id_dims,
            'exp_dims': self.facemodel.exp_dims,
            'tex_dims': self.facemodel.tex_dims
        }
        self.net_recon_deep3d = define_net_recon_deep3d(net_recon=self.args_model['net_recon'],
                                                        use_last_fc=False,
                                                        fc_dim_dict=fc_info,
                                                        pretrained_path=self.args_model['net_recon_path'])
        self.net_recon_deep3d = self.net_recon_deep3d.eval().requires_grad_(False)

        # renderer
        fov = 2 * np.arctan(self.args_model['center'] / self.args_model['focal']) * 180 / np.pi
        self.renderer = MeshRenderer(fov=fov,
                                     znear=self.args_model['znear'],
                                     zfar=self.args_model['zfar'],
                                     rasterize_size=int(2 * self.args_model['center']))
        self.hr_renderer = MeshRenderer(fov=fov,
                                     znear=self.args_model['znear'],
                                     zfar=self.args_model['zfar'],
                                     rasterize_size=512)
        # the recognition model
        self.net_recog = define_net_recog(net_recog=self.args_model['net_recog'],
                                          pretrained_path=self.args_model['net_recog_path'])
        self.net_recog = self.net_recog.eval().requires_grad_(False)

        # the vgg model
        with dnnlib.util.open_url(self.args_model['net_vgg_path']) as f:
            self.net_vgg = torch.jit.load(f).eval()

        # coeffs and latents
        self.pred_coeffs = None
        self.pred_latents_w = None
        self.pred_latents_z = None

        self.to(device)
        self.device = device

    def to(self, device):
        self.device = device
        self.facemodel.to(device)
        self.net_recon_deep3d.to(device)
        self.renderer.to(device)
        self.net_recog.to(device)
        self.net_vgg.to(device)

    def infer_render(self, is_uv_tex=True):
        # forward face model
        self.pred_coeffs_dict = self.facemodel.split_coeff(self.pred_coeffs)
        self.pred_vertex, self.pred_tex, self.pred_shading, self.pred_color, self.pred_lm = \
            self.facemodel.compute_for_render(self.pred_coeffs_dict)
        if is_uv_tex:
            # render front face
            vertex_uv_coord = self.facemodel.vtx_vt.unsqueeze(0).repeat(self.pred_coeffs.size()[0], 1, 1)
            render_feat = torch.cat([vertex_uv_coord, self.pred_shading], axis=2)
            self.render_face_mask, _, self.render_face = \
                self.renderer(self.pred_vertex, self.facemodel.face_buf, feat=render_feat, uv_map=self.pred_uv_map)
        else:
            # render front face
            self.render_face_mask, _, self.render_face = \
                self.renderer(self.pred_vertex, self.facemodel.face_buf, feat=self.pred_color)
    
    def infer_render_hr(self, is_uv_tex=True):
        # forward face model
        self.pred_coeffs_dict = self.facemodel.split_coeff(self.pred_coeffs)
        self.pred_vertex, self.pred_tex, self.pred_shading, self.pred_color, self.pred_lm = \
            self.facemodel.compute_for_render(self.pred_coeffs_dict)
        if is_uv_tex:
            # forward texture gan
            self.pred_uv_map = self.tex_gan.synth_uv_map(self.pred_latents_w)
            # render front face
            vertex_uv_coord = self.facemodel.vtx_vt.unsqueeze(0).repeat(self.pred_coeffs.size()[0], 1, 1)
            render_feat = torch.cat([vertex_uv_coord, self.pred_shading], axis=2)
            self.render_face_mask, _, self.render_face = \
                self.hr_renderer(self.pred_vertex, self.facemodel.face_buf, feat=render_feat, uv_map=self.pred_uv_map)
        else:
            # render front face
            self.render_face_mask, _, self.render_face = \
                self.hr_renderer(self.pred_vertex, self.facemodel.face_buf, feat=self.pred_color)

    def visualize(self, input_data, is_uv_tex=True):
        # input data
        input_img = tensor2np(input_data['img'][:1, :, :, :])
        skin_img = img3channel(tensor2np(input_data['skin_mask'][:1, :, :, :]))
        parse_mask = tensor2np(input_data['parse_mask'][:1, :, :, :], dst_range=1.0)
        gt_lm = input_data['lm'][0, :, :].detach().cpu().numpy()
        # predict data
        pred_face_img = self.render_face * self.render_face_mask + (1 - self.render_face_mask) * input_data['img']
        pred_face_img = tensor2np(pred_face_img[:1, :, :, :])
        pred_lm = self.pred_lm[0, :, :].detach().cpu().numpy()
        # draw mask and landmarks
        parse_img = draw_mask(input_img, parse_mask)
        gt_lm[..., 1] = pred_face_img.shape[0] - 1 - gt_lm[..., 1]
        pred_lm[..., 1] = pred_face_img.shape[0] - 1 - pred_lm[..., 1]
        lm_img = draw_landmarks(pred_face_img, gt_lm, color='b')
        lm_img = draw_landmarks(lm_img, pred_lm, color='r')
        # combine visual images
        combine_img = np.concatenate([input_img, skin_img, parse_img, lm_img, pred_face_img], axis=1)
        if is_uv_tex:
            pre_uv_img = tensor2np(F.interpolate(self.pred_uv_map, size=input_img.shape[:2], mode='area')[:1, :, :, :])
            combine_img = np.concatenate([combine_img, pre_uv_img], axis=1)
        return combine_img

    def visualize_3dmmtex_as_uv(self):
        tex_vertex = self.pred_tex[0, :, :].detach().cpu().numpy()
        unwrap_uv_idx_v_idx = self.facemodel.unwrap_uv_idx_v_idx.detach().cpu().numpy()
        unwrap_uv_idx_bw = self.facemodel.unwrap_uv_idx_bw.detach().cpu().numpy()
        tex_uv = unwrap_vertex_to_uv(tex_vertex, unwrap_uv_idx_v_idx, unwrap_uv_idx_bw) * 255.
        return tex_uv

    def save_mesh(self, path, mesh_name, mlt_name=None, uv_name=None, is_uv_tex=True, tex=None):
        pred_coeffs_dict = self.facemodel.split_coeff(self.pred_coeffs)
        pred_id_vertex, pred_exp_vertex, pred_alb_tex = self.facemodel.compute_for_mesh(pred_coeffs_dict)
        if is_uv_tex:
            assert mlt_name is not None and uv_name is not None
            write_mtl(os.path.join(path, mlt_name), uv_name)
            id_mesh_info = {
                'v': pred_id_vertex.detach()[0].cpu().numpy(),
                'vt': self.facemodel.vt_list.cpu().numpy(),
                'fv': self.facemodel.head_buf.cpu().numpy(),
                'fvt': self.facemodel.head_tri_vt.cpu().numpy(),
                'mtl_name': mlt_name
            }
            exp_mesh_info = {
                'v': pred_exp_vertex.detach()[0].cpu().numpy(),
                'vt': self.facemodel.vt_list.cpu().numpy(),
                'fv': self.facemodel.head_buf.cpu().numpy(),
                'fvt': self.facemodel.head_tri_vt.cpu().numpy(),
                'mtl_name': mlt_name
            }
        else:
            id_mesh_info = {
                'v': pred_id_vertex.detach()[0].cpu().numpy(),
                'vt': pred_alb_tex.detach()[0].cpu().numpy(),
                'fv': self.facemodel.head_buf.cpu().numpy()
            }
            exp_mesh_info = {
                'v': pred_exp_vertex.detach()[0].cpu().numpy(),
                'vt': pred_alb_tex.detach()[0].cpu().numpy(),
                'fv': self.facemodel.head_buf.cpu().numpy()
            }
        write_mesh_obj(mesh_info=id_mesh_info, file_path=os.path.join(path, f'{mesh_name[:-4]}_id{mesh_name[-4:]}'))
        write_mesh_obj(mesh_info=exp_mesh_info, file_path=os.path.join(path, f'{mesh_name[:-4]}_exp{mesh_name[-4:]}'))

        if tex is not None:
            basecolor_tex = Image.fromarray(np.uint8(tex)).convert('RGB')
            basecolor_tex.format = "JPEG"
            roughness = 0.7
            metallic = 0.
            bump_tex = None

            material = trimesh.visual.material.PBRMaterial(
                            baseColorTexture=basecolor_tex,
                            roughnessFactor=roughness,
                            metallicFactor=metallic,
                            normalTexture=bump_tex,
                        )

            vt_list = self.facemodel.vt_list.cpu().numpy()
            vt_vtx_idx = self.facemodel.vt_vtx_idx.cpu().numpy()
            new_vt_list = np.zeros_like(vt_list)
            for i in range(len(vt_list)):
                new_vt_list[vt_vtx_idx[i]] = vt_list[i]

            tmesh = trimesh.Trimesh(
                vertices=pred_id_vertex.detach()[0].cpu().numpy(),
                faces=self.facemodel.head_buf.cpu().numpy(),
                visual=trimesh.visual.texture.TextureVisuals(
                    uv=new_vt_list,
                    material=material
                ),
            )

            tmesh.export(os.path.join(path, f'{mesh_name[:-4]}_id.glb'), file_type='glb', include_normals=True)
    
    def save_id_mesh(self, path, id_coeffs, mesh_name, mlt_name=None, uv_name=None, is_uv_tex=True, tex=None):
        pred_id_vertex = self.facemodel.compute_id_shape(id_coeffs)
        if is_uv_tex:
            assert mlt_name is not None and uv_name is not None
            write_mtl(os.path.join(path, mlt_name), uv_name)
            id_mesh_info = {
                'v': pred_id_vertex.detach()[0].cpu().numpy(),
                'vt': self.facemodel.vt_list.cpu().numpy(),
                'fv': self.facemodel.head_buf.cpu().numpy(),
                'fvt': self.facemodel.head_tri_vt.cpu().numpy(),
                'mtl_name': mlt_name
            }
        write_mesh_obj(mesh_info=id_mesh_info, file_path=os.path.join(path, f'{mesh_name[:-4]}_id{mesh_name[-4:]}'))

        if tex is not None:
            basecolor_tex = Image.fromarray(np.uint8(tex)).convert('RGB')
            basecolor_tex.format = "JPEG"
            roughness = 0.7
            metallic = 0.
            bump_tex = None

            material = trimesh.visual.material.PBRMaterial(
                            baseColorTexture=basecolor_tex,
                            roughnessFactor=roughness,
                            metallicFactor=metallic,
                            normalTexture=bump_tex,
                        )

            vt_list = self.facemodel.vt_list.cpu().numpy()
            vt_vtx_idx = self.facemodel.vt_vtx_idx.cpu().numpy()
            new_vt_list = np.zeros_like(vt_list)
            for i in range(len(vt_list)):
                new_vt_list[vt_vtx_idx[i]] = vt_list[i]

            tmesh = trimesh.Trimesh(
                vertices=pred_id_vertex.detach()[0].cpu().numpy(),
                faces=self.facemodel.head_buf.cpu().numpy(),
                visual=trimesh.visual.texture.TextureVisuals(
                    uv=new_vt_list,
                    material=material
                ),
            )

            tmesh.export(os.path.join(path, f'{mesh_name[:-4]}_id.glb'), file_type='glb', include_normals=True)

    def save_coeffs(self, path, coeffs_name, is_uv_tex=True):
        # coeffs & landmarks
        coeffs_info = {'coeffs': self.pred_coeffs, 'lm68': self.pred_lm}
        if is_uv_tex:
            coeffs_info['latents_w'] = self.pred_latents_w
            coeffs_info['latents_z'] = self.pred_latents_z
        torch.save(coeffs_info, os.path.join(path, coeffs_name))

    def save_id_coeffs(self, path, id_coeffs, coeffs_name):
        # coeffs & landmarks
        coeffs_info = {'id_coeffs': id_coeffs}
        torch.save(coeffs_info, os.path.join(path, coeffs_name))

    def gather_args_str(self):
        args_str = '\n'
        for name in self.args_names:
            args_dict = getattr(self, 'args_' + name)
            args_str += f'----------------- Args-{name} ---------------\n'
            for k, v in args_dict.items():
                args_str += '{:>30}: {:<30}\n'.format(str(k), str(v))
        args_str += '----------------- End -------------------'
        return args_str
    
    def rec_training(self, input_img, uv_map=None):
        crop_img = input_img.to(self.device)
        self.pred_coeffs = self.net_recon_deep3d(crop_img)
        self.pred_uv_map = uv_map
        self.infer_render()
        render_face = self.render_face * self.render_face_mask + (1 - self.render_face_mask) * crop_img
        return render_face, self.render_face, self.render_face_mask
    
    def rec_id_rendering(self, rec_dir, uv_map, id_coeffs, stage):
        if id_coeffs is None:
            pred_coeffs_dict = self.facemodel.split_coeff(self.pred_coeffs)
            pred_id_vertex, pred_exp_vertex, pred_alb_tex = self.facemodel.compute_for_mesh(pred_coeffs_dict)
            pred_id_vertex = pred_exp_vertex
        else:
            pred_id_vertex = self.facemodel.compute_id_shape(id_coeffs)
        pred_vertex_camera = self.facemodel.to_camera(pred_id_vertex)
        vertex_uv_coord = self.facemodel.vtx_vt.unsqueeze(0).repeat(uv_map.size()[0], 1, 1)
        render_feat = vertex_uv_coord
        self.pred_uv_map = uv_map
        self.render_head_mask, _, self.render_head = \
            self.hr_renderer(pred_vertex_camera, self.facemodel.head_buf, feat=render_feat, uv_map=self.pred_uv_map, shade=False)

        pred_head_img = self.render_head * self.render_head_mask
        pred_head_img = tensor2np(pred_head_img[:1, :, :, :])
        torchvision.utils.save_image(torch.tensor(pred_head_img).permute(2, 0, 1).unsqueeze(0)/255.,
                                    os.path.join(rec_dir, f'id_rendering_{stage}.png'))
        return pred_head_img
        # return pred_head_img
    
    def rec_zt_rendering(self, target_img, uv_map):
        pred_coeffs = self.net_recon_deep3d(target_img)
        pred_coeffs_dict = self.facemodel.split_coeff(pred_coeffs)
        pred_vertex, pred_tex, pred_shading, pred_color, pred_lm = self.facemodel.compute_for_render(pred_coeffs_dict)
        vertex_uv_coord = self.facemodel.vtx_vt.unsqueeze(0).repeat(pred_coeffs.size()[0], 1, 1)
        render_feat = torch.cat([vertex_uv_coord, pred_shading], axis=2)
        render_face_mask, _, render_face = \
            self.renderer(pred_vertex, self.facemodel.face_buf, feat=render_feat, uv_map=uv_map, shade=False)
        zt_face_img = render_face * render_face_mask + (1 - render_face_mask) * target_img
        
        # render_face_mask, _, render_face = \
        #     self.renderer(pred_vertex, self.facemodel.head_buf, feat=render_feat, uv_map=uv_map, shade=False)
        # zt_face_img = render_face * render_face_mask + (1 - render_face_mask) * target_img
        
        return zt_face_img, render_face, render_face_mask
    

    def rec_fitting(self, input_data, rec_dir, uv_map=None, id_coeffs=None, stage=2, logger=None):
        #--------- Stage 1 - getting initial coeffs by Deep3D NN inference ---------
        if id_coeffs is None:
            if 'incomlete' in input_data:
                input_data.pop('incomlete')
            if 'face_emb' in input_data:
                input_data.pop('face_emb')
            if 'fill_mask' in input_data:
                input_data.pop('fill_mask')

            input_data = {k: v.to(self.device) for (k, v) in input_data.items()}

            if self.pred_coeffs is None:
                with torch.no_grad():
                    self.pred_coeffs = self.net_recon_deep3d(input_data['img'].to(self.device))
                self.infer_render(is_uv_tex=False)
                vis_img = self.visualize(input_data, is_uv_tex=False)
                vis_tex_uv = self.visualize_3dmmtex_as_uv()
                torchvision.utils.save_image(torch.tensor(vis_img).permute(2, 0, 1).unsqueeze(0)/255.,
                                                os.path.join(rec_dir, 'stage1_vis.png'))
                torchvision.utils.save_image(torch.tensor(vis_tex_uv).permute(2, 0, 1).unsqueeze(0)/255.,
                                                os.path.join(rec_dir, 'stage1_vis_3dmmtex_as_uv.png'))
                self.save_mesh(path=rec_dir, mesh_name='stage1_mesh.obj', is_uv_tex=False)
                self.save_coeffs(path=rec_dir, coeffs_name='stage1_coeffs.pt', is_uv_tex=False)

            #--------- Stage 2 - using infer_uv to save mesh ---------
            if uv_map is not None:
                self.pred_uv_map = uv_map
                self.infer_render()
                vis_img = self.visualize(input_data)
                vis_tex_uv = tensor2np(self.pred_uv_map[:1, :, :, :])
                torchvision.utils.save_image(torch.tensor(vis_img).permute(2, 0, 1).unsqueeze(0)/255., os.path.join(rec_dir, f'stage{stage}_vis.png'))
                torchvision.utils.save_image(torch.tensor(vis_tex_uv).permute(2, 0, 1).unsqueeze(0)/255., os.path.join(rec_dir, f'stage{stage}_uv.png'))
                self.save_mesh(path=rec_dir,
                            mesh_name=f'stage{stage}_mesh.obj',
                            mlt_name=f'stage{stage}_mesh.mtl',
                            uv_name=f'stage{stage}_uv.png')
                self.save_coeffs(path=rec_dir, coeffs_name=f'stage{stage}_coeffs.pt')
        else:
            self.pred_uv_map = uv_map
            vis_tex_uv = tensor2np(self.pred_uv_map[:1, :, :, :])
            self.save_id_mesh(path=rec_dir,
                            id_coeffs=id_coeffs,
                            mesh_name=f'stage{stage}_mesh.obj',
                            mlt_name=f'stage{stage}_mesh.mtl',
                            uv_name=f'stage{stage}_uv.png',
                            tex = vis_tex_uv)
            torchvision.utils.save_image(torch.tensor(vis_tex_uv).permute(2, 0, 1).unsqueeze(0)/255., os.path.join(rec_dir, f'stage{stage}_uv.png'))
            self.save_id_coeffs(path=rec_dir, 
                                id_coeffs=id_coeffs, 
                                coeffs_name=f'stage{stage}_coeffs.pt')
            
        
    def rec_opitming(self, input_data, rec_dir, uv_map_lr=None, uv_map_hr=None, logger=None, stage=1, is_opt=False):
        #--------- Stage 1 - getting initial coeffs by Deep3D NN inference ---------
        if 'incomlete' in input_data:
            input_data.pop('incomlete')
        if 'face_emb' in input_data:
            input_data.pop('face_emb')
        if 'fill_mask' in input_data:
            input_data.pop('fill_mask')

        logger.write_txt_log('Stage 1 getting initial coeffs by Deep3D NN inference.')
        with torch.no_grad():
            self.pred_coeffs = self.net_recon_deep3d(input_data['img'].to(self.device))
        self.infer_render(is_uv_tex=False)
        vis_img = self.visualize(input_data, is_uv_tex=False)
        vis_tex_uv = self.visualize_3dmmtex_as_uv()
        logger.write_disk_images([vis_img], [f'stage{stage}_vis'])
        logger.write_disk_images([vis_tex_uv], [f'stage{stage}_vis_3dmmtex_as_uv'])
        # torchvision.utils.save_image(torch.tensor(vis_img).permute(2, 0, 1).unsqueeze(0)/255.,
                                        # os.path.join(rec_dir, 'stage1_vis.png'))
        # torchvision.utils.save_image(torch.tensor(vis_tex_uv).permute(2, 0, 1).unsqueeze(0)/255.,
                                        # os.path.join(rec_dir, 'stage1_vis_3dmmtex_as_uv.png'))
        self.save_mesh(path=logger.vis_dir, mesh_name=f'stage{stage}_mesh.obj', is_uv_tex=False)
        self.save_coeffs(path=logger.vis_dir, coeffs_name=f'stage{stage}_coeffs.pt', is_uv_tex=False)


        #--------- Stage 2 - using infer_uv to save mesh ---------
        logger.write_txt_log('Start stage 2 using infer_uv to save mesh.')
        self.pred_uv_map = uv_map_hr
        self.infer_render()
        vis_img = self.visualize(input_data)
        vis_tex_uv = tensor2np(self.pred_uv_map[:1, :, :, :])
        logger.write_disk_images([vis_img], [f'stage{stage+1}_vis'])
        logger.write_disk_images([vis_tex_uv], [f'stage{stage+1}_uv'])
        self.save_mesh(path=logger.vis_dir,
                    mesh_name=f'stage{stage+1}_mesh.obj',
                    mlt_name=f'stage{stage+1}_mesh.mtl',
                    uv_name=f'stage={stage+1}_uv.png')
        self.save_coeffs(path=logger.vis_dir, coeffs_name=f'stage{stage+1}_coeffs.pt')

        logger.write_txt_log('End stage 2 using infer_uv to save mesh.')

        if is_opt:
            #--------- Stage 3 - search spherical/shape with fixed uv_tex ---------
            logger.write_txt_log('Start stage 3 search spherical/shape with fixed uv_tex.')

            fitter = spherical_shape_fixuvtex_fitter.Fitter(facemodel=self.facemodel,
                                                    uv_map_lr=uv_map_lr,
                                                    uv_map_hr=uv_map_hr,
                                                    renderer=self.renderer,
                                                    net_recog=self.net_recog,
                                                    net_vgg=self.net_vgg,
                                                    logger=logger,
                                                    input_data=input_data,
                                                    init_coeffs=self.pred_coeffs,
                                                    init_latents_z=None,
                                                    **self.args_s3_optimize_uvtex_shape_joint)

            self.pred_coeffs = fitter.iterate()

            logger.reset_prefix()
            logger.write_txt_log('End stage 3 search spherical/shape with fixed uv_tex.')
            self.infer_render()
            vis_img = self.visualize(input_data)
            logger.write_disk_images([vis_img], [f'stage{stage+2}_vis'])
            logger.write_disk_images([tensor2np(self.pred_uv_map[:1, :, :, :])], [f'stage{stage+2}_uv'])
            self.save_mesh(path=logger.vis_dir,
                        mesh_name=f'stage{stage+2}_mesh.obj',
                        mlt_name=f'stage{stage+2}_mesh.mtl',
                        uv_name=f'stage{stage+2}_uv.png')
            self.save_coeffs(path=logger.vis_dir, coeffs_name=f'stage{stage+2}_coeffs.pt')


            #--------- Stage 4 - save hr ---------
            logger.write_txt_log('Start stage 4 using infer_uv to save mesh.')
            self.pred_uv_map = uv_map_hr
            self.infer_render()
            vis_img = self.visualize(input_data)
            vis_tex_uv = tensor2np(self.pred_uv_map[:1, :, :, :])
            logger.write_disk_images([vis_img], [f'stage{stage+3}_vis'])
            logger.write_disk_images([vis_tex_uv], [f'stage{stage+3}_uv'])
            self.save_mesh(path=logger.vis_dir,
                        mesh_name=f'stage{stage+3}_mesh.obj',
                        mlt_name=f'stage{stage+3}_mesh.mtl',
                        uv_name=f'stage{stage+3}_uv.png')
            self.save_coeffs(path=logger.vis_dir, coeffs_name=f'stage{stage+3}_coeffs.pt')

            logger.write_txt_log('End stage 4using infer_uv to save mesh.')

    def rec_opitming_gr(self, input_data, rec_dir, uv_map_hr=None, stage=1, is_opt=False):
        #--------- Stage 1 - getting initial coeffs by Deep3D NN inference ---------
        if 'incomlete' in input_data:
            input_data.pop('incomlete')
        if 'face_emb' in input_data:
            input_data.pop('face_emb')
        if 'fill_mask' in input_data:
            input_data.pop('fill_mask')

        print('Stage 1 getting initial coeffs by Deep3D NN inference.')

        with torch.no_grad():
            self.pred_coeffs = self.net_recon_deep3d(input_data['img'].to(self.device))
        self.infer_render(is_uv_tex=False)
        vis_img = self.visualize(input_data, is_uv_tex=False)
        vis_tex_uv = self.visualize_3dmmtex_as_uv()
        torchvision.utils.save_image(torch.tensor(vis_img).permute(2, 0, 1).unsqueeze(0)/255.,
                                        os.path.join(rec_dir, f'stage{stage}_vis.png'))
        torchvision.utils.save_image(torch.tensor(vis_tex_uv).permute(2, 0, 1).unsqueeze(0)/255.,
                                        os.path.join(rec_dir, f'stage{stage}_vis_3dmmtex_as_uv.png'))
        self.save_mesh(path=rec_dir, mesh_name=f'stage{stage}_mesh.obj', is_uv_tex=False)
        self.save_coeffs(path=rec_dir, coeffs_name=f'stage{stage}_coeffs.pt', is_uv_tex=False)
        print('End stage 1 getting initial coeffs by Deep3D NN inference.')


        #--------- Stage 2 - using infer_uv to save mesh ---------
        # logger.write_txt_log('Start stage 2 using infer_uv to save mesh.')
        print('Start stage 2 using infer_uv to save mesh.')
        self.pred_uv_map = uv_map_hr
        self.infer_render()
        vis_img = self.visualize(input_data)
        vis_tex_uv = tensor2np(self.pred_uv_map[:1, :, :, :])
        torchvision.utils.save_image(torch.tensor(vis_img).permute(2, 0, 1).unsqueeze(0)/255.,
                                        os.path.join(rec_dir, f'stage{stage+1}_vis.png'))
        torchvision.utils.save_image(torch.tensor(vis_tex_uv).permute(2, 0, 1).unsqueeze(0)/255.,
                                        os.path.join(rec_dir, f'stage{stage+1}_uv.png'))
        self.save_mesh(path=rec_dir,
                    mesh_name=f'stage{stage+1}_mesh.obj',
                    mlt_name=f'stage{stage+1}_mesh.mtl',
                    uv_name=f'stage{stage+1}_uv.png',
                    tex = vis_tex_uv)
        self.save_coeffs(path=rec_dir, coeffs_name=f'stage{stage+1}_coeffs.pt')

        print('End stage 2 using infer_uv to save mesh.')

        if is_opt:
            #--------- Stage 3 - search spherical/shape with fixed uv_tex ---------
            print('Start stage 3 search spherical/shape with fixed uv_tex.')

            fitter = spherical_shape_fixuvtex_fitter.Fitter(facemodel=self.facemodel,
                                                    uv_map_lr=uv_map_lr,
                                                    uv_map_hr=uv_map_hr,
                                                    renderer=self.renderer,
                                                    net_recog=self.net_recog,
                                                    net_vgg=self.net_vgg,
                                                    logger=logger,
                                                    input_data=input_data,
                                                    init_coeffs=self.pred_coeffs,
                                                    init_latents_z=None,
                                                    **self.args_s3_optimize_uvtex_shape_joint)

            self.pred_coeffs = fitter.iterate()

            print('End stage 3 search spherical/shape with fixed uv_tex.')
            self.infer_render()
            vis_img = self.visualize(input_data)
            torchvision.utils.save_image(torch.tensor(vis_img).permute(2, 0, 1).unsqueeze(0)/255.,
                                        os.path.join(rec_dir, f'stage{stage+2}_vis.png'))

            torchvision.utils.save_image(torch.tensor(vis_tex_uv).permute(2, 0, 1).unsqueeze(0)/    255., os.path.join(rec_dir, f'stage{stage+2}_uv.png'))
            
            self.save_mesh(path=rec_dir,
                        mesh_name=f'stage{stage+2}_mesh.obj',
                        mlt_name=f'stage{stage+2}_mesh.mtl',
                        uv_name=f'stage{stage+2}_uv.png')
            self.save_coeffs(path=rec_dir, coeffs_name=f'stage{stage+2}_coeffs.pt')


            #--------- Stage 4 - save hr ---------
            print('Start stage 4 using infer_uv to save mesh.')
            self.pred_uv_map = uv_map_hr
            self.infer_render()
            vis_img = self.visualize(input_data)
            vis_tex_uv = tensor2np(self.pred_uv_map[:1, :, :, :])
            torchvision.utils.save_image(torch.tensor(vis_img).permute(2, 0, 1).unsqueeze(0)/255.,
                                        os.path.join(rec_dir, f'stage{stage+3}_vis.png'))
            torchvision.utils.save_image(torch.tensor(vis_tex_uv).permute(2, 0, 1).unsqueeze(0)/255., os.path.join(rec_dir, f'stage{stage+3}_uv.png')) 
            self.save_mesh(path=rec_dir,
                        mesh_name=f'stage{stage+3}_mesh.obj',
                        mlt_name=f'stage{stage+3}_mesh.mtl',
                        uv_name=f'stage{stage+3}_uv.png')
            self.save_coeffs(path=rec_dir, coeffs_name=f'stage{stage+3}_coeffs.pt')

            print('End stage 4using infer_uv to save mesh.')
        
        return 
        

    def rec_opitming_hr(self, input_data, rec_dir, uv_map_lr=None, uv_map_hr=None, logger=None, stage=1, is_opt=False):
        #--------- Stage 1 - getting initial coeffs by Deep3D NN inference ---------
        if 'incomplete' in input_data:
            input_data.pop('incomlete')
        if 'face_emb' in input_data:
            input_data.pop('face_emb')
        if 'fill_mask' in input_data:
            input_data.pop('fill_mask')

        logger.write_txt_log('Stage 1 getting initial coeffs by Deep3D NN inference.')
        with torch.no_grad():
            self.pred_coeffs = self.net_recon_deep3d(input_data['img'].to(self.device))
        self.infer_render_hr(is_uv_tex=False)
        vis_img = self.visualize(input_data, is_uv_tex=False)
        vis_tex_uv = self.visualize_3dmmtex_as_uv()
        logger.write_disk_images([vis_img], [f'stage{stage}_vis'])
        logger.write_disk_images([vis_tex_uv], [f'stage{stage}_vis_3dmmtex_as_uv'])
        # torchvision.utils.save_image(torch.tensor(vis_img).permute(2, 0, 1).unsqueeze(0)/255.,
                                        # os.path.join(rec_dir, 'stage1_vis.png'))
        # torchvision.utils.save_image(torch.tensor(vis_tex_uv).permute(2, 0, 1).unsqueeze(0)/255.,
                                        # os.path.join(rec_dir, 'stage1_vis_3dmmtex_as_uv.png'))
        self.save_mesh(path=logger.vis_dir, mesh_name=f'stage{stage}_mesh.obj', is_uv_tex=False)
        self.save_coeffs(path=logger.vis_dir, coeffs_name=f'stage{stage}_coeffs.pt', is_uv_tex=False)


        #--------- Stage 2 - using infer_uv to save mesh ---------
        logger.write_txt_log('Start stage 2 using infer_uv to save mesh.')
        self.pred_uv_map = uv_map_hr
        self.infer_render_hr()
        vis_img = self.visualize(input_data)
        vis_tex_uv = tensor2np(self.pred_uv_map[:1, :, :, :])
        logger.write_disk_images([vis_img], [f'stage{stage+1}_vis'])
        logger.write_disk_images([vis_tex_uv], [f'stage{stage+1}_uv'])
        self.save_mesh(path=logger.vis_dir,
                    mesh_name=f'stage{stage+1}_mesh.obj',
                    mlt_name=f'stage{stage+1}_mesh.mtl',
                    uv_name=f'stage={stage+1}_uv.png')
        self.save_coeffs(path=logger.vis_dir, coeffs_name=f'stage{stage+1}_coeffs.pt')

        logger.write_txt_log('End stage 2 using infer_uv to save mesh.')

        if is_opt:
            #--------- Stage 3 - search spherical/shape with fixed uv_tex ---------
            logger.write_txt_log('Start stage 3 search spherical/shape with fixed uv_tex.')

            fitter = spherical_shape_fixuvtex_fitter.Fitter(facemodel=self.facemodel,
                                                    uv_map_lr=uv_map_lr,
                                                    uv_map_hr=uv_map_hr,
                                                    renderer=self.renderer,
                                                    net_recog=self.net_recog,
                                                    net_vgg=self.net_vgg,
                                                    logger=logger,
                                                    input_data=input_data,
                                                    init_coeffs=self.pred_coeffs,
                                                    init_latents_z=None,
                                                    **self.args_s3_optimize_uvtex_shape_joint)

            self.pred_coeffs = fitter.iterate()

            logger.reset_prefix()
            logger.write_txt_log('End stage 3 search spherical/shape with fixed uv_tex.')
            self.infer_render_hr()
            vis_img = self.visualize(input_data)
            logger.write_disk_images([vis_img], [f'stage{stage+2}_vis'])
            logger.write_disk_images([tensor2np(self.pred_uv_map[:1, :, :, :])], [f'stage{stage+2}_uv'])
            self.save_mesh(path=logger.vis_dir,
                        mesh_name=f'stage{stage+2}_mesh.obj',
                        mlt_name=f'stage{stage+2}_mesh.mtl',
                        uv_name=f'stage{stage+2}_uv.png')
            self.save_coeffs(path=logger.vis_dir, coeffs_name=f'stage{stage+2}_coeffs.pt')


            #--------- Stage 4 - save hr ---------
            logger.write_txt_log('Start stage 4 using infer_uv to save mesh.')
            self.pred_uv_map = uv_map_hr
            self.infer_render_hr()
            vis_img = self.visualize(input_data)
            vis_tex_uv = tensor2np(self.pred_uv_map[:1, :, :, :])
            logger.write_disk_images([vis_img], [f'stage{stage+3}_vis'])
            logger.write_disk_images([vis_tex_uv], [f'stage{stage+3}_uv'])
            self.save_mesh(path=logger.vis_dir,
                        mesh_name=f'stage{stage+3}_mesh.obj',
                        mlt_name=f'stage{stage+3}_mesh.mtl',
                        uv_name=f'stage{stage+3}_uv.png')
            self.save_coeffs(path=logger.vis_dir, coeffs_name=f'stage{stage+3}_coeffs.pt')

            logger.write_txt_log('End stage 4using infer_uv to save mesh.')





 


    
