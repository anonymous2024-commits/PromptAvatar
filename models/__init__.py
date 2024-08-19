from DataSet_Step4_UV_Texture.tex.tex_func import remap_tex_from_input2D
from DataSet_Step4_UV_Texture.face3d_recon import Face3d_Recon_API
from DataSet_Step4_UV_Texture.preprocess.preprocess_func import (extract_lm5_from_lm68, POS, 
                                                                 resize_crop_img, resize_crop_img_retain_hr,trans_projXY_back_to_ori_coord)

from RGB_Fitting.third_party import SkinMask_API, FaceParsing_API
from RGB_Fitting.utils.data_utils import read_img, img3channel, img2mask, np2pillow, pillow2np, np2tensor, tensor2np, img3channel, draw_mask, draw_landmarks, save_img
from RGB_Fitting.utils.preprocess_utils import align_img, estimate_norm
from RGB_Fitting.model import ours_fit_model
from insightface.app import FaceAnalysis
from PIL import Image
from diffusers.utils import load_image
import cv2
import numpy as np
import torch
import os
import math
from scipy.io import loadmat


class Tex_API:

    def __init__(self,
                 unwrap_info_path,
                 unwrap_info_mask_path,
                 unwrap_size=1024):
        '''
        Args:
            unwrap_info_path: str. The file path of unwrap information.
            unwrap_info_mask_path: str. The file path of unwrap mask.
            unwrap_size: int. The image size of unwrap texture map.
        '''

        assert unwrap_size == 1024

        # unwrap information
        unwrap_info = loadmat(unwrap_info_path)
        self.unwrap_uv_idx_bw = unwrap_info['uv_idx_bw'].astype(np.float32)
        self.unwrap_uv_idx_v_idx = unwrap_info['uv_idx_v_idx'].astype(np.float32)
        self.unwrap_info_mask = read_img(unwrap_info_mask_path, resize=(unwrap_size, unwrap_size), dst_range=1.)
        self.unwrap_uv_idx_bw = self.unwrap_uv_idx_bw * self.unwrap_info_mask
        self.unwrap_uv_idx_v_idx = self.unwrap_uv_idx_v_idx * self.unwrap_info_mask

    def __call__(self, img, seg_mask, projXY, norm):
        # remap texture from input 2D image to UV map
        remap_tex, remap_mask = remap_tex_from_input2D(input_img=img,
                                                    seg_mask=seg_mask,
                                                    projXY=projXY,
                                                    norm=norm,
                                                    unwrap_uv_idx_v_idx=self.unwrap_uv_idx_v_idx,
                                                    unwrap_uv_idx_bw=self.unwrap_uv_idx_bw)

        save_remap_masks = {
            'remap_mask': remap_mask[..., 0],
        }
        
        return remap_tex, save_remap_masks


class FaceRecognition_API:
    def __init__(self, face_analysis_model_path=None):
        self.app = FaceAnalysis(name='antelopev2', root=face_analysis_model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.top, self.bottom, self.left, self.right = 100, 100, 100, 100
        self.color = [128, 128, 128]

    def __call__(self, face_image):
        padded_image = cv2.copyMakeBorder(np.array(face_image), self.top, self.bottom, self.left, self.right, cv2.BORDER_CONSTANT, value=self.color)
        face_info = self.app.get(padded_image, cv2.COLOR_RGB2BGR)
        if len(face_info) == 0:
            print("No face detected")
            return None, None
        face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*x['bbox'][3]-x['bbox'][1])[-1]
        face_emb = face_info['embedding']
        padded_lm = face_info['landmark_3d_68'][:,:2] #padded_lm = face_info['kps']
        raw_lm = np.array(padded_lm) - np.array([self.left, self.top])
        raw_lm[:,1] = face_image.size[1] - 1 - raw_lm[:,1]
        return face_emb, raw_lm


class FitDataset_mask:
    def __init__(self, parsing_model_pth, parsing_resnet18_path, lm68_3d_path,
                 unwrap_info_path, unwrap_info_mask_path, pfm_model_path,
                 recon_model_path, face_analysis_model_path, focal, camera_distance,
                 batch_size, device, 
                 eye_mask_path, nose_mask_path, mouth_mask_path, face_mask_path,
                 unwrap_size = 1024,
                 target_size=224,
                 rescale_factor=102.,
                 uv_mask_size=512):
        self.face_recognition = FaceRecognition_API(face_analysis_model_path=face_analysis_model_path)
        self.skin_model = SkinMask_API()
        self.parsing_model = FaceParsing_API(parsing_pth=parsing_model_pth,
                                             resnet18_path=parsing_resnet18_path,
                                             device=device)
        self.tex_model = Tex_API(unwrap_info_path=unwrap_info_path,
                                 unwrap_info_mask_path=unwrap_info_mask_path,
                                 unwrap_size=unwrap_size)
        self.face3d_model = Face3d_Recon_API(pfm_model_path=pfm_model_path,
                                    recon_model_path=recon_model_path,
                                    focal=focal,
                                    camera_distance=camera_distance,
                                    device=device)
        self.lm68_3d = loadmat(lm68_3d_path)['lm']
        self.batch_size = batch_size
        self.device = device
        self.target_size = target_size
        self.rescale_factor = rescale_factor
        
        eye_mask = read_img(eye_mask_path, resize=(uv_mask_size, uv_mask_size), dst_range=1.)
        nose_mask = read_img(nose_mask_path, resize=(uv_mask_size, uv_mask_size), dst_range=1.)
        mouth_mask = read_img(mouth_mask_path, resize=(uv_mask_size, uv_mask_size), dst_range=1.)
        face_mask = read_img(face_mask_path, resize=(uv_mask_size, uv_mask_size), dst_range=1.)

        # kernel = np.ones((5, 5), np.uint8)

        # self.eye_mask = cv2.dilate(eye_mask, kernel, iterations=1)
        self.eye_mask = eye_mask
        self.nose_mask = nose_mask
        # self.mouth_mask = cv2.dilate(mouth_mask, kernel, iterations=1)
        # self.face_mask = cv2.dilate(face_mask, kernel, iterations=3)
        self.face_mask = face_mask
        self.uv_mask_size = uv_mask_size

    def trans_projXY_back_to_ori_coord(self, projXY, trans_params):
        ''' Transfer project XY coordinates from (224x224) back to (w0 x h0).

        Args:
            projXY: numpy.array, (N, 2). The project XY coordinates (224x224).
            trans_params: numpy.array, (6). Contains w0, h0, s, t0, t1, target_size.
        Returns:
            projXY_ori: numpy.array, (N, 2). The project XY coordinates (w0 x h0).
        '''
        return trans_projXY_back_to_ori_coord(projXY, trans_params)
    
    def get_input_data(self, img_path):            
        with torch.no_grad():
            input_img = read_img(img_path)
            raw_img = np2pillow(input_img)

            # detect 68 landmarks
            # raw_lm = self.lm68_model(input_img)
            face_emb, raw_lm = self.face_recognition(raw_img)
            if raw_lm is None:
                return None
                
            raw_lm = raw_lm.astype(np.float32)

            # calculate skin attention mask
            raw_skin_mask = self.skin_model(input_img, return_uint8=True)
            raw_skin_mask = img3channel(raw_skin_mask)
            raw_skin_mask = np2pillow(raw_skin_mask)

            # face parsing mask
            require_part = ['face', 'l_eye', 'r_eye', 'mouth']
            seg_mask_dict, _ = self.parsing_model(input_img, require_part=require_part)
            face_mask = seg_mask_dict['face']
            
            ex_mouth_mask = 1 - seg_mask_dict['mouth']
            ex_eye_mask = 1 - img2mask(seg_mask_dict['l_eye'] + seg_mask_dict['r_eye'], thre=0.5)
            raw_parse_mask_numpy = face_mask * ex_mouth_mask * ex_eye_mask
            raw_parse_mask = np2pillow(raw_parse_mask_numpy, src_range=1.0)

            # alignment
            trans_params, img, lm, skin_mask, parse_mask = align_img(raw_img, raw_lm, self.lm68_3d, raw_skin_mask,
                                                                     raw_parse_mask)

            # PIL.Image -> numpy.array
            tar_img = pillow2np(img)
            hr_img = resize_crop_img_retain_hr(raw_img, trans_params)

            # PIL.Image -> numpy.array
            tar_img = pillow2np(tar_img)
            hr_img = pillow2np(hr_img)

            # Not detect faces
            if tar_img is None:
                return None
            
            # 3D face recon
            coeffs = self.face3d_model.pred_coeffs(np2tensor(tar_img, device=self.device))
            projXY, norm = self.face3d_model.compute_224projXY_norm_by_pin_hole(coeffs)
            projXY, norm = projXY[0].cpu().numpy(), norm[0].cpu().numpy()
            projXY = self.trans_projXY_back_to_ori_coord(projXY, trans_params)
            unwrap_uv_tex, save_remap_masks = self.tex_model(input_img, raw_parse_mask_numpy, projXY, norm)

            incomplete_pillow = np2pillow(unwrap_uv_tex, src_range=255.0)
            fill_mask = img2mask(save_remap_masks['remap_mask'], thre=0.5)
            fill_mask_pillow = np2pillow(fill_mask, src_range=1.0)

            # to tensor
            _, H = img.size
            M = estimate_norm(lm, H)
            img_tensor = np2tensor(pillow2np(img), device=self.device)
            skin_mask_tensor = np2tensor(pillow2np(skin_mask), device=self.device)[:, :1, :, :]
            parse_mask_tensor = np2tensor(pillow2np(parse_mask), device=self.device)[:, :1, :, :]
            lm_tensor = torch.tensor(np.array(lm).astype(np.float32)).unsqueeze(0).to(self.device)
            M_tensor = torch.tensor(np.array(M).astype(np.float32)).unsqueeze(0).to(self.device)

            return {
                'img': img_tensor,
                'skin_mask': skin_mask_tensor,
                'parse_mask': parse_mask_tensor,
                'lm': lm_tensor,
                'M': M_tensor,
                'trans_params': trans_params,
                'face_emb': face_emb,
                'incomplete': incomplete_pillow,
                'fill_mask': fill_mask_pillow,
            }
    
    def get_mask_from_uv(self, img_pillow):
        with torch.no_grad():
            resize_img =  img_pillow.resize((self.uv_mask_size, self.uv_mask_size)) # 512x512
            input_img = pillow2np(resize_img)

            # face parsing mask
            require_part = ['face', 'l_eye', 'r_eye', 'mouth', 'nose', 'l_lip', 'u_lip']
            seg_mask_dict, _ = self.parsing_model(input_img, require_part=require_part)
            face_mask = seg_mask_dict['face']
            
            ex_mouth_mask = 1 - img2mask(seg_mask_dict['mouth'], thre=0.5)
            # ex_eye_mask = 1 - img2mask(seg_mask_dict['l_eye'] + seg_mask_dict['r_eye'], thre=0.5)
            ex_lip_mask = 1 - img2mask(seg_mask_dict['l_lip'] + seg_mask_dict['u_lip'], thre=0.5)
            # ex_nose_mask = 1 - seg_mask_dict['nose']

            ex_eye_mask = 1 - img2mask(self.eye_mask, thre=0.5)
            ex_nose_mask = 1 - img2mask(self.nose_mask, thre=0.5)

            raw_parse_mask_numpy = self.face_mask * face_mask * ex_mouth_mask * ex_lip_mask * ex_nose_mask * ex_eye_mask
            raw_parse_mask = np2pillow(raw_parse_mask_numpy, src_range=1.0)
        return raw_parse_mask


class FitDataset:
    def __init__(self, parsing_model_pth, parsing_resnet18_path, lm68_3d_path,
                 unwrap_info_path, unwrap_info_mask_path, pfm_model_path,
                 recon_model_path, face_analysis_model_path, focal, camera_distance,
                 batch_size, device, 
                 unwrap_size = 1024,
                 target_size=224,
                 rescale_factor=102.):
        self.face_recognition = FaceRecognition_API(face_analysis_model_path)
        self.skin_model = SkinMask_API()
        self.parsing_model = FaceParsing_API(parsing_pth=parsing_model_pth,
                                             resnet18_path=parsing_resnet18_path,
                                             device=device)
        self.tex_model = Tex_API(unwrap_info_path=unwrap_info_path,
                                 unwrap_info_mask_path=unwrap_info_mask_path,
                                 unwrap_size=unwrap_size)
        self.face3d_model = Face3d_Recon_API(pfm_model_path=pfm_model_path,
                                    recon_model_path=recon_model_path,
                                    focal=focal,
                                    camera_distance=camera_distance,
                                    device=device)
        self.lm68_3d = loadmat(lm68_3d_path)['lm']
        self.batch_size = batch_size
        self.device = device
        self.target_size = target_size
        self.rescale_factor = rescale_factor

    def trans_projXY_back_to_ori_coord(self, projXY, trans_params):
        ''' Transfer project XY coordinates from (224x224) back to (w0 x h0).

        Args:
            projXY: numpy.array, (N, 2). The project XY coordinates (224x224).
            trans_params: numpy.array, (6). Contains w0, h0, s, t0, t1, target_size.
        Returns:
            projXY_ori: numpy.array, (N, 2). The project XY coordinates (w0 x h0).
        '''
        return trans_projXY_back_to_ori_coord(projXY, trans_params)
    
    def get_input_data(self, img_path):
        with torch.no_grad():
            if type(img_path) == str:
                input_img = read_img(img_path)
                raw_img = np2pillow(input_img)
            else:
                raw_img = img_path # pil image
                input_img = pillow2np(raw_img)

            # detect 68 landmarks
            # raw_lm = self.lm68_model(input_img)
            face_emb, raw_lm = self.face_recognition(raw_img)
            if raw_lm is None:
                return None
                
            raw_lm = raw_lm.astype(np.float32)

            # calculate skin attention mask
            raw_skin_mask = self.skin_model(input_img, return_uint8=True)
            raw_skin_mask = img3channel(raw_skin_mask)
            raw_skin_mask = np2pillow(raw_skin_mask)

            # face parsing mask
            require_part = ['face', 'l_eye', 'r_eye', 'mouth', 'eye_g']
            seg_mask_dict, _ = self.parsing_model(input_img, require_part=require_part)
            face_mask = seg_mask_dict['face']
            
            ex_mouth_mask = 1 - seg_mask_dict['mouth']
            ex_eye_mask = 1 - img2mask(seg_mask_dict['l_eye'] + seg_mask_dict['r_eye'], thre=0.5)
            ex_glasses_mask = 1 - img2mask(seg_mask_dict['eye_g'], thre=0.5)
            raw_parse_mask_numpy = face_mask * ex_mouth_mask * ex_eye_mask * ex_glasses_mask
            raw_parse_mask = np2pillow(raw_parse_mask_numpy, src_range=1.0)

            # alignment
            trans_params, img, lm, skin_mask, parse_mask = align_img(raw_img, raw_lm, self.lm68_3d, raw_skin_mask,
                                                                     raw_parse_mask)

            # PIL.Image -> numpy.array
            tar_img = pillow2np(img)
            hr_img = resize_crop_img_retain_hr(raw_img, trans_params)

            # PIL.Image -> numpy.array
            tar_img = pillow2np(tar_img)
            hr_img = pillow2np(hr_img)

            # Not detect faces
            if tar_img is None:
                return None
            
            # 3D face recon
            coeffs = self.face3d_model.pred_coeffs(np2tensor(tar_img, device=self.device))
            projXY, norm = self.face3d_model.compute_224projXY_norm_by_pin_hole(coeffs)
            projXY, norm = projXY[0].cpu().numpy(), norm[0].cpu().numpy()
            projXY = self.trans_projXY_back_to_ori_coord(projXY, trans_params)
            unwrap_uv_tex, save_remap_masks = self.tex_model(input_img, raw_parse_mask_numpy, projXY, norm)

            incomplete_pillow = np2pillow(unwrap_uv_tex, src_range=255.0)
            fill_mask = img2mask(save_remap_masks['remap_mask'], thre=0.5)
            fill_mask_pillow = np2pillow(fill_mask, src_range=1.0)

            # to tensor
            _, H = img.size
            M = estimate_norm(lm, H)
            img_tensor = np2tensor(pillow2np(img), device=self.device)
            skin_mask_tensor = np2tensor(pillow2np(skin_mask), device=self.device)[:, :1, :, :]
            parse_mask_tensor = np2tensor(pillow2np(parse_mask), device=self.device)[:, :1, :, :]
            lm_tensor = torch.tensor(np.array(lm).astype(np.float32)).unsqueeze(0).to(self.device)
            M_tensor = torch.tensor(np.array(M).astype(np.float32)).unsqueeze(0).to(self.device)

            return {
                'img': img_tensor,
                'skin_mask': skin_mask_tensor,
                'parse_mask': parse_mask_tensor,
                'lm': lm_tensor,
                'M': M_tensor,
                'trans_params': trans_params,
                'face_emb': face_emb,
                'incomplete': incomplete_pillow,
                'fill_mask': fill_mask_pillow,
            }