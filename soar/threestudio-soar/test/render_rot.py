import tyro
import torch
import random
import numpy as np
from sdf_fields import HashMLPSDFField
from smpl import SMPL_Guidance
from diff_gaussian_rasterizer import DiffGaussianRasterizer
from util import get_cam_info_gaussian
from torchvision.utils import save_image
from pytorch3d.transforms import matrix_to_axis_angle, axis_angle_to_matrix
from transforms3d.euler import euler2mat
import os
from tqdm import tqdm
import imageio

class SurfelModel():
    def __init__(self, points, rots, occs, attribute_field, SMPL_model, colors=None, scales=None):
        self.points = points
        self.rots = rots
        self.occs = occs
        self.attribute_field = attribute_field
        self.smpl_guidance = SMPL_model
        self.active_sh_degree = 0
        self.config = torch.tensor([True, True, True, False], device=points.device).float()
        
        self.colors = colors
        self.scales = scales
        
    @property
    def get_xyz(self):
        return self.points
    
    @property
    def get_colors(self):
        return torch.sigmoid(self.colors)
    
    @property
    def get_scaling(self):
        return torch.exp(self.scales)

    @property
    def get_rotation(self):
        return self.rots

    @property
    def get_opacity(self):
        return torch.ones_like(self.points)
    
    @property
    def get_occ(self):
        return torch.sigmoid(self.occs)
    
class Camera():
    FoVx: torch.Tensor
    FoVy: torch.Tensor
    camera_center: torch.Tensor
    image_width: int
    image_height: int
    world_view_transform: torch.Tensor
    full_proj_transform: torch.Tensor
    prcppoint: torch.Tensor
    
    def __init__(self, FoVx, FoVy, image_width, image_height, world_view_transform, full_proj_transform, camera_center, prcppoint):
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_width = image_width
        self.image_height = image_height
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        self.camera_center = camera_center
        self.prcppoint = prcppoint

    def random_patch(self, h_size=float("inf"), w_size=float("inf")):
        h = self.image_height
        w = self.image_width
        h_size = min(h_size, h)
        w_size = min(w_size, w)
        h0 = random.randint(0, h - h_size)
        w0 = random.randint(0, w - w_size)
        h1 = h0 + h_size
        w1 = w0 + w_size
        return torch.tensor([h0, w0, h1, w1]).to(torch.float32).to("cuda")
    
def render(rasterizer, pc, w2c, K, W, H, smpl_parms):
    w2c_ = w2c.clone()
    w2c_[1:3] *= -1
    c2w = torch.inverse(w2c_)
    fovx = 2 * np.arctan(W / (2 * K[0, 0]))
    fovy = 2 * np.arctan(H / (2 * K[1, 1]))
    cx = K[0, 2]
    cy = K[1, 2]
    
    w2c_, proj, cam_p = get_cam_info_gaussian(
            c2w=c2w,
            fovx=fovx,
            fovy=fovy,
            znear=0.1,
            zfar=100,
            cxcy=(cx, cy),
            img_wh=(W, H),
        )
    viewpoint_cam = Camera(
        FoVx=fovx,
        FoVy=fovy,
        image_width=W,
        image_height=H,
        world_view_transform=w2c_,
        full_proj_transform=proj,
        camera_center=cam_p,
        prcppoint=torch.tensor([0.5, 0.5], device=w2c_.device),
    )
    # rasterizer = DiffGaussianRasterizer(use_explicit=False)
    out = rasterizer(pc, viewpoint_cam, bg_color=torch.tensor([1.0, 1.0, 1.0], device='cuda'), smpl_parms=smpl_parms)
    return out

@torch.no_grad()
def load_from_ckpt(seq_name: str, ckpt_path: str, data_type: str = 'custom',
                   gender:str = 'neutral', exp_name: str = 'test', 
                   ablation_name: str = 'rot_360', use_explicit=False):

    ckpt_dict = torch.load(ckpt_path, map_location="cpu")

    points = ckpt_dict["state_dict"]["geometry._xyz"].cuda()
    rots = ckpt_dict["state_dict"]["geometry._rotation"].cuda()
    occs = ckpt_dict["state_dict"]["geometry._occ"].cuda()
    colors = ckpt_dict["state_dict"]["geometry._colors"].cuda()
    scales = ckpt_dict["state_dict"]["geometry._scaling"].cuda()
    
    aabb = ckpt_dict['state_dict']['geometry.attribute_field.aabb']
    attribute_field = HashMLPSDFField(aabb).cuda()
    attribute_field_state_dict = {
        key: ckpt_dict["state_dict"]["geometry.attribute_field." + key]
        for key in attribute_field.state_dict()
    }
    attribute_field.load_state_dict(attribute_field_state_dict)
    
    SMPL_model = SMPL_Guidance(seq=seq_name, gender=gender, data_type=data_type)
    smpl_data = SMPL_model.smpl_data_all
    
    pc = SurfelModel(points, rots, occs, attribute_field, SMPL_model, colors=colors, scales=scales)
    rasterizer = DiffGaussianRasterizer(use_explicit=use_explicit)
    cams = smpl_data.keys()
    cam = 'cam_00'

    cam_out_dir_rgb = f"outputs/{exp_name}/{seq_name}/{ablation_name}/{str(cam).zfill(2)}/rgb"
    cam_out_dir_occ = f"outputs/{exp_name}/{seq_name}/{ablation_name}/{str(cam).zfill(2)}/occ"
    cam_out_dir_normal = f"outputs/{exp_name}/{seq_name}/{ablation_name}/{str(cam).zfill(2)}/normal"
    cam_out_dir_mask = f"outputs/{exp_name}/{seq_name}/{ablation_name}/{str(cam).zfill(2)}/mask"
    os.makedirs(cam_out_dir_rgb, exist_ok=True)
    os.makedirs(cam_out_dir_occ, exist_ok=True)
    os.makedirs(cam_out_dir_normal, exist_ok=True)
    os.makedirs(cam_out_dir_mask, exist_ok=True)
    
    zero_trans = torch.zeros(3).cuda()
    zero_trans[-1] = 50.0 
    frames = []
    frames_normal = []

    for i in tqdm(range(36)):
        first_R = axis_angle_to_matrix(smpl_data[cam]['global_orient'][0]).detach().cpu().numpy()
        #rotation = euler2mat(2 * np.pi * vid / 10, 0.0, 0.0, "syxz")
        rotation = euler2mat(2 * np.pi * i / 36, 0.0, 0.0, "syxz")
        rotation = torch.from_numpy(first_R @rotation).float().cuda()
        global_orient = matrix_to_axis_angle(rotation[None])[0]

        smpl_parms = {
            "betas": smpl_data[cam]['betas'][[0]].cuda(),
            "body_pose": smpl_data[cam]['body_pose'][0][None].cuda(),
            "global_orient": global_orient[None].cuda(),
            "transl": smpl_data[cam]['transl'][0][None].cuda(),
            "left_hand_pose": smpl_data[cam]['left_hand_pose'][0][None].cuda(),
            "right_hand_pose": smpl_data[cam]['right_hand_pose'][0][None].cuda(),
            "jaw_pose": smpl_data[cam]['jaw_pose'][0][None].cuda(),
            "expression": smpl_data[cam]['expression'][0][None].cuda(),
            "leye_pose": smpl_data[cam]['leye_pose'][0][None].cuda(),
            "reye_pose": smpl_data[cam]['reye_pose'][0][None].cuda(),
        }
        out = render(rasterizer, pc, smpl_data[cam]['w2c'], smpl_data[cam]['Ks'][0 ], smpl_data[cam]['img_wh'][0], smpl_data[cam]['img_wh'][1], smpl_parms)
        out_rgb = torch.cat([out['render'], out['mask']], dim=0)
        out_normal = torch.cat([out['normal'], out['mask']], dim=0)
        out_occ = torch.cat([out['occ'], out['mask']], dim=0)
        frames.append(out_rgb.detach().cpu())
        frames_normal.append(out_normal.detach().cpu())
        
        save_image(out_rgb, os.path.join(cam_out_dir_rgb, f"{str(i).zfill(5)}.png"))
        save_image(out_normal, os.path.join(cam_out_dir_normal, f"{str(i).zfill(5)}.png"))
        save_image(out_occ, os.path.join(cam_out_dir_occ, f"{str(i).zfill(5)}.png"))
        save_image(out['mask'], os.path.join(cam_out_dir_mask, f"{str(i).zfill(5)}.png"))
    
    with imageio.get_writer(os.path.join(cam_out_dir_rgb, "video.mp4"), mode='I', fps=25) as writer:
        for i in range(36):
            writer.append_data((frames[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
    with imageio.get_writer(os.path.join(cam_out_dir_normal, "video.mp4"), mode='I', fps=25) as writer:
        for i in range(36):
            writer.append_data((frames_normal[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
     
    return


def main(seq_name: str, ckpt_path: str, data_type: str = 'custom', gender: str = 'neutral', exp_name: str = 'test', ablation_name: str = 'rot_360', use_explicit: bool = False):
    load_from_ckpt(seq_name, ckpt_path, data_type, gender, exp_name, ablation_name, use_explicit)

if __name__  == '__main__':
    tyro.cli(main)