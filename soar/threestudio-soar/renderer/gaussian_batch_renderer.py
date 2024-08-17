import math
import random

import numpy as np
import torch
from torch.cuda.amp import autocast
from torchvision.utils import save_image

from threestudio.utils.ops import get_cam_info_gaussian

from ..geometry.gaussian_base import BasicPointCloud, Camera


class GaussianBatchRenderer:
    def gt_forward(self, batch, mode="full", stage=0):
        renders = []
        viewspace_points = []
        visibility_filters = []
        radiis = []
        normals = []
        pred_normals = []
        depths = []
        curvs = []
        masks = []
        normal_masks = []
        occs = []
        comp_bgs = []
        fovx = batch["gt_fovx"]
        fovy = batch["gt_fovy"]
        c2w = batch["gt_c2w"][0]

        normal_fovx = batch["gt_normal_fovx"]
        normal_fovy = batch["gt_normal_fovy"]
        w2c, proj, cam_p = get_cam_info_gaussian_cxcy(
            c2w=c2w,
            fovx=fovx,
            fovy=fovy,
            znear=0.1,
            zfar=100,
            #cxcy=(batch["gt_cx"][0].item(), batch["gt_cy"][0].item()),
            #img_wh=(batch["gt_width"], batch["gt_height"]),
        )

        # breakpoint()
        w2c_normal, proj_normal, cam_p_normal = get_cam_info_gaussian_cxcy(
            c2w=c2w,
            fovx=normal_fovx,
            fovy=normal_fovy,
            znear=0.1,
            zfar=100,
            cxcy=(batch["gt_normal_cx"][0].item(), batch["gt_normal_cy"][0].item()),
            #  cxcy=(256.0, 256.0),
            img_wh=(batch["gt_normal_res"], batch["gt_normal_res"]),
        )

        w2c_normal_B, proj_normal_B, cam_p_normal_B = get_cam_info_gaussian_cxcy(
            c2w=c2w,
            fovx=normal_fovx,
            fovy=normal_fovy,
            znear=0.1,
            zfar=100,
            cxcy=(batch["gt_normal_cx"][0].item(), batch["gt_normal_cy"][0].item()),
            #  cxcy=(256.0, 256.0),
            img_wh=(batch["gt_normal_res"], batch["gt_normal_res"]),
        )
        prcppoint = torch.tensor(
            [
                batch["gt_cx"][0].item() / batch["gt_width"],
                batch["gt_cy"][0].item() / batch["gt_height"],
            ],
            device=w2c.device,
        )
        prcppoint_normal = torch.tensor(
            [
                batch["gt_normal_cx"][0].item() / batch["gt_normal_res"],
                batch["gt_normal_cy"][0].item() / batch["gt_normal_res"],
                1.0,
            ],
            device=w2c.device,
        )
        viewpoint_cam = Camera(
            FoVx=fovx,
            FoVy=fovy,
            image_width=batch["gt_width"],
            image_height=batch["gt_height"],
            world_view_transform=w2c,
            full_proj_transform=proj,
            camera_center=cam_p,
            prcppoint=prcppoint, #torch.tensor([0.5, 0.5], device=w2c.device),
        )
        viewpoint_cam_normal = Camera(
            FoVx=normal_fovx,
            FoVy=normal_fovy,
            image_width=batch["gt_normal_res"],
            image_height=batch["gt_normal_res"],
            world_view_transform=w2c_normal,
            full_proj_transform=proj_normal,
            camera_center=cam_p_normal,
            prcppoint=torch.tensor([0.5, 0.5], device=w2c.device),
        )
        viewpoint_cam_normal_B = Camera(
            FoVx=normal_fovx,
            FoVy=normal_fovy,
            image_width=batch["gt_normal_res"],
            image_height=batch["gt_normal_res"],
            world_view_transform=w2c_normal_B,
            full_proj_transform=proj_normal_B,
            camera_center=cam_p_normal_B,
            prcppoint=torch.tensor([0.5, 0.5], device=w2c.device),
        )
        with autocast(enabled=False):
            render_pkg = self.forward(
                viewpoint_cam,
                batch["rand_bg_color"],
                #torch.ones_like(self.background_tensor),
                gt=True,
                mode=mode,
                stage=stage,
                **batch
            )
            renders.append(render_pkg["render"])
            viewspace_points.append(render_pkg["viewspace_points"])
            visibility_filters.append(render_pkg["visibility_filter"])
            radiis.append(render_pkg["radii"])
            # if render_pkg.__contains__("normal"):
            #     normals.append(render_pkg["normal"])
            # if (
            #     render_pkg.__contains__("pred_normal")
            #     and render_pkg["pred_normal"] is not None
            # ):
            #     pred_normals.append(render_pkg["pred_normal"])
            if render_pkg.__contains__("depth"):
                depths.append(render_pkg["depth"])
            if render_pkg.__contains__("mask"):
                masks.append(render_pkg["mask"])
            if render_pkg.__contains__("occ"):
                occs.append(render_pkg["occ"])
            if render_pkg.__contains__("curv"):
                curvs.append(render_pkg["curv"])
            if render_pkg.__contains__("comp_bg"):
                comp_bgs.append(render_pkg["comp_bg"])
            
            # if render_pkg.__contains__("normal"):
            #     normals.append(render_pkg["normal"])
            #     normals.append(render_pkg["normal"])
            # if (
            #     render_pkg.__contains__("pred_normal")
            #     and render_pkg["pred_normal"] is not None
            # ):
            #     pred_normals.append(render_pkg["pred_normal"])
            #     pred_normals.append(render_pkg["pred_normal"])
            # if render_pkg.__contains__("mask"):
            #     normal_masks.append(render_pkg["mask"])
            #     normal_masks.append(render_pkg["mask"])

        with autocast(enabled=False):
            render_pkg_normal = self.forward(
                viewpoint_cam_normal,
                self.background_tensor,
                gt=True,
                mode=mode,
                stage=stage,
                #  normal_crop=True,
                **batch
            )
            viewspace_points.append(render_pkg_normal["viewspace_points"])
            visibility_filters.append(render_pkg_normal["visibility_filter"])
            radiis.append(render_pkg_normal["radii"])
            if render_pkg_normal.__contains__("normal"):
                normals.append(render_pkg_normal["normal"])
            if (
                render_pkg_normal.__contains__("pred_normal")
                and render_pkg_normal["pred_normal"] is not None
            ):
                pred_normals.append(render_pkg_normal["pred_normal"])
            if render_pkg_normal.__contains__("mask"):
                normal_masks.append(render_pkg_normal["mask"])

        with autocast(enabled=False):
            render_pkg_normal_B = self.forward(
                viewpoint_cam_normal_B,
                self.background_tensor,
                gt=True,
                mode=mode,
                stage=stage,
                #  normal_crop=True,
                render_front=False,
                **batch
            )
            viewspace_points.append(render_pkg_normal_B["viewspace_points"])
            visibility_filters.append(render_pkg_normal_B["visibility_filter"])
            radiis.append(render_pkg_normal_B["radii"])
            if render_pkg_normal_B.__contains__("normal"):
                normals.append(render_pkg_normal_B["normal"])
            if (
                render_pkg_normal_B.__contains__("pred_normal")
                and render_pkg_normal_B["pred_normal"] is not None
            ):
                pred_normals.append(render_pkg_normal_B["pred_normal"])
            if render_pkg_normal_B.__contains__("mask"):
                normal_masks.append(render_pkg_normal_B["mask"])

        outputs = {
            "comp_rgb": torch.stack(renders, dim=0).permute(0, 2, 3, 1),
            "viewspace_points": viewspace_points,
            "visibility_filter": visibility_filters,
            "radii": radiis,
        }
        if len(normals) > 0:
            outputs.update(
                {
                    "comp_normal": torch.stack(normals, dim=0).permute(0, 2, 3, 1),
                }
            )
        if len(pred_normals) > 0:
            outputs.update(
                {
                    "comp_pred_normal": torch.stack(pred_normals, dim=0).permute(
                        0, 2, 3, 1
                    ),
                }
            )
        if len(depths) > 0:
            outputs.update(
                {
                    "comp_depth": torch.stack(depths, dim=0).permute(0, 2, 3, 1),
                }
            )
        if len(masks) > 0:
            outputs.update(
                {
                    "comp_mask": torch.stack(masks, dim=0).permute(0, 2, 3, 1),
                }
            )
        if len(normal_masks) > 0:
            outputs.update(
                {
                    "comp_normal_mask": torch.stack(normal_masks, dim=0).permute(
                        0, 2, 3, 1
                    ),
                }
            )
        if len(occs) > 0:
            outputs.update(
                {
                    "comp_occ": torch.stack(occs, dim=0).permute(0, 2, 3, 1),
                }
            )
        if len(curvs) > 0:
            outputs.update(
                {
                    "comp_curv": torch.stack(curvs, dim=0).permute(0, 2, 3, 1),
                }
            )
        if len(comp_bgs) > 0:
            outputs.update(
                {
                    "comp_bg": torch.stack(comp_bgs, dim=0),
                }
            )
        return outputs

    def batch_forward(self, batch, mode="full", head_flag=False, stage=0):
        bs = batch["c2w"].shape[0]
        renders = []
        viewspace_points = []
        visibility_filters = []
        radiis = []
        normals = []
        pred_normals = []
        depths = []
        curvs = []
        masks = []
        occs = []
        comp_bgs = []

        if "gt_c2w" and "gt_rays_d" in batch:
            # This is for the imagedream module
            rays_d_all = torch.cat([batch["rays_d"], batch["gt_rays_d"]], dim=0)
        else:
            rays_d_all = batch["rays_d"]
        comp_rgb_bg_all = self.background(dirs=rays_d_all)
        # head_flag = ("gt_c2w" in batch) and (random.random() < 0.4)
        T_ocam, fovy_deg = sample_camera(
            random_elevation_range=[-10.0, 20.0],
            camera_distance_range=[0.28, 0.28],
            relative_radius=True,
            # random_azimuth_range=[0.0,0.0],
            fovy_range=[30, 45],  # [15, 60],
            zoom_range=[1.0, 1.0],
        )
        batch["head_c2ws"] = []
        for batch_idx in range(bs):
            batch["batch_idx"] = batch_idx
            batch["head_c2w"] = T_ocam[batch_idx]
            batch["head_fovy"] = fovy_deg[batch_idx]
            fovy = batch["fovy"][batch_idx]

            w2c, proj, cam_p = get_cam_info_gaussian_cxcy(
                c2w=batch["c2w"][batch_idx], fovx=fovy, fovy=fovy, znear=0.1, zfar=100
            )

            # import pdb; pdb.set_trace()
            viewpoint_cam = Camera(
                FoVx=fovy,
                FoVy=fovy,
                image_width=batch["width"],
                image_height=batch["height"],
                world_view_transform=w2c,
                full_proj_transform=proj,
                camera_center=cam_p,
                prcppoint=torch.tensor([0.5, 0.5], device=w2c.device),
            )

            with autocast(enabled=False):
                render_pkg = self.forward(
                    viewpoint_cam,
                    torch.zeros_like(self.background_tensor) * 0.5,
                    mode=mode,
                    head_flag=head_flag,
                    stage=stage,
                    **batch
                )
                renders.append(render_pkg["render"])
                viewspace_points.append(render_pkg["viewspace_points"])
                visibility_filters.append(render_pkg["visibility_filter"])
                radiis.append(render_pkg["radii"])
                if render_pkg.__contains__("normal"):
                    normals.append(render_pkg["normal"])
                if (
                    render_pkg.__contains__("pred_normal")
                    and render_pkg["pred_normal"] is not None
                ):
                    pred_normals.append(render_pkg["pred_normal"])
                if render_pkg.__contains__("depth"):
                    depths.append(render_pkg["depth"])
                if render_pkg.__contains__("mask"):
                    masks.append(render_pkg["mask"])
                if render_pkg.__contains__("occ"):
                    occs.append(render_pkg["occ"])
                if render_pkg.__contains__("curv"):
                    curvs.append(render_pkg["curv"])
                if render_pkg.__contains__("comp_bg"):
                    comp_bgs.append(render_pkg["comp_bg"])
        # if head_flag:
        #     batch["head_c2ws"] = torch.stack(batch["head_c2ws"], dim=0)
        # breakpoint()
        renders = torch.stack(renders, dim=0)
        masks_tensor = torch.stack(masks, dim=0)

        # save_image(comp_rgb_bg_all.permute(0, 3, 1, 2), "comp_rgb_bg_all.png")
        rendered_rgb = renders + (1 - masks_tensor) * comp_rgb_bg_all[:bs].permute(
            0, 3, 1, 2
        )
        outputs = {
            "comp_rgb": rendered_rgb.permute(0, 2, 3, 1),
            "viewspace_points": viewspace_points,
            "visibility_filter": visibility_filters,
            "radii": radiis,
        }
        if len(normals) > 0:
            outputs.update(
                {
                    "comp_normal": torch.stack(normals, dim=0).permute(0, 2, 3, 1),
                }
            )
        if len(pred_normals) > 0:
            outputs.update(
                {
                    "comp_pred_normal": torch.stack(pred_normals, dim=0).permute(
                        0, 2, 3, 1
                    ),
                }
            )
        if len(depths) > 0:
            outputs.update(
                {
                    "comp_depth": torch.stack(depths, dim=0).permute(0, 2, 3, 1),
                }
            )
        if len(masks) > 0:
            outputs.update(
                {
                    "comp_mask": torch.stack(masks, dim=0).permute(0, 2, 3, 1),
                }
            )
        if len(occs) > 0:
            outputs.update(
                {
                    "comp_occ": torch.stack(occs, dim=0).permute(0, 2, 3, 1),
                }
            )
        if len(curvs) > 0:
            outputs.update(
                {
                    "comp_curv": torch.stack(curvs, dim=0).permute(0, 2, 3, 1),
                }
            )
        if len(comp_bgs) > 0:
            outputs.update(
                {
                    "comp_bg": torch.stack(comp_bgs, dim=0),
                }
            )

        if "gt_c2w" in batch:
            # breakpoint()
            # rand_bg = torch.ones_like(batch['gt_rgb']) * torch.rand(1, 3, 1, 1)
            rand_bg_color = torch.rand(3).to(batch["gt_rgb"].device)
            rand_bg = torch.ones_like(batch['gt_rgb']) * rand_bg_color
            batch["rand_bg_color"] = rand_bg_color
            gt_outputs = self.gt_forward(batch)
            gt_outputs["comp_bg"] = comp_rgb_bg_all[[-1]]
            gt_outputs['rand_bg'] = rand_bg
            # gt_outputs = outputs
            # outputs["viewspace_points"].extend(gt_outputs["viewspace_points"])
            # outputs["visibility_filter"].extend(gt_outputs["visibility_filter"])
            # outputs["radii"].extend(gt_outputs["radii"])

            return outputs, gt_outputs
        return outputs
    
# gaussian splatting functions
def convert_pose(C2W):
    flip_yz = torch.eye(4, device=C2W.device)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    C2W = torch.matmul(C2W, flip_yz)
    return C2W


def get_projection_matrix_gaussian(
    znear, zfar, fovX, fovY, device="cuda", cxcy=None, img_wh=None, z_sign=1.0
):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4, device=device)

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * (zfar + znear) / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    if cxcy is not None and img_wh is not None:
        cx, cy = cxcy
        W, H = img_wh
        P[0, 2] = (2.0 * cx - W) / W
        P[1, 2] = (2.0 * cy - H) / H
    else:
        P[0, 2] = (right + left) / (right - left)
        P[1, 2] = (top + bottom) / (top - bottom)
    # P[2] *= -1
    return P

def get_cam_info_gaussian_cxcy(
    c2w, fovx, fovy, znear, zfar, cxcy=None, img_wh=None, back=False
):
    c2w_converted = convert_pose(c2w)
    world_view_transform = torch.inverse(c2w_converted)

    world_view_transform = world_view_transform.transpose(0, 1).cuda().float()

    projection_matrix = get_projection_matrix_gaussian(
        znear=znear,
        zfar=zfar,
        fovX=fovx,
        fovY=fovy,
        cxcy=cxcy,
        img_wh=img_wh,
    )
    # if img_wh is not None:
    #     focal = img_wh[1] / (2 * torch.tan(torch.deg2rad(fovx) / 2))
    #     projection_matrix = torch.tensor([
    #                 [2*focal/img_wh[0], 0, 0, 0],
    #                 [0, -2*focal/img_wh[1], 0, 0],
    #                 [0, 0, -(zfar+znear)/(zfar-znear), -(2*zfar*znear)/(zfar-znear)],
    #                 [0, 0, -1, 0]
    #             ], dtype=torch.float32, device='cuda')
    if back:
        projection_matrix[2] *= -1
    projection_matrix = projection_matrix.transpose(0, 1).cuda()

    full_proj_transform = (
        world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))
    ).squeeze(0)
    camera_center = world_view_transform.inverse()[3, :3]

    return world_view_transform, full_proj_transform, camera_center

def sample_camera(
    global_step=1,
    n_view=4,
    real_batch_size=1,
    random_azimuth_range=[-180.0, 180.0],
    random_elevation_range=[0.0, 30.0],
    eval_elevation_deg=15,
    camera_distance_range=[0.8, 1.0],  # relative
    fovy_range=[15, 60],
    zoom_range=[1.0, 1.0],
    progressive_until=0,
    relative_radius=True,
):
    # camera_perturb = 0.0
    # center_perturb = 0.0
    # up_perturb: 0.0

    # ! from uncond.py
    # ThreeStudio has progressive increase of camera poses, from eval to random
    r = min(1.0, global_step / (progressive_until + 1))
    elevation_range = [
        (1 - r) * eval_elevation_deg + r * random_elevation_range[0],
        (1 - r) * eval_elevation_deg + r * random_elevation_range[1],
    ]
    azimuth_range = [
        (1 - r) * 0.0 + r * random_azimuth_range[0],
        (1 - r) * 0.0 + r * random_azimuth_range[1],
    ]

    # sample elevation angles
    if random.random() < 0.5:
        # sample elevation angles uniformly with a probability 0.5 (biased towards poles)
        elevation_deg = (
            torch.rand(real_batch_size) * (elevation_range[1] - elevation_range[0])
            + elevation_range[0]
        ).repeat_interleave(n_view, dim=0)
        elevation = elevation_deg * math.pi / 180
    else:
        # otherwise sample uniformly on sphere
        elevation_range_percent = [
            (elevation_range[0] + 90.0) / 180.0,
            (elevation_range[1] + 90.0) / 180.0,
        ]
        # inverse transform sampling
        elevation = torch.asin(
            2
            * (
                torch.rand(real_batch_size)
                * (elevation_range_percent[1] - elevation_range_percent[0])
                + elevation_range_percent[0]
            )
            - 1.0
        ).repeat_interleave(n_view, dim=0)
        elevation_deg = elevation / math.pi * 180.0

    # sample azimuth angles from a uniform distribution bounded by azimuth_range
    # ensures sampled azimuth angles in a batch cover the whole range
    azimuth_deg = (
        torch.rand(real_batch_size).reshape(-1, 1) + torch.arange(n_view).reshape(1, -1)
    ).reshape(-1) / n_view * (azimuth_range[1] - azimuth_range[0]) + azimuth_range[0]
    azimuth = azimuth_deg * math.pi / 180

    ######## Different from original ########
    # sample fovs from a uniform distribution bounded by fov_range
    fovy_deg = (
        torch.rand(real_batch_size) * (fovy_range[1] - fovy_range[0]) + fovy_range[0]
    ).repeat_interleave(n_view, dim=0)
    fovy = fovy_deg * math.pi / 180

    # sample distances from a uniform distribution bounded by distance_range
    camera_distances = (
        torch.rand(real_batch_size)
        * (camera_distance_range[1] - camera_distance_range[0])
        + camera_distance_range[0]
    ).repeat_interleave(n_view, dim=0)
    if relative_radius:
        scale = 1 / torch.tan(0.5 * fovy)
        camera_distances = scale * camera_distances

    # zoom in by decreasing fov after camera distance is fixed
    zoom = (
        torch.rand(real_batch_size) * (zoom_range[1] - zoom_range[0]) + zoom_range[0]
    ).repeat_interleave(n_view, dim=0)
    fovy = fovy * zoom
    fovy_deg = fovy_deg * zoom
    ###########################################

    # convert spherical coordinates to cartesian coordinates
    # right hand coordinate system, x back, y right, z up
    # elevation in (-90, 90), azimuth from +x to +y in (-180, 180)
    camera_positions = torch.stack(
        [
            camera_distances * torch.cos(elevation) * torch.cos(azimuth),
            camera_distances * torch.cos(elevation) * torch.sin(azimuth),
            camera_distances * torch.sin(elevation),
        ],
        dim=-1,
    )

    azimuth, elevation
    # build opencv camera
    z = -torch.stack(
        [
            torch.cos(elevation) * torch.cos(azimuth),
            torch.cos(elevation) * torch.sin(azimuth),
            torch.sin(elevation),
        ],
        -1,
    )  # nview, 3
    # up is 0,0,1
    x = torch.cross(
        z, torch.tensor([0.0, 0.0, 1.0], device=z.device).repeat(n_view, 1), -1
    )
    y = torch.cross(x, z, -1)

    R_wc = torch.stack([x, y, -z], dim=2)  # nview, 3, 3, col is basis
    t_wc = camera_positions

    T_wc = torch.eye(4, device=R_wc.device).repeat(n_view, 1, 1)
    T_wc[:, :3, :3] = R_wc
    T_wc[:, :3, 3] = t_wc

    return T_wc, fovy_deg  # B,4,4, B
