
import os
from os.path import dirname, join, realpath
from typing import List

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh
from copy import deepcopy
from pytorch3d.ops import knn_points
from pytorch3d.transforms import matrix_to_quaternion

import sys
sys.path.append(realpath(join(dirname(__file__), "..", )))
from utils.smplx import SMPL, SMPLX

from util import load_obj_mesh


def get_face_per_pixel(mask, flist):
    """
    :param mask: the uv_mask returned from posmap renderer, where -1 stands for background
                 pixels in the uv map, where other value (int) is the face index that this
                 pixel point corresponds to.
    :param flist: the face list of the body model,
        - smpl, it should be an [13776, 3] array
        - smplx, it should be an [20908,3] array
    :return:
        flist_uv: an [img_size, img_size, 3] array, each pixel is the index of the 3 verts that belong to the triangle
    Note: we set all background (-1) pixels to be 0 to make it easy to parralelize, but later we
        will just mask out these pixels, so it's fine that they are wrong.
    """
    mask2 = mask.clone()
    mask2[
        mask == -1
    ] = 0  # remove the -1 in the mask, so that all mask elements can be seen as meaningful faceid
    flist_uv = flist[mask2]
    return flist_uv


def getIdxMap_torch(img, offset=False):
    # img has shape [channels, H, W]
    C, H, W = img.shape
    import torch

    idx = torch.stack(torch.where(~torch.isnan(img[0])))
    if offset:
        idx = idx.float() + 0.5
    idx = idx.view(2, H * W).float().contiguous()
    idx = idx.transpose(0, 1)

    idx = idx / (H - 1) if not offset else idx / H
    return idx


def load_masks(PROJECT_DIR, posmap_size, body_model="smpl"):
    uv_mask_faceid = np.load(
        join(
            PROJECT_DIR,
            "assets",
            "uv_masks",
            "uv_mask{}_with_faceid_{}.npy".format(posmap_size, body_model),
        )
    ).reshape(posmap_size, posmap_size)
    uv_mask_faceid = torch.from_numpy(uv_mask_faceid).long()

    smpl_faces = np.load(
        join(PROJECT_DIR, "assets", "{}_faces.npy".format(body_model.lower()))
    )  # faces = triangle list of the body mesh
    flist = torch.tensor(smpl_faces.astype(np.int32)).long()
    flist_uv = get_face_per_pixel(
        uv_mask_faceid, flist
    )  # Each (valid) pixel on the uv map corresponds to a point on the SMPL body; flist_uv is a list of these triangles

    points_idx_from_posmap = (uv_mask_faceid != -1).reshape(-1)

    uv_coord_map = getIdxMap_torch(torch.rand(3, posmap_size, posmap_size))
    uv_coord_map.requires_grad = True

    return flist_uv, points_idx_from_posmap, uv_coord_map


def init_xyz_on_mesh(v_init, faces, subdivide_num):
    # * xyz
    denser_v, denser_f = v_init.detach().cpu().numpy(), faces
    for i in range(subdivide_num):
        denser_v, denser_f = trimesh.remesh.subdivide(denser_v, denser_f)
    body_mesh = trimesh.Trimesh(denser_v, denser_f, process=False)
    v_init = torch.as_tensor(denser_v, dtype=torch.float32)
    return v_init, body_mesh


def init_qso_on_mesh(
    body_mesh,
    scale_init_factor,
    thickness_init_factor,
    max_scale,
    min_scale,
    s_inv_act,
    opacity_base_logit,
):
    # * Quaternion
    # each column is a basis vector
    # the local frame is z to normal, xy on the disk
    normal = body_mesh.vertex_normals.copy()
    v_init = torch.as_tensor(body_mesh.vertices.copy())
    faces = torch.as_tensor(body_mesh.faces.copy())

    uz = torch.as_tensor(normal, dtype=torch.float32)
    rand_dir = torch.randn_like(uz)
    ux = F.normalize(torch.cross(uz, rand_dir, dim=-1), dim=-1)
    uy = F.normalize(torch.cross(uz, ux, dim=-1), dim=-1)
    frame = torch.stack([ux, uy, uz], dim=-1)  # N,3,3
    ret_q = matrix_to_quaternion(frame)

    # * Scaling
    xy = v_init[faces[:, 1]] - v_init[faces[:, 0]]
    xz = v_init[faces[:, 2]] - v_init[faces[:, 0]]
    area = torch.norm(torch.cross(xy, xz, dim=-1), dim=-1) / 2
    vtx_nn_area = torch.zeros_like(v_init[:, 0])
    for i in range(3):
        vtx_nn_area.scatter_add_(0, faces[:, i], area / 3.0)
    radius = torch.sqrt(vtx_nn_area / np.pi)
    # radius = torch.clamp(radius * scale_init_factor, max=max_scale, min=min_scale)
    # ! 2023.11.22, small eps
    radius = torch.clamp(
        radius * scale_init_factor, max=max_scale - 1e-4, min=min_scale + 1e-4
    )
    thickness = radius * thickness_init_factor
    # ! 2023.11.22, small eps
    thickness = torch.clamp(thickness, max=max_scale - 1e-4, min=min_scale + 1e-4)
    radius_logit = s_inv_act(radius)
    thickness_logit = s_inv_act(thickness)
    ret_s = torch.stack([radius_logit, radius_logit, thickness_logit], dim=-1)

    ret_o = torch.ones_like(v_init[:, :1]) * opacity_base_logit
    return ret_q, ret_s, ret_o


class SMPL_Guidance:

    def __init__(
        self,
        smpl_model_path: str = "./data/smpl_related/models/",
        smpl_type: str = "smplx",
        gender: str = "male",
        seq: str = "dance",
        data_type: str = "insav_wild",
        batch_size: int = 1,
        num_subdiv: int = 2,
    ) -> None:
        assert smpl_type in ["smplx", "smpl"]
        model_path = os.path.join(smpl_model_path, smpl_type)
        self.smpl_type = smpl_type
        self.device = torch.device("cuda")
        if smpl_type == "smpl":
            self.smpl_model = (
                SMPL(
                    model_path=model_path,
                    gender=gender,
                    batch_size=batch_size,
                )
                .cuda()
                .eval()
            )

            smpl_data = torch.load(f"data/dna_rendering/{seq}/train/smpl_parms.pth")
            self.smpl_parms = {
                "betas": smpl_data["beta"].cuda()[0],
                "body_pose": smpl_data["body_pose"].cuda()[..., 3:],
                "global_orient": smpl_data["body_pose"].cuda()[..., :3],
                "transl": smpl_data["trans"].cuda(),
            }

            verts, faces, uvs, uv_faces = load_obj_mesh(
                "custom/threestudio-smpl/assets/template_mesh_smpl_uv.obj",
                with_texture=True,
            )
            self.cano_mesh = {
                "verts": torch.from_numpy(verts).cuda().float(),
                "faces": torch.from_numpy(faces.astype(np.int32)).cuda(),
                "uvs": torch.from_numpy(uvs).cuda().float(),
                "uv_faces": torch.from_numpy(uv_faces.astype(np.int32)).cuda(),
            }

            leg_angle = 30
            smpl_cpose_param = torch.zeros(1, 72).cuda()
            smpl_cpose_param[:, 5] = leg_angle / 180 * math.pi
            smpl_cpose_param[:, 8] = -leg_angle / 180 * math.pi
            cano_smpl = self.smpl_model.forward(
                betas=self.smpl_parms["betas"][None],
                global_orient=smpl_cpose_param[:, :3],
                transl=torch.tensor([[0, 0.30, 0]]).cuda(),
                body_pose=smpl_cpose_param[:, 3:],
            )

            res = 512
            query_map = torch.from_numpy(
                np.load(
                    f"data/insav_wild/{self.cfg.seq}/train/query_posemap_{res}_cano_smpl.npz"
                )["posmap" + str(res)].reshape(-1, 3)
            ).cuda()
            self.query_map = query_map
            flist_uv, valid_idx, uv_coord_map = load_masks(
                "custom/threestudio-smpl",
                res,
                body_model="smpl",
            )
            self.uv_coord_map = uv_coord_map[None].cuda()
            query_lbs = torch.from_numpy(
                np.load(
                    f"custom/threestudio-smpl/assets/lbs_map_smpl_{res}.npy"
                ).reshape(
                    res * res,
                    24,
                )
            ).cuda()
            self.query_lbs = query_lbs[valid_idx, :][None]
            self.valid_idx = valid_idx

            self.inv_mats = torch.linalg.inv(cano_smpl.A)
            self.ori_lbs = self.smpl_model.lbs_weights[None]
            self.cano_vertices = cano_smpl.vertices[0]

            new_sampled_points, new_mesh = init_xyz_on_mesh(
                self.cano_vertices, faces, num_subdiv
            )
            init_q, init_s, init_o = init_qso_on_mesh(
                new_mesh,
                1.0,
                0.5,
                0.1,
                0.0,
                torch.sigmoid,
                torch.logit(torch.tensor(0.9)),
            )

            cano_deform_points = query_map[valid_idx, :].contiguous()[None]
            self.query_points = new_sampled_points[None].to("cuda")
            self.init_q = init_q.to("cuda")

            num_training_frames = len(self.smpl_parms["body_pose"])
            param = []
            self.pose = torch.nn.Embedding(
                num_training_frames,
                72,
                _weight=torch.cat(
                    [self.smpl_parms["global_orient"], self.smpl_parms["body_pose"]],
                    dim=-1,
                ),
                # self.smpl_parms["body_pose"],
                sparse=True,
            ).cuda()
            param += list(self.pose.parameters())

            self.transl = torch.nn.Embedding(
                num_training_frames,
                3,
                _weight=self.smpl_parms["transl"],
                sparse=True,
            ).cuda()
            param += list(self.transl.parameters())
            self.param = param
        else:
            self.smpl_model = (
                SMPLX(
                    model_path=model_path,
                    gender=gender,
                    batch_size=batch_size,
                    use_pca=False,
                    use_face_contour=True,
                )
                .cuda()
                .eval()
            )

            smpl_data = torch.load(
                f"./data/custom/{seq}/smplx/params.pth", map_location="cpu"
            )
            self.smpl_data_all = {"cam_00": smpl_data}

            self.smpl_parms = {
                "betas": smpl_data["betas"].cuda()[0],
                "body_pose": smpl_data["body_pose"].cuda().flatten(-2, -1),
                "global_orient": smpl_data["global_orient"].cuda(),
                "transl": smpl_data["transl"].cuda(),
                "left_hand_pose": smpl_data["left_hand_pose"].cuda().flatten(-2, -1),
                "right_hand_pose": smpl_data["right_hand_pose"].cuda().flatten(-2, -1),
                "jaw_pose": smpl_data["jaw_pose"].cuda(),
                "leye_pose": smpl_data["leye_pose"].cuda(),
                "reye_pose": smpl_data["reye_pose"].cuda(),
                "expression": smpl_data["expression"].cuda(),
            }

            verts, faces, uvs, uv_faces = load_obj_mesh(
                "soar/threestudio-soar/utils/assets/template_mesh_smplx_uv.obj",
                with_texture=True,
            )
            self.cano_mesh = {
                "verts": torch.from_numpy(verts).cuda().float(),
                "faces": torch.from_numpy(faces.astype(np.int32)).cuda(),
                "uvs": torch.from_numpy(uvs).cuda().float(),
                "uv_faces": torch.from_numpy(uv_faces.astype(np.int32)).cuda(),
            }

            leg_angle = 30
            smplx_cpose_param = torch.zeros(1, 165).cuda()
            smplx_cpose_param[:, 5] = leg_angle / 180 * math.pi
            smplx_cpose_param[:, 8] = -leg_angle / 180 * math.pi
            cano_smpl = self.smpl_model.forward(
                betas=self.smpl_parms["betas"][None],
                global_orient=smplx_cpose_param[:, :3],
                transl=torch.tensor([[0, 0.30, 0]]).cuda(),
                body_pose=smplx_cpose_param[:, 3 : 3 + 21 * 3],
            )
            # cano_smpl = tpose_out

            self.inv_mats = torch.linalg.inv(cano_smpl.A.detach())
            self.ori_lbs = self.smpl_model.lbs_weights[None]
            self.cano_vertices = cano_smpl.vertices[0].detach()

            new_sampled_points, new_mesh = init_xyz_on_mesh(
                self.cano_vertices, faces, num_subdiv
            )
            init_q, init_s, init_o = init_qso_on_mesh(
                new_mesh,
                1.0,
                0.5,
                0.1,
                0.0,
                torch.sigmoid,
                torch.logit(torch.tensor(0.9)),
            )

            self.query_points = new_sampled_points[None].to("cuda")


    def densify(self, factor=2):
        pass

    def load_smpl_param(self, path):
        assert self.cfg.smpl_type == "smpl", "Only support smpl model"
        smpl_params = dict(np.load(str(path)))
        print("smpl_params", smpl_params)
        if "thetas" in smpl_params:
            smpl_params["body_pose"] = smpl_params["thetas"][..., 3:]
            smpl_params["global_orient"] = smpl_params["thetas"][..., :3]

        print(
            "betas",
            smpl_params["betas"].shape,
            "body_pose",
            smpl_params["body_pose"].shape,
            "global_orient",
            smpl_params["global_orient"].shape,
            "transl",
            smpl_params["transl"].shape,
        )
        return {
            "betas": torch.from_numpy(smpl_params["betas"].astype(np.float32)).cuda(),
            "body_pose": torch.from_numpy(
                smpl_params["body_pose"].astype(np.float32)
            ).cuda(),
            "global_orient": torch.from_numpy(
                smpl_params["global_orient"].astype(np.float32)
            ).cuda(),
            "transl": torch.from_numpy(
                smpl_params["transl"].astype(np.float32)
            ).cuda(),  # + torch.tensor([0.0, 0.0, 3.0], device='cuda')
        }

    def __call__(
        self,
        points,
        smpl_parms=None,
        delta=None,
        **kwargs,
    ):

        cano_deform_point = self.query_points
        if delta is not None:
            cano_deform_point += delta[None]
        with torch.no_grad():

            live_smpl = self.smpl_model.forward(
                    betas=smpl_parms["betas"],
                    body_pose=smpl_parms["body_pose"],
                    global_orient=smpl_parms["global_orient"],
                    transl=smpl_parms["transl"],
                    **{
                        k: v
                        for k, v in smpl_parms.items()
                        if k
                        not in ["betas", "body_pose", "global_orient", "transl"]
                    },
                )
        cano2live_jnt_mats = torch.matmul(live_smpl.A, self.inv_mats)
        updated_weights = self.query_weights_smpl(points)[None].detach()

        pt_mats = torch.einsum("bnj,bjxy->bnxy", updated_weights, cano2live_jnt_mats)
        # ori_pt_mats = torch.einsum("bnj,bjxy->bnxy", self.ori_lbs, cano2live_jnt_mats)
        return pt_mats

    # ref: https://github.com/tijiang13/InstantAvatar/blob/master/instant_avatar/deformers/fast_snarf/deformer_torch.py
    def query_weights_smpl(self, x, smpl_verts=None, smpl_weights=None, K=30):
        smpl_verts = self.cano_vertices if smpl_verts is None else smpl_verts
        smpl_weights = self.ori_lbs if smpl_weights is None else smpl_weights
        assert (
            smpl_verts is not None and smpl_weights is not None
        ), "Please run forward() first."

        smpl_weights = smpl_weights.squeeze(0)

        my_dist, my_idx, _ = knn_points(
            x.unsqueeze(0), smpl_verts.unsqueeze(0).detach(), K=30
        )
        my_dist = my_dist.sqrt().squeeze(0).clamp_(0.0001, 1.0)
        my_idx = my_idx.squeeze(0)
        weights = smpl_weights[my_idx]
        ws = 1.0 / my_dist
        ws = ws / (ws.sum(-1)[..., None])
        weights = (ws[..., None] * weights).sum(-2)

        return weights
