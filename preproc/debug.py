#!/usr/bin/env python3
#
# File   : debug.py
# Author : Hang Gao
# Email  : hangg.sv7@gmail.com
# Date   : 08/04/2024
#
# Distributed under terms of the MIT license.

import torch

params = torch.load("debug.pth", map_location="cpu", weights_only=True)
joints = params["joints"]
pred_joints = params["pred_joints"]
kp_mask = params["kp_mask"]


def prepare_smplx_to_openpose137():
    kp_mask = torch.tensor(
        [
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            0,
        ],
        dtype=torch.float32,
    )
    src_inds = [
        55,
        12,
        17,
        19,
        21,
        16,
        18,
        20,
        0,
        2,
        5,
        8,
        1,
        4,
        7,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        37,
        38,
        39,
        66,
        25,
        26,
        27,
        67,
        28,
        29,
        30,
        68,
        34,
        35,
        36,
        69,
        31,
        32,
        33,
        70,
        52,
        53,
        54,
        71,
        40,
        41,
        42,
        72,
        43,
        44,
        45,
        73,
        49,
        50,
        51,
        74,
        46,
        47,
        48,
        75,
        127,
        128,
        129,
        130,
        131,
        132,
        133,
        134,
        135,
        136,
        137,
        138,
        139,
        140,
        141,
        142,
        143,
        86,
        87,
        88,
        89,
        90,
        91,
        92,
        93,
        94,
        95,
        96,
        97,
        98,
        99,
        100,
        101,
        102,
        103,
        104,
        105,
        106,
        107,
        108,
        109,
        110,
        111,
        112,
        113,
        114,
        115,
        116,
        117,
        118,
        119,
        120,
        121,
        122,
        123,
        124,
        125,
        126,
    ]
    dst_inds = [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        45,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        66,
        67,
        68,
        69,
        70,
        71,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        80,
        81,
        82,
        83,
        94,
        95,
        96,
        97,
        98,
        99,
        100,
        101,
        102,
        103,
        104,
        105,
        106,
        107,
        108,
        109,
        110,
        111,
        112,
        113,
        114,
        115,
        116,
        117,
        118,
        119,
        120,
        121,
        122,
        123,
        124,
        125,
        126,
        127,
        128,
        129,
        130,
        131,
        132,
        133,
        134,
    ]

    def convert_kps(kps):
        assert kps.shape[1] == 144
        new_kps = kps.new_zeros((kps.shape[0], 137, 3))
        new_kps[:, dst_inds] = kps[:, src_inds]
        new_kps[:, 8] = 0.5 * (new_kps[:, 9] + new_kps[:, 12])
        new_kps[:, [9, 12], :2] = (
            new_kps[:, [9, 12], :2]
            + 0.25 * (new_kps[:, [9, 12], :2] - new_kps[:, [12, 9], :2])
            + 0.5
            * (
                new_kps[:, [8], :2]
                - 0.5 * (new_kps[:, [9, 12], :2] + new_kps[:, [12, 9], :2])
            )
        )
        return new_kps

    return convert_kps, kp_mask


convert_kps, kp_mask_2 = prepare_smplx_to_openpose137()
pred_joints_2 = convert_kps(joints)
print(torch.allclose(pred_joints, pred_joints_2))
print(torch.allclose(kp_mask.float(), kp_mask_2))
__import__("ipdb").set_trace()
