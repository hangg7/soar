/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#pragma once
#include <cstdio>
#include <string>
#include <torch/extension.h>
#include <tuple>

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
    const torch::Tensor &background, const torch::Tensor &means3D,
    const torch::Tensor &colors, const torch::Tensor &opacity,
    // torch::Tensor& cutoff,
    const torch::Tensor &scales, const torch::Tensor &rotations,
    const float scale_modifier, const torch::Tensor &cov3D_precomp,
    const torch::Tensor &viewmatrix, const torch::Tensor &projmatrix,
    const torch::Tensor &prcppoint, const torch::Tensor &patchbbox,
    const float tan_fovx, const float tan_fovy, const int image_height,
    const int image_width, const torch::Tensor &sh, const int degree,
    const torch::Tensor &campos, const bool prefiltered,
    const bool render_front, const bool sort_descending, const bool debug,
    const torch::Tensor &config);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansBackwardCUDA(
    const torch::Tensor &background, const torch::Tensor &means3D,
    const torch::Tensor &radii, const torch::Tensor &colors,
    const torch::Tensor &scales, const torch::Tensor &rotations,
    const float scale_modifier, const torch::Tensor &cov3D_precomp,
    const torch::Tensor &viewmatrix, const torch::Tensor &projmatrix,
    const torch::Tensor &prcppoint, const torch::Tensor &patchbbox,
    const float tan_fovx, const float tan_fovy,
    const torch::Tensor &dL_dout_color, const torch::Tensor &dL_dout_normal,
    const torch::Tensor &dL_dout_depth, const torch::Tensor &dL_dout_opac,
    const torch::Tensor &sh, const int degree, const torch::Tensor &campos,
    const torch::Tensor &geomBuffer, const int R,
    const torch::Tensor &binningBuffer, const torch::Tensor &imageBuffer,
    const bool debug, const torch::Tensor &config);

torch::Tensor markVisible(torch::Tensor &means3D, torch::Tensor &viewmatrix,
                          torch::Tensor &projmatrix);