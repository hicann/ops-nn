#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import numpy as np
import torch
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.api_execute.base_api import BaseApi


@register("aclnn_rms_norm_grad_quant")
class FunctionRmsNormGradQuantApi(BaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        # 不注入 epsilon：aclnnRmsNormGradQuant 签名无此参数（rstd 由前向存档
        # 直接传入，golden 亦不消费），此前照抄 aclnnAddRmsNormQuant（其签名
        # 确有 epsilon）属模板残留。
        pass

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        input_dy = input_data.kwargs["dy"]
        input_x = input_data.kwargs["x"]
        input_rstd = input_data.kwargs["rstd"]
        input_gamma = input_data.kwargs["gamma"]
        input_scales_x = input_data.kwargs["scalesX"]
        input_offset_x = input_data.kwargs.get("offsetX", None)
        div_mode = input_data.kwargs["div_mode"]
        dst_type = input_data.kwargs["dst_type"]
        if self.device == "cpu":
            output_dx, output_dgamma = self.npu_rms_norm_grad_quant_golden(
                input_dy,
                input_x,
                input_rstd,
                input_gamma,
                input_scales_x,
                input_offset_x,
                div_mode,
                dst_type,
            )
        return output_dx, output_dgamma

    def npu_rms_norm_grad_quant_golden(
        self,
        input_dy,
        input_x,
        input_rstd,
        input_gamma,
        input_scales_x,
        input_offset_x,
        div_mode,
        dst_type,
    ):
        # golden 按 aclnnRmsNormGradQuant 文档公式实现：
        #   dx = rstd*gamma*dy - x*rstd^3*mean(dy*gamma*x)，随后量化 dx*scales+offset（div_mode 时 dx/scales+offset）
        #   dgamma = sum(dy * x * rstd, axis=batch)
        # 形状约定（对齐 host tiling CheckRelationShapes 与 assets/golden.py）：
        #   x/dy 统一展平为 [prod(rstd.shape), prod(gamma.shape)]，rstd 展平为 [rows, 1]，
        #   dgamma 沿完整展平批次轴归约后恢复 gamma 原形状——支持多维批次（rstd.shape = x.shape[0:n]）。
        torch_type32 = torch.float32
        ori_x_shape = input_x.shape
        ori_gamma_shape = input_gamma.shape

        input_gamma_f = input_gamma.to(torch_type32).reshape(
            np.multiply.reduce(np.array(ori_gamma_shape))
        )
        rows = int(np.multiply.reduce(np.array(input_rstd.shape)))
        x_shape = (rows,) + tuple(input_gamma_f.shape)
        input_x_f = input_x.to(torch_type32).reshape(x_shape)
        input_dy_f = input_dy.to(torch_type32).reshape(x_shape)

        # rstd: 形状 = batch 维展开（文档要求 rstd_shape = x_shape[0:n]），转成列向量便于广播
        rstd_f = input_rstd.to(torch_type32).reshape(rows, 1)

        # 空批次约定（对齐 EmptyDgamma::VFDuplicateRows 与 assets/golden.py rows==0 分支）：
        # dx 保持空；非空 dgamma 显式输出全 NaN（空和本应无定义，算子以 NaN 标记）
        if rows == 0:
            dx_out = torch.empty(ori_x_shape, dtype=torch.int8)
            dgamma = torch.full(
                ori_gamma_shape, float("nan"), dtype=torch_type32, device=input_x.device
            )
            return dx_out, dgamma

        mean_term = torch.mean(
            input_dy_f * input_gamma_f * input_x_f, axis=-1, keepdims=True
        )
        dx = (
            rstd_f * input_gamma_f * input_dy_f
            - input_x_f * torch.pow(rstd_f, 3) * mean_term
        )

        # 量化出口：div_mode 决定 dx/scales 或 dx*scales；offset 可选
        scales_f = input_scales_x.to(torch_type32)
        if div_mode:
            dx_quant = dx / scales_f
        else:
            dx_quant = dx * scales_f
        if input_offset_x is not None:
            offset_f = input_offset_x.to(torch_type32)
            dx_quant = dx_quant + offset_f
        # 量化出口按 dst_type 分流（对齐 tests/assets/golden.py 的 2/34 分支，
        # 即 infershape.cpp 白名单 DT_INT8(2)/DT_HIFLOAT8(34)）：
        #   int8: 饱和 + 取整后转 int8；
        #   hif8: 浮点 cast（无取整），饱和到 [-32768, 32768]；torch 无原生
        #         hif8 载体，转 numpy(en_dtypes) 承接，与 assets/golden.py 同策。
        if dst_type == 2:
            dx_out = torch.clamp(torch.round(dx_quant), -128, 127).to(torch.int8)
            dx_out = dx_out.reshape(ori_x_shape)
        elif dst_type == 34:
            from en_dtypes import hifloat8

            dx_out = np.clip(
                dx_quant.reshape(ori_x_shape).detach().cpu().numpy(), -32768.0, 32768.0
            ).astype(hifloat8)
        else:
            raise ValueError(f"Unsupported dx dtype code: {dst_type}; expected 2 or 34")

        dgamma = torch.sum(input_dy_f * input_x_f * rstd_f, axis=0)
        dgamma = dgamma.reshape(ori_gamma_shape).to(torch_type32)

        return dx_out, dgamma


@register("sample_aclnn_api")
class SampleAclnnApi(AclnnBaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        # aclnn 签名(aclnn_rms_norm_grad_quant.h:41-46)不含 dst_type——该属性仅在
        # infershape/tiling 层消费，aclnn 侧由 dxOut 的 dtype 反推后传给 L0
        # (aclnn_rms_norm_grad_quant.cpp:122-123)，故删除以免污染形参表；
        # div_mode 在签名内必须保留。邻仓 add_rms_norm_quant 删 kernelType 是
        # 因其签名确无该参——照抄时不可换错对象。
        # 注：不注入 epsilon——aclnnRmsNormGradQuant 签名无此参数（rstd 由前向
        # 存档直接传入，golden 也不消费它），邻仓注入是因其签名确有 epsilon。
        del input_data.kwargs["dst_type"]
        input_args, output_packages = super().init_by_input_data(input_data)
        return input_args, output_packages

    def after_call(self, output_packages):
        output1, output2 = super().after_call(output_packages)
        return output1, output2
