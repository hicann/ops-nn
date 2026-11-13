#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
import torch
from atk.common.log import Logger
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import AclFormat
from atk.tasks.backends.lib_interface.acl_wrapper import TensorPtr
from atk.tasks.api_execute.base_api import BaseApi

logging = Logger().get_logger()
try:
    pass
except Exception:
    logging.warning("import torch_npu failed!!!")

GATES = 3


# 改造：不跑前向，直接接收外部传入r/z/n/h_n/h时序张量
def golden_grad_layer(
    time, batch, input_size, hidden_size, x, w_ih, w_hh, h0, dy, dh, rs, zs, ns, hns, hs
):
    grad_h = dh
    dgi_list, dgh_list, x_list, hp_list = [], [], [], []
    for t in reversed(range(time)):
        hp = h0 if t == 0 else hs[t - 1]
        ght = dy[t] + grad_h
        dn = ght * (1 - zs[t])
        dz_raw = ght * (hp - ns[t])
        dhph = ght * zs[t]
        di_n = dn * (1 - ns[t] ** 2)
        drh = di_n * rs[t]
        dr = (di_n * hns[t]) * rs[t] * (1 - rs[t])
        dz = dz_raw * zs[t] * (1 - zs[t])
        dgi = torch.cat([dr, dz, di_n], 1)
        dgh = torch.cat([dr, dz, drh], 1)
        grad_h = dgh @ w_hh
        grad_h = grad_h + dhph
        dgi_list.append(dgi)
        dgh_list.append(dgh)
        x_list.append(x[t])
        hp_list.append(hp)
    dgi_all = torch.cat(dgi_list, 0)
    dgh_all = torch.cat(dgh_list, 0)
    x_all = torch.cat(x_list, 0)
    hp_all = torch.cat(hp_list, 0)
    dx = dgi_all @ w_ih
    dh_prev = grad_h
    dw_ih = dgi_all.T @ x_all
    print("~" * 20, flush=True)
    print(dgi_all, flush=True)
    print("~" * 20, flush=True)
    print(x_all, flush=True)
    print("~" * 20, flush=True)
    print(dw_ih, flush=True)
    dw_hh = dgh_all.T @ hp_all

    def sequential_sum(tensor, dim=0):
        # 初始化0
        res = tensor.select(dim, 0).clone()
        n = tensor.size(dim)
        for i in range(1, n):
            res += tensor.select(dim, i)
        return res

    db_ih = sequential_sum(dgi_all, dim=0)
    db_hh = sequential_sum(dgh_all, dim=0)
    grads = {
        "dx": dx,
        "dh_prev": dh_prev,
        "dw_ih": dw_ih,
        "dw_hh": dw_hh,
        "db_ih": db_ih,
        "db_hh": db_hh,
        "dgi_all": dgi_all,
        "dgh_all": dgh_all,
    }
    return grads


def multi_layer_golden_grad(
    time,
    batch,
    input_size,
    hidden_size,
    x_t,
    params_t_list,
    hx_t,
    dy_t,
    dh_t,
    gate_t_list,
    num_layers,
    bidirectional,
    has_bias,
):
    D = 2 if bidirectional else 1
    H = hidden_size
    grads_per_layer = {}
    cur_dy = dy_t
    layer_output_cache = {}
    for layer_idx in range(num_layers):
        idx_fwd = layer_idx * D + 0
        h_fwd = gate_t_list["h"][idx_fwd]
        if D == 2:
            idx_bwd = layer_idx * D + 1
            h_bwd = gate_t_list["h"][idx_bwd]
            layer_out = torch.cat([h_fwd, h_bwd], dim=-1)
        else:
            layer_out = h_fwd
        layer_output_cache[layer_idx] = layer_out

    for layer_idx in reversed(range(num_layers)):
        if layer_idx == 0:
            x_layer = x_t
            cur_in_size = input_size
        else:
            x_layer = layer_output_cache[layer_idx - 1]
            cur_in_size = H * D
        dx_accum = None
        for d in range(D):
            idx = layer_idx * D + d
            p_off = idx * (4 if has_bias else 2)
            w_ih_t = params_t_list[p_off + 0]
            w_hh_t = params_t_list[p_off + 1]
            h0_layer = hx_t[idx]
            gate_r = gate_t_list["r"][idx]
            gate_z = gate_t_list["z"][idx]
            gate_n = gate_t_list["n"][idx]
            gate_hn = gate_t_list["h_n"][idx]
            gate_h = gate_t_list["h"][idx]
            if bidirectional and d == 1:
                gate_r_rev = torch.flip(gate_r, [0])
                gate_z_rev = torch.flip(gate_z, [0])
                gate_n_rev = torch.flip(gate_n, [0])
                gate_hn_rev = torch.flip(gate_hn, [0])
                gate_h_rev = torch.flip(gate_h, [0])
                x_layer_rev = torch.flip(x_layer, [0])
                dy_dir = cur_dy[:, :, d * H : (d + 1) * H]
                dh_dir = dh_t[idx]
                dy_dir_rev = torch.flip(dy_dir, [0])
                g = golden_grad_layer(
                    time,
                    batch,
                    cur_in_size,
                    H,
                    x_layer_rev,
                    w_ih_t,
                    w_hh_t,
                    h0_layer,
                    dy_dir_rev,
                    dh_dir,
                    gate_r_rev,
                    gate_z_rev,
                    gate_n_rev,
                    gate_hn_rev,
                    gate_h_rev,
                )
                grads_per_layer[idx] = {
                    "dx": g["dx"],
                    "dh_prev": g["dh_prev"],
                    "dw_ih": g["dw_ih"],
                    "dw_hh": g["dw_hh"],
                    "db_ih": g["db_ih"],
                    "db_hh": g["db_hh"],
                }
                dx_accum = g["dx"] if dx_accum is None else dx_accum + g["dx"]
            else:
                dy_dir = cur_dy if D == 1 else cur_dy[:, :, :H]
                dh_dir = dh_t[idx]
                g = golden_grad_layer(
                    time,
                    batch,
                    cur_in_size,
                    H,
                    x_layer,
                    w_ih_t,
                    w_hh_t,
                    h0_layer,
                    dy_dir,
                    dh_dir,
                    gate_r,
                    gate_z,
                    gate_n,
                    gate_hn,
                    gate_h,
                )
                dx_fwd = g["dx"].reshape(time, batch, -1)
                dx_fwd = torch.flip(dx_fwd, [0]).reshape(time * batch, -1)
                grads_per_layer[idx] = {
                    "dx": dx_fwd,
                    "dh_prev": g["dh_prev"],
                    "dw_ih": g["dw_ih"],
                    "dw_hh": g["dw_hh"],
                    "db_ih": g["db_ih"],
                    "db_hh": g["db_hh"],
                }
                dx_accum = dx_fwd if dx_accum is None else dx_accum + dx_fwd
        cur_dy = dx_accum.reshape(time, batch, -1)
    dx_out = cur_dy.reshape(time * batch, input_size)
    dh_prev_list = []
    dw_ih_list, dw_hh_list, db_ih_list, db_hh_list = [], [], [], []
    for layer_idx in range(num_layers):
        for d in range(D):
            idx = layer_idx * D + d
            dh_prev_list.append(grads_per_layer[idx]["dh_prev"].unsqueeze(0))
            dw_ih_list.append(grads_per_layer[idx]["dw_ih"])
            dw_hh_list.append(grads_per_layer[idx]["dw_hh"])
            db_ih_list.append(grads_per_layer[idx]["db_ih"])
            db_hh_list.append(grads_per_layer[idx]["db_hh"])
    dh_prev_out = torch.cat(dh_prev_list, 0)
    return dx_out, dh_prev_out, dw_ih_list, dw_hh_list, db_ih_list, db_hh_list


@register("gru_backward")
class GruBackwardGoldenApi(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        input_x = input_data.kwargs["input"]
        params = input_data.kwargs["params"]
        hx = input_data.kwargs["hx"]
        dy = input_data.kwargs["dy"]
        dh = input_data.kwargs["dh"]
        r_list = input_data.kwargs["r"]
        z_list = input_data.kwargs["z"]
        n_list = input_data.kwargs["n"]
        h_n_list = input_data.kwargs["h_n"]
        h_list = input_data.kwargs["h"]
        # F841 fix: current golden not support variable‑length batchSizesOptional
        _ = input_data.kwargs.get("batchSizesOptional", None)

        hasBias = input_data.kwargs.get("hasBias", True)
        numLayers = input_data.kwargs.get("numLayers", 1)
        bidirectional = input_data.kwargs.get("bidirectional", False)
        batchFirst = input_data.kwargs.get("batchFirst", False)

        compute_dtype = input_x.dtype
        x_f = input_x.to(compute_dtype)
        dy_f = dy.to(compute_dtype)
        dh_f = dh.to(compute_dtype)
        if batchFirst:
            x_f = x_f.transpose(0, 1)
            dy_f = dy_f.transpose(0, 1)
        x_shape = x_f.shape
        time = x_shape[0]
        batch = x_shape[1]
        input_size = x_shape[2]
        hx_f = hx.to(compute_dtype)
        D = 2 if bidirectional else 1
        H = hx_f.shape[-1]
        param_t_list = [p.to(compute_dtype) for p in params]
        gate_t_list = {
            "r": [r_list[i].to(compute_dtype) for i in range(D * numLayers)],
            "z": [z_list[i].to(compute_dtype) for i in range(D * numLayers)],
            "n": [n_list[i].to(compute_dtype) for i in range(D * numLayers)],
            "h_n": [h_n_list[i].to(compute_dtype) for i in range(D * numLayers)],
            "h": [h_list[i].to(compute_dtype) for i in range(D * numLayers)],
        }
        dx_out, dh_prev_out, dw_ih_list, dw_hh_list, db_ih_list, db_hh_list = (
            multi_layer_golden_grad(
                time,
                batch,
                input_size,
                H,
                x_f,
                param_t_list,
                hx_f,
                dy_f,
                dh_f,
                gate_t_list,
                numLayers,
                bidirectional,
                hasBias,
            )
        )
        dx = dx_out.reshape(time, batch, input_size)
        if batchFirst:
            dx = dx.transpose(0, 1)
        dx = dx.to(compute_dtype)
        dh_prev = dh_prev_out.to(compute_dtype)
        if hasBias:
            num_params_per = 4
        else:
            num_params_per = 2
        total_params = num_params_per * D * numLayers
        dparams = []
        for i in range(total_params):
            idx_frac = i // num_params_per
            idx_mod = i % num_params_per
            if idx_mod == 0:
                dparams.append(dw_ih_list[idx_frac].to(compute_dtype))
            elif idx_mod == 1:
                dparams.append(dw_hh_list[idx_frac].to(compute_dtype))
            elif idx_mod == 2:
                dparams.append(db_ih_list[idx_frac].to(compute_dtype))
            else:
                dparams.append(db_hh_list[idx_frac].to(compute_dtype))
        return (dx, dh_prev) + tuple(dparams)


@register("gru_pyaclnn_backward")
class GruBackwardAclnnApi(AclnnBaseApi):
    def __init__(self, task_result: TaskResult, backend):
        super().__init__(task_result, backend)

    def init_by_input_data(self, input_data):
        self.task_result.output_info_list = [
            self.task_result.output_info_list[0],
            self.task_result.output_info_list[1],
            self.task_result.output_info_list[2:],
        ]
        input_args, output_packages = super().init_by_input_data(input_data)
        input_args[10] = TensorPtr()
        return input_args, output_packages

    def get_format(self, input_data: InputDataset, index=None, name=None):
        return AclFormat.ACL_FORMAT_ND

    def get_cpp_func_signature_type(self):
        return (
            "aclnnStatus aclnnGRUBackwardGetWorkspaceSize("
            "const aclTensor *input, const aclTensorList *params, const aclTensor *hx, "
            "const aclTensor *dy, const aclTensor *dh, "
            "const aclTensorList *r, const aclTensorList *z, const aclTensorList *n, "
            "const aclTensorList *h_n, const aclTensorList *h, "
            "const aclTensor *batchSizesOptional, "
            "bool hasBias, int64_t numLayers, bool bidirectional, bool batchFirst, "
            "aclTensor *dxOut, aclTensor *dhPrevOut, aclTensorList *dparamsOut, "
            "uint64_t *workspaceSize, aclOpExecutor **executor)"
        )

    def after_call(self, output_packages):
        dx = self.acl_tensor_to_torch(output_packages[0])
        dh_prev = self.acl_tensor_to_torch(output_packages[1])
        dparams_list = self.acl_tensorlist_to_torch(output_packages[2])
        return (dx, dh_prev) + tuple(dparams_list)
