#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
import torch

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult

from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.dataset.base_dataset import OpsDataset

@register("function_aclnn_group_norm_swish_grad")
class AsceneGroupNormSwishGradApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(AsceneGroupNormSwishGradApi, self).__init__(task_result)
        OpsDataset.seed_everything()
        self.change_flag = None

    def init_by_input_data(self, input_data: InputDataset):
        """
        该接口可实现部门场景下api的初始化需要依赖于当前的输入数据，且不希望计入耗时，
        可以在此接口实现
        :param input_data:
        :return:
        """

        if self.device == "pyaclnn":
            input_data.kwargs["dy"] = input_data.kwargs["dy"].npu()
            input_data.kwargs["mean"] = input_data.kwargs["mean"].npu()
            input_data.kwargs["rstd"] = input_data.kwargs["rstd"].npu()
            input_data.kwargs["x"] = input_data.kwargs["x"].npu()
            input_data.kwargs["gamma"] = input_data.kwargs["gamma"].npu()
            input_data.kwargs["beta"] = input_data.kwargs["beta"].npu()
            

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        if self.device == "gpu":
            device = f"cuda:{self.device_id}"
        elif self.device == "npu":
            device = f"{self.device}:{self.device_id}"
        else:
            device = "cpu"

        if 0 in input_data.kwargs["dy"].shape or input_data.kwargs["dy"].shape == torch.Size([]): ## 空tensor、标量tensor
            dxout = input_data.kwargs["x"]
            dgammaOut = input_data.kwargs["gamma"]
            dbetaOut = input_data.kwargs["beta"]
            return dxout, dgammaOut, dbetaOut
        else:
            x = input_data.kwargs["x"]
            dtype = x.dtype
            # N
            batch_num = x.size(0)
            # C
            num_channels = x.size(1)
            num_groups = input_data.kwargs["numGroups"]
            grad = input_data.kwargs["dy"]
            gamma = input_data.kwargs["gamma"]
            beta = input_data.kwargs["beta"]
            swish_scale = input_data.kwargs["swishScale"]
            mean = input_data.kwargs["mean"]
            rstd = input_data.kwargs["rstd"]

            max_shape = max(grad.shape)
            if max_shape >= 2147483649: ## 上边界用例
                dxout = input_data.kwargs["x"]
                dgammaOut = input_data.kwargs["gamma"]
                dbetaOut = input_data.kwargs["beta"]
                return dxout, dgammaOut, dbetaOut
            else:
                remaining_dims = x.size()[2:]
                hw = 1
                count = 0
                for size in remaining_dims:
                    hw *= size
                    count += 1

                num_per_group = num_channels // num_groups * hw
                if count > 1:
                    x = x.reshape((batch_num, num_channels, -1)).to(torch.float32)
                    grad = grad.reshape((batch_num, num_channels, -1)).to(torch.float32)

                dL_dgamma = torch.zeros(num_channels).to(device)
                dL_dbeta = torch.zeros(num_channels).to(device)
                dL_dx = torch.zeros(batch_num, num_channels, hw).to(device)

                for n_i in range(batch_num):
                    for g_i in range(num_groups):
                        x_i = x[n_i, g_i * (num_channels // num_groups):(g_i + 1) * (num_channels // num_groups)]
                        dy_i = grad[n_i, g_i * (num_channels // num_groups):(g_i + 1) * (num_channels // num_groups)]

                        var_x = torch.Tensor.var(x_i, False)
                        rstd_x = rstd[n_i, g_i]
                        mean_x = mean[n_i, g_i]
                        x_norm_i = (x_i - mean_x) * rstd_x

                        gamma_i = gamma[g_i * (num_channels // num_groups):(g_i + 1) * (num_channels // num_groups)]
                        beta_i = beta[g_i * (num_channels // num_groups):(g_i + 1) * (num_channels // num_groups)]

                        if count == 0:
                            gamma_reshape_i = gamma_i.reshape(num_channels // num_groups)
                            beta_reshape_i = beta_i.reshape(num_channels // num_groups)
                        else:
                            gamma_reshape_i = gamma_i.reshape(num_channels // num_groups, 1)
                            beta_reshape_i = beta_i.reshape(num_channels // num_groups, 1)

                        dy_new_i = x_norm_i * gamma_reshape_i

                        dy_new_i = dy_new_i + beta_reshape_i
                        # 正向结果输入给激活函数的反向
                        dswish_res_i = dy_new_i * (-swish_scale)
                        dswish_res_i = torch.exp(dswish_res_i)
                        dswish_res_i = dswish_res_i + 1.0
                        tmp_res = swish_scale * dy_new_i / dswish_res_i
                        tmp_res = swish_scale * dy_new_i - tmp_res
                        tmp_res = tmp_res + 1.0
                        dswish_res_i = tmp_res / dswish_res_i
                        # dy_new_i 等于激活函数的结果
                        dy_new_i = dswish_res_i * dy_i

                        # 激活函数的输出计算结合dy算出来最终的结果；
                        if count == 0:
                            temp_1 = dy_new_i
                            temp_2 = dy_new_i * x_norm_i
                        else:
                            temp_1 = torch.sum(dy_new_i, dim=(1))
                            temp_2 = torch.sum(dy_new_i * x_norm_i, dim=(1))

                        temp_1.to(device)
                        temp_2.to(device)

                        dL_dbeta[g_i * (num_channels // num_groups):(g_i + 1) * (num_channels // num_groups)] += temp_1
                        dL_dgamma[g_i * (num_channels // num_groups):(g_i + 1) * (num_channels // num_groups)] += temp_2

                        c1 = 0
                        c2 = 0
                        c1 = torch.sum(temp_1 * gamma_i) / num_per_group
                        c2 = torch.sum(temp_2 * gamma_i) / num_per_group

                        dL_dx_G_C = torch.zeros(num_channels // num_groups, hw)

                        for i in range(num_channels // num_groups):
                            dL_dx_G_C[i] = rstd_x * (
                                        dy_new_i[i] * gamma[g_i * (num_channels // num_groups) + i] - x_norm_i[i] * c2 - c1)
                        dL_dx[n_i, g_i * (num_channels // num_groups):(g_i + 1) * (num_channels // num_groups)] = dL_dx_G_C
            return dL_dx.to(dtype), dL_dgamma.to(dtype), dL_dbeta.to(dtype)

    def get_cpp_func_signature_type(self):
        return "aclnnStatus aclnnGroupNormSwishGradGetWorkspaceSize( \
            const aclTensor* dy, const aclTensor* mean, const aclTensor* rstd, \
            const aclTensor* x, const aclTensor* gamma, const aclTensor* beta, \
            int64_t group, char* dataFormat, double swishScale, bool dgammaIsRequire,\
             bool dbetaIsRequire, aclTensor* dxOut, aclTensor* dgammaOutOptional, \
             aclTensor* dbetaOutOptional, uint64_t* workspaceSize, aclOpExecutor** executor)"