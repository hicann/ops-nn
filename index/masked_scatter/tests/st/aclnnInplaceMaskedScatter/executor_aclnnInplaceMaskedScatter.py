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
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi

from einops import rearrange


@register("aclnn_masked_scatter")
class AclnnMaskedScatterApi(BaseApi):  
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        output = None
        if not with_output:                                         
            eval(self.api_name)(*input_data.args, **input_data.kwargs)
            return output

        if self.device == 'cpu':
            output = eval(self.api_name)(*input_data.args, **input_data.kwargs)
            output = torch.reshape(output, (-1,))
            return output

        elif self.device == "npu":
            if self.device_id == 0:  
                torch.npu.set_compile_mode(jit_compile=False)
            elif self.device_id == 1: 
                torch.npu.set_compile_mode(jit_compile=True)
            output = eval(self.api_name)(
                *input_data.args, **input_data.kwargs
            )
            return output

        if self.output is None:
            output = eval(self.api_name)(
                *input_data.args, **input_data.kwargs
            )
        else:
            if isinstance(self.output, int):
                output = input_data.args[self.output]
            elif isinstance(self.output, str):
                output = input_data.kwargs[self.output]
            else:
                raise ValueError(
                    f"self.output {self.output} value is " f"error"
                )
        return output

@register("pyaclnn_masked_scatter")
class AddAclnnApi(AclnnBaseApi):
    def __call__(self):
        super().__call__()

    def init_by_input_data(self, input_data: InputDataset):
        input_args, output_packages = super().init_by_input_data(input_data)
        input_args.pop()
        output_packages[:] = [input_args[0]]
        return input_args, output_packages

    def after_call(self, output_packages):
        output = []
        for output_pack in output_packages:
            output.append(self.acl_tensor_to_torch(output_pack))
        return output