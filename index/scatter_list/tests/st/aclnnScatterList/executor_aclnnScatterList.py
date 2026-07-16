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
from typing import List

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi


@register("function_scatter_for_aclnn_scatter_list")
class FunctionScatterListApi(BaseApi):
    def __call__(
        self, input_data: InputDataset, with_output: bool = False
    ) -> List[torch.Tensor]:
        var = input_data.kwargs["var"]
        indice = input_data.kwargs["indice"]
        updates = input_data.kwargs["updates"]
        mask = input_data.kwargs.get("mask")  # 使用 get 避免 KeyError
        axis = input_data.kwargs["axis"]

        # 克隆 var 作为结果
        result = [v.clone() for v in var]
        var_dim = result[0].dim()
        axis = axis if axis >= 0 else axis + var_dim

        if indice.dim() == 0:
            self._process_0d_indices(result, indice, updates, mask, axis)
        elif indice.dim() == 1:
            self._process_1d_indices(result, indice, updates, mask, axis)
        elif indice.dim() == 2:
            self._process_2d_indices(result, indice, updates, mask, axis)
        else:
            raise ValueError(f"Unsupported indice dimension: {indice.dim()}")

        return result

    def _process_0d_indices(self, result, indice, updates, mask, axis):
        """处理 0 维索引"""
        offset = indice.item()
        for i, src in enumerate(updates):
            if mask is not None and i < len(mask) and mask[i].item() == 0:
                continue
            if offset >= result[i].size(axis):
                continue

            length = min(src.size(axis), result[i].size(axis) - offset)
            slices_src = [slice(None)] * src.dim()
            slices_src[axis] = slice(0, length)

            slices_dst = [slice(None)] * result[i].dim()
            slices_dst[axis] = slice(offset, offset + length)

            result[i][tuple(slices_dst)] = src[tuple(slices_src)]

    def _process_1d_indices(self, result, indice, updates, mask, axis):
        """处理 1 维索引"""
        for i in range(indice.size(0)):
            if mask is not None and i < len(mask) and mask[i].item() == 0:
                continue

            idx = indice[i].item()
            src = updates[i]
            length = max(0, min(src.size(axis), result[i].size(axis) - idx))

            if length <= 0:
                continue

            slices = [slice(None)] * src.dim()
            slices[axis] = slice(0, length)
            result[i][tuple(slices)] = src[tuple(slices)]

    def _process_2d_indices(self, result, indice, updates, mask, axis):
        """处理 2 维索引"""
        for i in range(indice.size(0)):
            if mask is not None and i < len(mask) and mask[i].item() == 0:
                continue

            start = indice[i, 0].item()
            length = indice[i, 1].item()
            src = updates[i]

            length = max(0, min(length, result[i].size(axis) - start, src.size(axis)))

            if length <= 0:
                continue

            slices_src = [slice(None)] * src.dim()
            slices_src[axis] = slice(0, length)

            slices_dst = [slice(None)] * result[i].dim()
            slices_dst[axis] = slice(start, start + length)

            result[i][tuple(slices_dst)] = src[tuple(slices_src)]


@register("aclnn_scatter_list")
class ScatterListAclnnApi(AclnnBaseApi):
    def init_by_input_data(self, input_data):
        self.task_result.output_info_list = [self.task_result.output_info_list]
        input_args, output_packages = super().init_by_input_data(input_data)
        input_args.pop()
        output_packages[:] = [input_args[0]]
        return input_args, output_packages

    def __call__(self):
        super().__call__()

    def after_call(self, output_packages):
        output = []
        for output_package in output_packages:
            output.append(self.acl_tensorlist_to_torch(output_package))
        return output
