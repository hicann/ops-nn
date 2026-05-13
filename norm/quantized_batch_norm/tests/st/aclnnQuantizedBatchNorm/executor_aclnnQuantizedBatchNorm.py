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
#

import torch
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.dataset.base_dataset import OpsDataset
from atk.tasks.backends.lib_interface.acl_wrapper import AclFormat
import torch.nn.functional as F
from atk.tasks.backends.lib_interface.acl_wrapper import aclnn, ascendcl, nnopbase
from ctypes import *
from atk.tasks.backends.lib_interface.acl_wrapper import *


ACLTYPE_TO_TORCH = {
    AclDataType.ACL_FLOAT: torch.float,
    AclDataType.ACL_FLOAT16: torch.float16,
    AclDataType.ACL_UINT8: torch.uint8,
    AclDataType.ACL_INT8: torch.int8,
    AclDataType.ACL_INT16: torch.int16,
    AclDataType.ACL_INT32: torch.int,
    AclDataType.ACL_INT64: torch.int64,
    AclDataType.ACL_DOUBLE: torch.double,
    AclDataType.ACL_BOOL: torch.bool,
    AclDataType.ACL_COMPLEX64: torch.complex64,
    AclDataType.ACL_COMPLEX128: torch.complex128,
    AclDataType.ACL_BF16: torch.bfloat16,
    AclDataType.ACL_COMPLEX32: torch.complex32,
}


@register("function_quantized_batch_norm")
class AclnnQuantizedBatchNorm(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        
        if self.device == "cpu":
            # print(input_data)
            qx = input_data.kwargs["input"]
            # print(qx)
            qxtype = input_data.kwargs["input"].dtype
            # print(qxtype)
            quant_dtype = QUANT_TYPE_MAP.get(qxtype)
            mean = input_data.kwargs["mean"]
            var = input_data.kwargs["var"]
            weight = input_data.kwargs["weight"]
            bias = input_data.kwargs["bias"]
            input_scale = input_data.kwargs["inputScale"]
            input_zero_point = input_data.kwargs["inputZeroPoint"]
            output_scale = input_data.kwargs["outputScale"]
            output_zero_point = input_data.kwargs["outputZeroPoint"]
            eps = input_data.kwargs["epsilon"]


            if qx.dim() == 4 :
                x_data = torch._make_per_tensor_quantized_tensor(qx,input_scale,input_zero_point)
            # x_data = torch.quantize_per_tensor(qx,input_scale,input_zero_point,quant_dtype)
            # print(output_zero_point)
                output= torch.quantized_batch_norm(x_data,weight,bias,mean, var,eps,output_scale,output_zero_point).int_repr()
            else:
                output = qx

            return output

    def get_format(self, input_data: InputDataset, index=None, name=None):
        if input_data.kwargs["format"] == "NCHW":
            return AclFormat.ACL_FORMAT_NCHW
        if input_data.kwargs["format"] == "NCL":
            return AclFormat.ACL_FORMAT_NCL
        return AclFormat.ACL_FORMAT_ND

from atk.configs.results_config import AccuracyConfig
from atk.tasks.post_process import ACCURACY_REGISTRY
from atk.tasks.post_process.base_compare import BaseAccuracyCompare

@ACCURACY_REGISTRY.register("stand_quantize")
class SingleBenchmarkAccuracyCompare(BaseAccuracyCompare):
    @staticmethod
    def compute_quantize_accuracy(local_output, remote_output, data_file):
        diff_value = torch.subtract(local_output.to(torch.int64), remote_output)
        diff_abs = torch.abs(diff_value)

        flat_diff_abs = diff_abs.view(-1)
        max_diff, max_diff_idx = torch.max(flat_diff_abs, dim = 0)
        result = torch.all(diff_abs <= 1)

        acc_result = AccuracyConfig(
            result = result.item(),
            filename = data_file,
            max_diff = max_diff.item(),
            max_diff_idx = max_diff_idx.item(),
        )
        return acc_result
    def compute_accuracy_result(self, local_output, remote_output, data_file):
        acc_ret = self.compute_quantize_accuracy(local_output, remote_output, data_file)
        return acc_ret

QUANT_TYPE_MAP = {
    torch.int8: torch.qint8,
    torch.uint8: torch.quint8,
    torch.int32: torch.qint32,
}

@register("function_quantized_batch_norm_discontinuous")
class QuantizedBatchNorm(BaseApi):
    transpose_dim_a = 0
    transpose_dim_b = 3

    def init_by_input_data(self, input_data: InputDataset):
        # if self.device == "pyaclnn"or self.device == "cpu":
            # print("-------------",input_data.kwargs['input'])
            # print("-------------",input_data.kwargs["input"].is_contiguous())

        if len(input_data.kwargs['input'].shape) == 4 :
                input_data.kwargs['input'].transpose_(self.transpose_dim_a, self.transpose_dim_b)
                # print("cpu",input_data.kwargs["input"].is_contiguous())
                # print("cpu",input_data.kwargs['input'].shape)     
                                  

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        
        if self.device == "cpu":
            # print(input_data)
            qx = input_data.kwargs["input"]
            # print(qx)
            qxtype = input_data.kwargs["input"].dtype
            # print(qxtype)
            quant_dtype = QUANT_TYPE_MAP.get(qxtype)
            mean = input_data.kwargs["mean"]
            var = input_data.kwargs["var"]
            weight = input_data.kwargs["weight"]
            bias = input_data.kwargs["bias"]
            input_scale = input_data.kwargs["inputScale"]
            input_zero_point = input_data.kwargs["inputZeroPoint"]
            output_scale = input_data.kwargs["outputScale"]
            output_zero_point = input_data.kwargs["outputZeroPoint"]
            eps = input_data.kwargs["epsilon"]
            x_data = torch._make_per_tensor_quantized_tensor(qx,input_scale,input_zero_point)
            # x_data = torch.quantize_per_tensor(qx,input_scale,input_zero_point,quant_dtype)
            # print(output_zero_point)
            output= torch.quantized_batch_norm(x_data,weight,bias,mean, var,eps,output_scale,output_zero_point).int_repr()
            # shapeold = output.shape
            # print(output)
            # outputnew = output.transpose_(0, 3).contiguous().reshape(shapeold)
            # print(outputnew)
        return output

    def get_format(self, input_data: InputDataset, index=None, name=None):
        if input_data.kwargs["format"] == "NCHW":
            return AclFormat.ACL_FORMAT_NCHW
        if input_data.kwargs["format"] == "NCL":
            return AclFormat.ACL_FORMAT_NCL
        return AclFormat.ACL_FORMAT_ND


@register("function_aclnn_quantized_batch_norm_discontinuous")
class AclnnQuantizedBatchNorm(AclnnBaseApi):
    transpose_dim_a = 0
    transpose_dim_b = 3

    def init_by_input_data(self, input_data: InputDataset):
        input_args = []  # 算子的入参列表
        output_packages = []  # 算子的出参数据包列表
        for i, arg in enumerate(input_data.args):
            data = self.backend.convert_input_data(arg, index=i)
            input_args.extend(data)

        for name, kwarg in input_data.kwargs.items():
            if(name == "input"):
                data = self.convert_input_data_self(kwarg, name=name)
                input_args.extend(data)
            else:
                data = self.backend.convert_input_data(kwarg, name=name)
                input_args.extend(data)

        for index, output_data in enumerate(self.task_result.output_info_list):
            output = self.backend.convert_output_data(output_data, index)
            output_packages.extend(output)
        input_args.extend(output_packages)

        # print(input_data.kwargs['input'])

        return input_args, output_packages

    def create_input_acl_tensor_self(self, tensor: torch.Tensor, fmt: AclFormat) -> AclTensorStruct:
        dtype = TORCH_TO_ACLTYPE[str(tensor.dtype)]
        dtype_size = acl.data_type_size(dtype)
        data_size = 1
        for dim in tensor.shape:
            data_size *= dim
        data_size *= dtype_size
        strides = (Int64 * len(tensor.shape))(*[1] * len(tensor.shape))
        origintensorshape = tensor.transpose(self.transpose_dim_a, self.transpose_dim_b).shape
        for i in range(len(origintensorshape)-2 , -1 , -1):
            strides[i] = origintensorshape[i+1] * strides[i+1]
        temp_a = strides[self.transpose_dim_a]
        strides[self.transpose_dim_a] = strides[self.transpose_dim_b]
        strides[self.transpose_dim_b] = temp_a

        device_addr = ctypes.c_void_p(tensor.data_ptr())

        acl_tensor = nnopbase.aclCreateTensor(
        (Int64 * len(tensor.shape))(*tensor.shape), len(tensor.shape), dtype, strides, 0,fmt,(Int64 * len(origintensorshape))(*origintensorshape),len(origintensorshape),device_addr
        )

        return acl_tensor                       

    def convert_input_data_self(self, data, index=None, name=None):
        """
        从json中读取对应的
        int、float、bool、str、Tensor、Scalar、TensorList、ScalarList、IntArray、FloatArray、BoolArray
        """

        # print("iam in here")
        if isinstance(data, (list, tuple)):  # 将列表和元组转换为xxxList和xxxArray
            data_list = []
            for tmp in data:
                data_list.extend(self.convert_input_data(tmp, index, name))  # 递归调用并展开结果
            return [nnopbase.create_x_list(data_list)]
        elif isinstance(data, torch.Tensor):  # 将torch.tensor转换为aclTensor
            fmt = AclFormat.ACL_FORMAT_NCHW
            acl_tensor = self.create_input_acl_tensor_self(data, fmt)
            return [acl_tensor]
        elif isinstance(data, (float, int, bool, str, np.float32)):  # 将这四个数据类型转换为aclScalar或者对应的C++数据类型
            data_config = self.case_config.get_input_data_config(index, name)
            data_dtype = data_config.dtype
            if data_dtype == "non_param":
                return []
            if "attr" in data_config.type:  # 如果type写的是attr，则转换为对应的C++数据类型，否则转换为Scalar
                if data_dtype not in PYTYPE_TO_CTYPE:  # 根据dtype选择C++的数据类型（使用ctypes）
                    raise ValueError(f"PYTYPE_TO_CTYPE不支持的dtype：{data_dtype}")
                ctype = PYTYPE_TO_CTYPE[data_dtype]
                if ctype == ctypes.c_char_p:  # Python的字符串是Unicode字符串，C风格的字符串一般是字节字符串（如ASCII或UTF-8）
                    data = data.encode('utf-8')
                return [ctype(data)]
            if data_dtype not in PYTYPE_TO_ACLTYPE:  # 根据dtype选择acl的dtype的数据类型（即AclDataType）
                raise ValueError(f"PYTYPE_TO_ACLTYPE不支持的dtype：{data_dtype}")
            return [nnopbase.create_acl_scalar(data, PYTYPE_TO_ACLTYPE[data_dtype])]
        elif isinstance(data, torch.dtype):
            if data not in TorchDtype_to_AclDtype:
                raise ValueError(f"TorchDtype_to_AclDtype不支持的dtype：{data}")
            data = TorchDtype_to_AclDtype.get(data)
            return [ctypes.c_int64(data)]
        elif data is None:
            return [ctypes.c_void_p(0)]

        return [data]


    def create_acl_tensor_transpose_self(self, tensor: torch.Tensor, fmt: AclFormat) -> AclTensorStruct:
        dtype = TORCH_TO_ACLTYPE[str(tensor.dtype)]
        dtype_size = acl.data_type_size(dtype)
        data_size = 1
        for dim in tensor.shape:
            data_size *= dim
        data_size *= dtype_size
        strides = (Int64 * len(tensor.shape))(*[1] * len(tensor.shape))
        for i in range(len(tensor.shape)-2 , -1 , -1):
            strides[i] = tensor.shape[i+1] * strides[i+1]
        temp_a = strides[self.transpose_dim_a]
        strides[self.transpose_dim_a] = strides[self.transpose_dim_b]
        strides[self.transpose_dim_b] = temp_a

        newtensor = tensor.reshape(tensor.transpose(self.transpose_dim_a, self.transpose_dim_b).shape)
        newshape = newtensor.shape
        device_addr = ctypes.c_void_p(newtensor.data_ptr())

        # print("goooooo",strides[0],"goooooo",strides[1],"goooooo",strides[2],"goooooo",strides[3])
        # print("g11",newshape[0],"g11",newshape[1],"g11",newshape[2],"g11",newshape[3])
        # print("g22",tensor.shape[0],"g22",tensor.shape[1],"g22",tensor.shape[2],"g22",tensor.shape[3])

        # acl_tensor = nnopbase.aclCreateTensor(
        # (Int64 * len(tensor.shape))(*tensor.shape), len(tensor.shape), dtype, strides, 0,fmt,(Int64 * len(tensor.shape))(*tensor.shape),len(tensor.shape),device_addr
        # )

        acl_tensor = nnopbase.aclCreateTensor(
        (Int64 * len(newshape))(*newshape), len(newshape), dtype, strides, 0,fmt,(Int64 * len(newshape))(*newshape),len(newshape),device_addr
        )

        return AclTensorStruct(acl_tensor, device_addr, newshape, dtype)

    def after_call(self, output_packages):
        output = []
        for output_pack in output_packages:
            if isinstance(output_pack, AclTensorStruct):
                output.append(self.acl_tensor_to_torch_self(output_pack))
            elif isinstance(output_pack, AclTensorlistStruct):
                output.append(self.backend.acl_tensorlist_to_torch(output_pack))
                
        return output

    def acl_tensor_to_torch_self(self, tensor_struct: AclTensorStruct) -> torch.Tensor:
        # 计算数据的总字节数
        size = 1
        for dim in tensor_struct.shape:
            size *= dim
        data_size = size * acl.data_type_size(tensor_struct.dtype)

        if tensor_struct.dtype not in ACLTYPE_TO_TORCH:
            raise ValueError(f"ACLTYPE_TO_TORCH不支持的dtype：{tensor_struct.dtype}")

        # result_tensor_temp = torch.empty(tensor_struct.shape, dtype=ACLTYPE_TO_TORCH[tensor_struct.dtype], device='cpu')

        
        # 创建一个空的PyTorch Tensor
        result_tensor = torch.empty(tensor_struct.shape, dtype=ACLTYPE_TO_TORCH[tensor_struct.dtype], device='cpu')

        # 将设备端的数据拷贝到PyTorch Tensor的内存中
        # 需要获取PyTorch Tensor的指针
        result_ptr = result_tensor.data_ptr()

        # 执行内存拷贝
        ascendcl.aclrtMemcpy(result_ptr, data_size, tensor_struct.addr, data_size,
                             AclrtMemcpyKind.ACL_MEMCPY_DEVICE_TO_HOST)

        # print(result_tensor)

        oldshape = result_tensor.shape
        newshape = result_tensor.transpose(self.transpose_dim_a,self.transpose_dim_b).shape

        # print(newshape)

        # finalresult = result_tensor.reshape(newshape).transpose_(self.transpose_dim_a,self.transpose_dim_b)
        finalresult = result_tensor.transpose(self.transpose_dim_a,self.transpose_dim_b).contiguous().reshape(oldshape)
        # print(finalresult)
        return result_tensor

    def get_format(self, input_data: InputDataset, index=None, name=None):
        if input_data.kwargs["format"] == "NCHW":
            return AclFormat.ACL_FORMAT_NCHW
        if input_data.kwargs["format"] == "NCL":
            return AclFormat.ACL_FORMAT_NCL
        return AclFormat.ACL_FORMAT_ND
