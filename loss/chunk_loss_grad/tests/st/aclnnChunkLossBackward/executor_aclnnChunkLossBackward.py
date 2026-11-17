# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""ATK executors for standalone ChunkLoss weight gradients."""

import ctypes

import torch
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi

SIGNATURE = (
    "aclnnStatus aclnnChunkLossBackwardGetWorkspaceSize("
    "const aclTensor* hidden, const aclTensor* weight, const aclTensor* labels, "
    "double alpha, const aclTensor* softmaxMax, const aclTensor* softmaxSum, "
    "const aclTensor* gradOutput, int64_t ignoreIndex, aclTensor* gradWeightOut, "
    "uint64_t* workspaceSize, aclOpExecutor** executor)"
)


def tensor_view(value, kind):
    """Create offset views without flattening their real backing storage."""
    if kind == "expand":
        return value[:1].expand_as(value), value
    if kind == "transpose" and value.ndim == 2:
        rows, columns = value.shape
        backing = torch.full(
            (columns + 1, rows + 2), 29, dtype=value.dtype, device=value.device
        )
        view = backing[1:, :rows].T
    elif kind == "gap" and value.ndim == 2:
        rows, columns = value.shape
        backing = torch.full(
            (rows + 2, columns + 3), 29, dtype=value.dtype, device=value.device
        )
        view = backing[1 : rows + 1, 1 : columns + 1]
    else:
        shape = tuple(2 * size + 1 for size in value.shape)
        backing = torch.full(shape, 29, dtype=value.dtype, device=value.device)
        view = backing[tuple(slice(1, 2 * size + 1, 2) for size in value.shape)]
    view.copy_(value)
    return view, backing


def prepare_inputs(input_data, options, dtype_name):
    """Build paired forward caches from the exact quantized input values."""
    if options.get("reject"):
        return
    parameters = input_data.kwargs
    layout = options.get("input_view")
    # ATK may prepare a dataset more than once for different backends.
    if layout == "expand":
        for name in ("hidden", "labels", "softmaxMax", "softmaxSum"):
            parameters[name] = parameters[name].clone()
    input_dtype = torch.bfloat16 if "bf" in str(dtype_name) else torch.float16
    for name, values in options.get("input_values", {}).items():
        tensor = parameters[name]
        value_dtype = input_dtype if name in ("hidden", "weight") else tensor.dtype
        value = torch.as_tensor(values, dtype=value_dtype, device=tensor.device)
        tensor.copy_(value.to(tensor.dtype).expand_as(tensor))
    pattern = options.get("data")
    if pattern == "negative":
        parameters["hidden"].copy_(-parameters["hidden"].abs())
        parameters["weight"].copy_(-parameters["weight"].abs())
    elif pattern == "mixed_subnormal":
        parameters["hidden"][:, ::2] = torch.finfo(input_dtype).tiny / 2
    labels = parameters["labels"]
    pattern = options.get("labels", "endpoints")
    if pattern == "endpoints":
        labels[0], labels[-1] = 0, parameters["weight"].shape[0] - 1
    elif pattern == "mixed_ignore":
        labels[::2] = parameters["ignoreIndex"]
    elif pattern == "all_ignore":
        labels.fill_(parameters["ignoreIndex"])
    if layout == "expand":
        parameters["labels"].zero_()
        for name in ("hidden", "labels"):
            parameters[name] = parameters[name][:1].expand_as(parameters[name])
    hidden = parameters["hidden"].detach().cpu().float()
    weight = parameters["weight"].detach().cpu().float()
    logits = hidden @ weight.T
    maximum = logits.amax(1)
    denominator = (logits - maximum[:, None]).exp().sum(1)
    if "cache_shift" in options:
        maximum = maximum + options["cache_shift"]
    if "cache_sum" in options:
        denominator.fill_(options["cache_sum"])
    # copy_ preserves ATK's non-contiguous input descriptors (-si).
    parameters["softmaxMax"].copy_(maximum.to(parameters["softmaxMax"].device))
    parameters["softmaxSum"].copy_(denominator.to(parameters["softmaxSum"].device))
    backings = []
    if layout:
        names = (
            ("hidden", "labels", "softmaxMax", "softmaxSum")
            if layout == "expand"
            else (
                "hidden",
                "weight",
                "labels",
                "softmaxMax",
                "softmaxSum",
                "gradOutput",
            )
        )
        for name in names:
            parameters[name], backing = tensor_view(parameters[name], layout)
            backings.append((backing, backing.clone()))
    return backings


def weight_gradient(input_data, high_precision, input_dtype, options):
    parameters = input_data.kwargs
    if options.get("reject"):
        return torch.zeros_like(parameters["weight"])
    compute_dtype = torch.float64 if high_precision else torch.float32
    hidden = parameters["hidden"].to(compute_dtype)
    weight = parameters["weight"].to(compute_dtype)
    logits = hidden @ weight.T
    logits_gradient = (
        logits - parameters["softmaxMax"].to(compute_dtype)[:, None]
    ).exp()
    logits_gradient /= parameters["softmaxSum"].to(compute_dtype)[:, None]
    labels = parameters["labels"].long()
    valid = labels != parameters["ignoreIndex"]
    rows = torch.where(valid)[0]
    logits_gradient[rows, labels[valid]] -= 1
    logits_gradient[~valid] = 0
    alpha = float(torch.tensor(parameters["alpha"], dtype=torch.float32))
    logits_gradient *= alpha
    if not high_precision and not options.get("fp32_gradient_reference"):
        logits_gradient = logits_gradient.to(input_dtype).float()
    result = (logits_gradient.T @ hidden) * parameters["gradOutput"].to(compute_dtype)
    return result.to(parameters["hidden"].dtype)


@register("ascend_function_aclnn_chunk_loss_backward")
class ChunkLossBackwardReference(BaseApi):
    def __init__(self, task_result):
        super().__init__(task_result)
        self.options = (task_result.case_config.strategy or {}).get(
            "chunk_loss_grad", {}
        )
        self.dtype_name = task_result.case_config.inputs[0].dtype
        self.input_dtype = (
            torch.bfloat16 if "bf" in str(self.dtype_name) else torch.float16
        )

    def init_by_input_data(self, input_data: InputDataset):
        prepare_inputs(input_data, self.options, self.dtype_name)

    def get_cpp_func_signature_type(self):
        return SIGNATURE

    def __call__(self, input_data: InputDataset, with_output=False):
        return weight_gradient(
            input_data, self.device == "cpu", self.input_dtype, self.options
        )


@register("function_aclnn_chunk_loss_backward_pyaclnn")
class ChunkLossBackwardAclnn(AclnnBaseApi):
    def __init__(self, task_result, backend):
        super().__init__(task_result, backend)
        self.options = (task_result.case_config.strategy or {}).get(
            "chunk_loss_grad", {}
        )
        self.dtype_name = task_result.case_config.inputs[0].dtype

    @classmethod
    def get_cpp_func_signature_type(cls):
        return SIGNATURE

    def view_package(self, view, template):
        # ATK's general converter records a dense storage shape and offset 0.
        # Views need the actual base pointer, storage length and offset.
        self.descriptor_lib = ctypes.CDLL("libnnopbase.so", mode=ctypes.RTLD_GLOBAL)
        integer = ctypes.c_int64
        create = self.descriptor_lib.aclCreateTensor
        create.restype = type(template.tensor)
        create.argtypes = [
            ctypes.POINTER(integer),
            ctypes.c_uint64,
            ctypes.c_int,
            ctypes.POINTER(integer),
            integer,
            ctypes.c_int,
            ctypes.POINTER(integer),
            ctypes.c_uint64,
            ctypes.c_void_p,
        ]
        storage = view.untyped_storage()
        dimensions = (integer * view.ndim)(*view.shape)
        strides = (integer * view.ndim)(*view.stride())
        storage_shape = (integer * 1)(storage.nbytes() // view.element_size())
        dtype = {
            torch.float16: 1,
            torch.bfloat16: 27,
            torch.float32: 0,
            torch.int64: 9,
        }[view.dtype]
        descriptor = create(
            dimensions,
            view.ndim,
            dtype,
            strides,
            view.storage_offset(),
            2,
            storage_shape,
            1,
            ctypes.c_void_p(storage.data_ptr()),
        )
        if not descriptor:
            raise RuntimeError("aclCreateTensor failed for the tensor view")
        # ATK owns and releases descriptors returned in its input/output packages.
        return type(template)(descriptor, storage.data_ptr(), view, view.numel())

    def init_by_input_data(self, input_data):
        self.input_backings = (
            prepare_inputs(input_data, self.options, self.dtype_name) or []
        )
        arguments, outputs = super().init_by_input_data(input_data)
        arguments[3] = ctypes.c_double(input_data.kwargs["alpha"])
        arguments[7] = ctypes.c_int64(input_data.kwargs["ignoreIndex"])
        if self.options.get("input_view"):
            for index, name in (
                (0, "hidden"),
                (1, "weight"),
                (2, "labels"),
                (4, "softmaxMax"),
                (5, "softmaxSum"),
                (6, "gradOutput"),
            ):
                arguments[index] = self.view_package(
                    input_data.kwargs[name], arguments[index]
                )
        if self.options.get("output_view"):
            self.output_view, self.output_backing = tensor_view(
                torch.zeros_like(outputs[0].pytensor), self.options["output_view"]
            )
            self.output_guard = self.output_backing == 29
            outputs[0] = self.view_package(self.output_view, outputs[0])
            arguments[8] = outputs[0]
        return arguments, outputs

    def __call__(self):
        self.backend.aclnn_x_get_workspace_size()
        self.backend.aclnn_x()

    def after_call(self, output_packages):
        if self.input_backings:
            torch.npu.synchronize(self.input_backings[0][0].device)
            if any(
                not torch.equal(backing, before)
                for backing, before in self.input_backings
            ):
                raise AssertionError("ACLNN modified readonly input storage")
        if not self.options.get("output_view"):
            return super().after_call(output_packages)
        torch.npu.synchronize(self.output_view.device)
        if not bool(torch.all(self.output_backing[self.output_guard] == 29)):
            raise AssertionError(
                "ACLNN output writeback modified storage outside the output view"
            )
        return self.output_view
