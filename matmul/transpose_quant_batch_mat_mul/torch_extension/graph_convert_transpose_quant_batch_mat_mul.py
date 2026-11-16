# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# GE Converter for TransposeQuantBatchMatMul (TorchAir 图模式)
#
# 三种量化模式共用同一个 GE 算子：
# - MX（MXFP8 / MXFP4）：x1/x2 为 float8_e4m3fn，或 uint8 存储并分别通过 x1_dtype/x2_dtype
#   指定为 torch_npu.float4_e2m1fn_x2；x1_scale/x2_scale 为 float8_e8m0fnu；
# - FP8（K-C）：x1/x2 为 float8_e4m3fn / float8_e5m2（可任意组合），x1_scale/x2_scale 为 float32；
# - HIFP8：x1/x2 为 uint8 存储并通过 x1_dtype/x2_dtype 指定为 torch_npu.hifloat8，
#   x2_scale 为 uint64（int64 按位重解释）；
# - HIFP4：x1/x2 为 uint8 存储并通过 x1_dtype/x2_dtype 指定为 hifloat4，
#   x1_scale/x2_scale 为 hifloat4_scale（4 字节存储），64 个元素一组。
# 模式在转换时按输入/scale 的数据类型分派，因此 MXFP8、MXFP4、HIFP4 可以在同一个图里同时使用。
# x2 支持 FRACTAL_NZ 格式：MX（MXFP8/MXFP4）与 HIFP8 支持；HIFP4 目前仅支持 ND。运行时注入的 NZ 描述符
# 由 _relabel_nz_packed_chains 接回算子输入。

try:
    from typing import List, Optional

    import torch
    from torchair._ge_concrete_graph import ge_apis as ge
    from torchair._ge_concrete_graph.compat_ir import IrDef, ge_op
    from torchair._ge_concrete_graph.fx2ge_converter import (
        register_fx_node_ge_converter,
    )
    from torchair.ge import attr
    from torchair.ge._ge_graph import (
        DataType,
        Tensor,
        TensorSpec,
        _ge_dtype_to_ge_proto_dtype,
        torch_dtype_value_to_ge_type,
    )

    _TORCHAIR_AVAILABLE = True
except ImportError:
    _TORCHAIR_AVAILABLE = False


if _TORCHAIR_AVAILABLE:
    # Keep older torchair versions importable: HIFP4 only has to fail when an
    # HIFP4 call actually reaches the converter, not when this module loads.
    _HIFLOAT4_GE_DTYPE = getattr(DataType, "DT_HIFLOAT4", None)
    _HIFLOAT4_SCALE_GE_DTYPE = getattr(DataType, "DT_HIFLOAT4_SCALE", None)
    _HIFLOAT4_IR = ", DT_HIFLOAT4" if _HIFLOAT4_GE_DTYPE is not None else ""
    _HIFLOAT4_SCALE_IR = (
        ", DT_HIFLOAT4_SCALE" if _HIFLOAT4_SCALE_GE_DTYPE is not None else ""
    )
    _GROUP_DIM = 3
    _GROUP_VALUE_MAX = 65535
    _GROUP_M_SHIFT = 32
    _GROUP_N_SHIFT = 16
    _GROUP_K_SHIFT = 0
    _MX_GROUP_K = 32

    # attr dtype (raw ACL code) -> GE output dtype
    _OUTPUT_DTYPE_MAP = {
        1: DataType.DT_FLOAT16,
        27: DataType.DT_BF16,
        34: DataType.DT_HIFLOAT8,
    }
    _HIFLOAT8_OUTPUT = 34
    # HIFLOAT8 在 torch 侧以 uint8 承载
    _HIFLOAT8_STORAGE_DTYPE = DataType.DT_UINT8

    _INPUT_DTYPE_SET = {
        DataType.DT_FLOAT8_E4M3FN,
        DataType.DT_FLOAT8_E5M2,
        DataType.DT_FLOAT4_E2M1,
        DataType.DT_HIFLOAT8,
    }
    if _HIFLOAT4_GE_DTYPE is not None:
        _INPUT_DTYPE_SET.add(_HIFLOAT4_GE_DTYPE)

    _TQBMM_IR = (
        IrDef("TransposeQuantBatchMatMul")
        .input(
            "x1",
            "DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2, DT_FLOAT4_E2M1, DT_HIFLOAT8"
            + _HIFLOAT4_IR,
        )
        .input(
            "x2",
            "DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2, DT_FLOAT4_E2M1, DT_HIFLOAT8"
            + _HIFLOAT4_IR,
        )
        .optional_input("bias", "DT_FLOAT, DT_FLOAT16, DT_BF16")
        .optional_input(
            "x1_scale", "DT_FLOAT, DT_FLOAT8_E8M0, DT_UINT64" + _HIFLOAT4_SCALE_IR
        )
        .optional_input(
            "x2_scale", "DT_FLOAT, DT_FLOAT8_E8M0, DT_UINT64" + _HIFLOAT4_SCALE_IR
        )
        .required_attr("dtype", attr.Int)
        .attr("group_size", attr.Int(0))
        .attr("perm_x1", attr.ListInt([1, 0, 2]))
        .attr("perm_x2", attr.ListInt([0, 1, 2]))
        .attr("perm_y", attr.ListInt([1, 0, 2]))
        .attr("batch_split_factor", attr.Int(1))
        .output("y", "DT_FLOAT16, DT_BF16, DT_HIFLOAT8")
    )

    def _check_and_get_group_size(group_sizes):
        if len(group_sizes) != _GROUP_DIM:
            raise RuntimeError(
                "group_sizes must be a list with 3 elements, actual: %s"
                % str(group_sizes)
            )
        group_m, group_n, group_k = (int(value) for value in group_sizes)
        for value in (group_m, group_n, group_k):
            if value < 0 or value > _GROUP_VALUE_MAX:
                raise RuntimeError(
                    "group_sizes value must be in range [0, 65535], actual: %s"
                    % str(group_sizes)
                )
        return (
            (group_m << _GROUP_M_SHIFT)
            + (group_n << _GROUP_N_SHIFT)
            + (group_k << _GROUP_K_SHIFT)
        )

    def _normalize_dtype_value(dtype_value):
        """兼容传入原始ACL dtype编码（<256）的场景，与单算子模式保持一致。"""
        if isinstance(dtype_value, int) and dtype_value < 256:
            return dtype_value + 256
        return dtype_value

    # ACL code (offset by 256 in the torch attr) -> GE dtype. The torchair
    # lookup table may predate HIFP4, so keep an explicit fallback.
    _HIFP4_ACL_CODES = (42, 298)
    _HIFP4_SCALE_ACL_CODES = (43, 299)

    def _resolve_dtype(tensor, dtype_override):
        if dtype_override is not None:
            normalized = _normalize_dtype_value(dtype_override)
            if normalized in _HIFP4_ACL_CODES:
                if _HIFLOAT4_GE_DTYPE is None:
                    raise RuntimeError("current torchair does not support HIFP4")
                return _HIFLOAT4_GE_DTYPE
            if normalized in _HIFP4_SCALE_ACL_CODES:
                if _HIFLOAT4_SCALE_GE_DTYPE is None:
                    raise RuntimeError("current torchair does not support HIFP4")
                return _HIFLOAT4_SCALE_GE_DTYPE
            try:
                return torch_dtype_value_to_ge_type(dtype_override)
            except RuntimeError:
                try:
                    return torch_dtype_value_to_ge_type(normalized)
                except RuntimeError:
                    pass
        return None if tensor is None else tensor.dtype

    def _bitcast_to(tensor, ge_dtype):
        """按位重解释为 ge_dtype（4bit 目标会显式将物理最后一维翻倍为逻辑长度）。"""
        if tensor is None or ge_dtype is None or tensor.dtype == ge_dtype:
            return tensor
        if ge_dtype == DataType.DT_FLOAT4_E2M1 or (
            _HIFLOAT4_GE_DTYPE is not None and ge_dtype == _HIFLOAT4_GE_DTYPE
        ):
            # 与 MxToBlockMxQuant 等既有实现保持一致：显式按 2 倍逻辑长度 reshape，
            # 不依赖 GE Bitcast 对 4bit 类型的 shape 推导行为。
            bit_shape = [1] * (tensor.rank - 1) + [2]
            shape_logical = ge.Mul(
                ge.Shape(tensor), ge.Const(bit_shape, dtype=DataType.DT_INT32)
            )
            out = ge.Reshape(ge.Bitcast(tensor, type=ge_dtype), shape_logical)
        else:
            out = ge.Bitcast(tensor, type=ge_dtype)
        try:
            out.desc.dtype = _ge_dtype_to_ge_proto_dtype(ge_dtype)
        except ValueError:
            # 个别类型（如 DT_UINT64）没有 proto 映射，由 GE 按算子原型推导
            pass
        return out

    def _is_mx_mode(x1_scale, x2_scale, x1_scale_dtype, x2_scale_dtype, group_size):
        for scale, override in (
            (x1_scale, x1_scale_dtype),
            (x2_scale, x2_scale_dtype),
        ):
            if scale is None:
                continue
            scale_dtype = _resolve_dtype(scale, override)
            if scale_dtype == DataType.DT_FLOAT8_E8M0:
                return True
            # e8m0 以 uint8/int8 承载且无 dtype override 时，按 MX 的 group_k=32 辅助判定；
            # 该分支只用于 dtype 模糊场景，避免 FP8（group_k=32 且 float scale）被误判为 MX。
            if (
                scale_dtype in (DataType.DT_UINT8, DataType.DT_INT8)
                and (group_size & _GROUP_VALUE_MAX) == _MX_GROUP_K
            ):
                return True
        return False

    @register_fx_node_ge_converter(
        torch.ops.cann_ops_nn.transpose_quant_batch_mat_mul.default
    )
    def convert_transpose_quant_batch_mat_mul(
        x1: Tensor,
        x2: Tensor,
        *,
        dtype: int,
        bias: Optional[Tensor] = None,
        x1_scale: Optional[Tensor] = None,
        x2_scale: Optional[Tensor] = None,
        group_sizes: Optional[List[int]] = None,
        perm_x1: Optional[List[int]] = None,
        perm_x2: Optional[List[int]] = None,
        perm_y: Optional[List[int]] = None,
        batch_split_factor: Optional[int] = None,
        x1_dtype: Optional[int] = None,
        x2_dtype: Optional[int] = None,
        x1_scale_dtype: Optional[int] = None,
        x2_scale_dtype: Optional[int] = None,
        meta_outputs: TensorSpec = None,
    ):
        output_ge_dtype = _OUTPUT_DTYPE_MAP.get(dtype)
        if output_ge_dtype is None:
            raise RuntimeError(
                "dtype only supports float16(1), bfloat16(27) and hifloat8(34), but got %s"
                % dtype
            )

        group_size = 0
        if group_sizes is not None:
            group_size = _check_and_get_group_size(group_sizes)
        batch_split = 1 if batch_split_factor is None else int(batch_split_factor)
        if batch_split != 1:
            raise RuntimeError(
                "batch_split_factor only supports 1, but got %s" % batch_split
            )
        perm_x1 = [1, 0, 2] if perm_x1 is None else [int(v) for v in perm_x1]
        perm_x2 = [0, 1, 2] if perm_x2 is None else [int(v) for v in perm_x2]
        perm_y = [1, 0, 2] if perm_y is None else [int(v) for v in perm_y]

        # MXFP4/HIFP8/HIFP4 的 Tensor 以 uint8 承载，必须通过 x1_dtype/x2_dtype 指定逻辑类型，
        # 这里按 override 做按位重解释；FP8 与 MXFP8 的输入 dtype 已与算子一致，保持原样。
        x1_ge_dtype = _resolve_dtype(x1, x1_dtype)
        x2_ge_dtype = _resolve_dtype(x2, x2_dtype)
        if x1_ge_dtype not in _INPUT_DTYPE_SET:
            raise RuntimeError(
                "unsupported x1 dtype %s, please pass x1_dtype to specify the logical dtype "
                "(float4_e2m1fn_x2 / hifloat8 / hifloat4) for packed inputs"
                % str(x1_ge_dtype)
            )
        if x2_ge_dtype not in _INPUT_DTYPE_SET:
            raise RuntimeError(
                "unsupported x2 dtype %s, please pass x2_dtype to specify the logical dtype "
                "(float4_e2m1fn_x2 / hifloat8 / hifloat4) for packed inputs"
                % str(x2_ge_dtype)
            )
        x1 = _bitcast_to(x1, x1_ge_dtype)
        x2 = _bitcast_to(x2, x2_ge_dtype)

        is_hifp4 = _HIFLOAT4_GE_DTYPE is not None and (
            x1_ge_dtype == _HIFLOAT4_GE_DTYPE or x2_ge_dtype == _HIFLOAT4_GE_DTYPE
        )
        is_hifp8 = (
            x1_ge_dtype == DataType.DT_HIFLOAT8 or x2_ge_dtype == DataType.DT_HIFLOAT8
        )
        if is_hifp4:
            if x1_ge_dtype != _HIFLOAT4_GE_DTYPE or x2_ge_dtype != _HIFLOAT4_GE_DTYPE:
                raise RuntimeError("x1 and x2 must both be hifloat4 for HIFP4 input")
            if dtype == _HIFLOAT8_OUTPUT:
                raise RuntimeError(
                    "hifp4 output supports only float16(1) and bfloat16(27), but got %s"
                    % dtype
                )
            for name, scale, override in (
                ("x1_scale", x1_scale, x1_scale_dtype),
                ("x2_scale", x2_scale, x2_scale_dtype),
            ):
                if scale is None:
                    raise RuntimeError("%s is required for HIFP4 input" % name)
                scale_dtype = _resolve_dtype(scale, override)
                if scale_dtype != _HIFLOAT4_SCALE_GE_DTYPE:
                    raise RuntimeError(
                        "%s must be hifloat4_scale for HIFP4 input, got dtype %s"
                        % (name, str(scale_dtype))
                    )
                if scale.dtype != DataType.DT_FLOAT:
                    raise RuntimeError(
                        "%s must use float32 storage for HIFP4 input, got dtype %s"
                        % (name, str(scale.dtype))
                    )
                scale = _bitcast_to(scale, _HIFLOAT4_SCALE_GE_DTYPE)
                if name == "x1_scale":
                    x1_scale = scale
                else:
                    x2_scale = scale
        elif is_hifp8:
            # HIFP8: x2_scale 为 uint64 位模式，int64 传入时按位重解释
            for name, scale, override in (
                ("x1_scale", x1_scale, x1_scale_dtype),
                ("x2_scale", x2_scale, x2_scale_dtype),
            ):
                if scale is None:
                    continue
                scale_dtype = _resolve_dtype(scale, override)
                if scale_dtype not in (DataType.DT_UINT64, DataType.DT_INT64):
                    raise RuntimeError(
                        "%s must carry the uint64 bit pattern for HIFP8, got dtype %s"
                        % (name, str(scale_dtype))
                    )
                if scale_dtype == DataType.DT_INT64:
                    scale_dtype = DataType.DT_UINT64
                scale = _bitcast_to(scale, scale_dtype)
                if name == "x1_scale":
                    x1_scale = scale
                else:
                    x2_scale = scale
        else:
            is_mx = _is_mx_mode(
                x1_scale, x2_scale, x1_scale_dtype, x2_scale_dtype, group_size
            )
            for name, scale, override in (
                ("x1_scale", x1_scale, x1_scale_dtype),
                ("x2_scale", x2_scale, x2_scale_dtype),
            ):
                if scale is None:
                    continue
                scale_dtype = _resolve_dtype(scale, override)
                if is_mx:
                    # MX：scale 为 float8_e8m0fnu；兼容以 uint8/int8 承载 e8m0 的场景
                    if scale_dtype not in (
                        DataType.DT_FLOAT8_E8M0,
                        DataType.DT_UINT8,
                        DataType.DT_INT8,
                    ):
                        raise RuntimeError(
                            "%s must be float8_e8m0fnu for MX input, got dtype %s"
                            % (name, str(scale_dtype))
                        )
                    scale = _bitcast_to(scale, DataType.DT_FLOAT8_E8M0)
                else:
                    # FP8（K-C）：scale 为 float32
                    if scale_dtype != DataType.DT_FLOAT:
                        raise RuntimeError(
                            "%s must be float32 for FP8 input, got dtype %s"
                            % (name, str(scale_dtype))
                        )
                if name == "x1_scale":
                    x1_scale = scale
                else:
                    x2_scale = scale

        inputs = {"x1": x1, "x2": x2}
        if bias is not None:
            inputs["bias"] = bias
        if x1_scale is not None:
            inputs["x1_scale"] = x1_scale
        if x2_scale is not None:
            inputs["x2_scale"] = x2_scale

        y = ge_op(
            op_type="TransposeQuantBatchMatMul",
            inputs=inputs,
            attrs={
                "dtype": attr.Int(dtype),
                "group_size": attr.Int(group_size),
                "perm_x1": attr.ListInt(perm_x1),
                "perm_x2": attr.ListInt(perm_x2),
                "perm_y": attr.ListInt(perm_y),
                "batch_split_factor": attr.Int(batch_split),
            },
            outputs=["y"],
            ir=_TQBMM_IR,
        )

        if dtype == _HIFLOAT8_OUTPUT:
            # HIFLOAT8 输出在 torch 侧以 uint8 承载
            y = ge.Bitcast(y, type=_HIFLOAT8_STORAGE_DTYPE)
            y.desc.dtype = _ge_dtype_to_ge_proto_dtype(_HIFLOAT8_STORAGE_DTYPE)
        else:
            y.desc.dtype = _ge_dtype_to_ge_proto_dtype(output_ge_dtype)
        return y

    _FP4_PROTO_DTYPE = _ge_dtype_to_ge_proto_dtype(DataType.DT_FLOAT4_E2M1)
    _UINT8_PROTO_DTYPE = _ge_dtype_to_ge_proto_dtype(DataType.DT_UINT8)
    _HIFLOAT8_PROTO_DTYPE = _ge_dtype_to_ge_proto_dtype(DataType.DT_HIFLOAT8)
    try:
        _HIFLOAT4_PROTO_DTYPE = (
            _ge_dtype_to_ge_proto_dtype(_HIFLOAT4_GE_DTYPE)
            if _HIFLOAT4_GE_DTYPE is not None
            else None
        )
    except (AttributeError, ValueError, KeyError):
        _HIFLOAT4_PROTO_DTYPE = None
    _ND_FORMAT = 2

    def _relabel_nz_packed_chains(graph):
        """把 torchair 注入的 NZ 描述符接回算子的输入上。

        torchair 在 FX->GE 转换之后才把运行时存储格式（如 FRACTAL_NZ）写到输入 Data 节点上；
        打包输入插入的 uint8 -> 逻辑类型链把 Data 藏在中间，注入的 NZ 描述符到不了
        TransposeQuantBatchMatMul，算子会按 ND 读 NZ 布局的权重导致精度失败。
        这里把链尾的输入改回 Data 直连，与 MXFP8 直连 NZ 输入时 torchair 的形态保持一致：
        - MXFP4：uint8 -> float4_e2m1 链为 Bitcast -> Reshape，dtype 就地重标为
          float4_e2m1，打包维度的 origin_shape 翻倍；
        - HIFP8：uint8 -> hifloat8 链只有一个 Bitcast，dtype 就地重标为 hifloat8，
          shape 不变。
        """
        consumers = {}
        for op in graph.op:
            for inp in op.input:
                consumers.setdefault(inp.split(":")[0], []).append(op)

        def _real_consumers(op_name):
            # 形状计算节点（Shape/Size）只读元数据，不影响 dtype/布局重标
            return [
                c for c in consumers.get(op_name, []) if c.type not in ("Shape", "Size")
            ]

        dropped = set()
        for op in graph.op:
            if op.type not in ("Data", "RefData") or len(op.output_desc) == 0:
                continue
            desc = op.output_desc[0]
            if desc.dtype != _UINT8_PROTO_DTYPE:
                continue
            fmt = (
                desc.attr["format_for_int"].i
                if "format_for_int" in desc.attr
                else _ND_FORMAT
            )
            if fmt == _ND_FORMAT:
                continue
            chain = _real_consumers(op.name)
            if len(chain) != 1 or chain[0].type != "Bitcast":
                continue
            bitcast = chain[0]
            if not bitcast.input or bitcast.input[0].split(":")[0] != op.name:
                continue

            # HIFP8 的 Bitcast 输出 dtype 在 _bitcast_to 中已重标为 hifloat8，
            # 据此与 4bit 的 Bitcast -> Reshape 链区分。
            bitcast_dtype = (
                bitcast.output_desc[0].dtype if len(bitcast.output_desc) > 0 else None
            )
            reshape = None
            if bitcast_dtype == _HIFLOAT8_PROTO_DTYPE:
                new_dtype = _HIFLOAT8_PROTO_DTYPE
                tail = bitcast
            else:
                chain = _real_consumers(bitcast.name)
                if len(chain) != 1 or chain[0].type != "Reshape":
                    continue
                reshape = chain[0]
                if not reshape.input or reshape.input[0].split(":")[0] != bitcast.name:
                    continue
                new_dtype = (
                    _HIFLOAT4_PROTO_DTYPE
                    if _HIFLOAT4_PROTO_DTYPE is not None
                    and bitcast_dtype == _HIFLOAT4_PROTO_DTYPE
                    else _FP4_PROTO_DTYPE
                )
                tail = reshape

            chain = _real_consumers(tail.name)
            if len(chain) != 1 or chain[0].type != "TransposeQuantBatchMatMul":
                continue
            tqbmm = chain[0]
            target = None
            for idx, inp in enumerate(tqbmm.input):
                if inp.split(":")[0] == tail.name:
                    target = idx
                    break
            if target is None:
                continue

            if reshape is not None:
                origin = (
                    desc.attr["origin_shape"].list.i
                    if "origin_shape" in desc.attr
                    else None
                )
                if origin is None or len(origin) == 0 or origin[-1] <= 0:
                    continue
                origin[-1] *= 2

            desc.dtype = new_dtype
            if len(op.input_desc) > 0:
                op.input_desc[0].dtype = new_dtype
            tqbmm.input[target] = "%s:0" % op.name
            dropped.add(bitcast.name)
            if reshape is not None:
                dropped.add(reshape.name)

        for idx in range(len(graph.op) - 1, -1, -1):
            if graph.op[idx].name in dropped:
                del graph.op[idx]
        return bool(dropped)

    def _keep_injected_nz_format():
        """让上面的修复挂在 torchair 的格式注入之后执行（torchair 版本不带该钩子时静默跳过）。"""
        try:
            from torchair._ge_concrete_graph import fx2ge_converter as _fx2ge

            original = getattr(_fx2ge, "_update_internal_format_from_inputs", None)
            if original is None or getattr(_fx2ge, "_tqbmm_nz_chain_patched", False):
                return
            try:
                from torchair.core.utils import logger as _logger
            except Exception:
                _logger = None

            def _update_internal_format_from_inputs(graph, runtime_inputs):
                original(graph, runtime_inputs)
                try:
                    if _relabel_nz_packed_chains(graph) and _logger is not None:
                        _logger.debug(
                            "TQBMM: reconnected injected NZ descriptor to the op input"
                        )
                except Exception as err:
                    if _logger is not None:
                        _logger.warning(
                            "TQBMM: keeping injected NZ descriptor failed: %s", err
                        )

            _fx2ge._update_internal_format_from_inputs = (
                _update_internal_format_from_inputs
            )
            _fx2ge._tqbmm_nz_chain_patched = True
        except Exception:
            pass

    _keep_injected_nz_format()

else:

    def convert_transpose_quant_batch_mat_mul(*args, **kwargs):
        raise RuntimeError(
            "TransposeQuantBatchMatMul graph converter: torchair is not available."
        )
