# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import importlib
import logging

_logger = logging.getLogger(__name__)

_failed_ops = {}


class OpLoadError(AttributeError):
    """算子加载失败；直接访问抛出并携带根因。"""


def _load_op_module(*names):
    """导入 names[0] 模块并绑定算子名；失败仅记录不中断兄弟算子，访问时经 __getattr__ 回抛根因。"""
    module_name = names[0]
    try:
        module = importlib.import_module("." + module_name, __name__)
    except Exception as _load_err:
        for name in names:
            _failed_ops[name] = _load_err
        _logger.error(
            "Failed to load op '%s' [%s]: %s",
            module_name,
            type(_load_err).__name__,
            _load_err,
        )
        return
    for name in names:
        op = getattr(module, name, None)
        if op is None:
            # 模块可导入但未导出同名符号（重构改名/命名空间包等）：
            # 记入 _failed_ops 并回抛，而不是静默绑定模块对象掩盖故障
            _failed_ops[name] = AttributeError(
                "module '%s' has no attribute '%s'" % (module_name, name)
            )
            _logger.error(
                "Failed to load op '%s' [%s]: %s",
                name,
                "AttributeError",
                _failed_ops[name],
            )
            continue
        globals()[name] = op


_load_op_module("quant_matmul_activation_quant")

_load_op_module("flat_quant")

_load_op_module("rotate_quant")

_load_op_module("transpose_batch_mat_mul")

_load_op_module("transpose_quant_batch_mat_mul")

_load_op_module("confidence_head_fused")

__all__ = [
    name
    for name in (
        "quant_matmul_activation_quant",
        "flat_quant",
        "rotate_quant",
        "transpose_batch_mat_mul",
        "transpose_quant_batch_mat_mul",
        "confidence_head_fused",
    )
    if name not in _failed_ops
]


def __getattr__(name):
    _err = _failed_ops.get(name)
    if _err is not None:
        raise OpLoadError(
            "op '%s' failed to load [%s]: %s" % (name, type(_err).__name__, _err)
        ) from _err
    raise AttributeError("module %r has no attribute %r" % (__name__, name))
