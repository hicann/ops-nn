# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import os
import importlib
import logging

logger = logging.getLogger(__name__)


def _discover_ops_from_entry_points():
    ops = {}
    try:
        from importlib.metadata import entry_points

        eps = entry_points(group="cann_ops_nn.ops")
        for ep in eps:
            ops[ep.name] = ep.value
    except Exception:
        pass
    return ops


def _discover_ops_from_dir():
    ops = {}
    current_dir = os.path.dirname(os.path.abspath(__file__))
    package_name = __name__

    _skip = frozenset(("graph_convert",))
    for entry in sorted(os.listdir(current_dir)):
        if entry in _skip or entry.startswith((".", "_", "__")):
            continue
        entry_path = os.path.join(current_dir, entry)
        if not os.path.isdir(entry_path):
            continue
        for sub in sorted(os.listdir(entry_path)):
            if sub in _skip or sub.startswith((".", "_", "__")):
                continue
            sub_path = os.path.join(entry_path, sub)
            sub_init = os.path.join(sub_path, "__init__.py")
            if os.path.isdir(sub_path) and os.path.isfile(sub_init) and sub not in ops:
                ops[sub] = "%s.%s.%s" % (package_name, entry, sub)

    return ops


def _load_op(name, target):
    if ":" in target:
        module_path, attr_name = target.split(":", 1)
        try:
            _mod = importlib.import_module(module_path)
            if hasattr(_mod, attr_name):
                globals()[name] = getattr(_mod, attr_name)
            elif hasattr(_mod, name):
                globals()[name] = getattr(_mod, name)
            else:
                globals()[name] = _mod
        except (ImportError, RuntimeError) as e:
            logger.warning("Failed to load op '%s': %s", name, e)
        return
    try:
        _mod = importlib.import_module(target)
        if hasattr(_mod, name):
            globals()[name] = getattr(_mod, name)
        else:
            globals()[name] = _mod
        try:
            _gmod = importlib.import_module(
                "%s.graph_convert_%s" % (target.rsplit(".", 1)[0], name)
            )
            _func = "convert_%s" % name
            if hasattr(_gmod, _func):
                globals()[_func] = getattr(_gmod, _func)
        except ImportError:
            pass
    except (ImportError, RuntimeError) as e:
        logger.warning("Failed to load op '%s': %s", name, e)


_ep_ops = _discover_ops_from_entry_points()
_dir_ops = _discover_ops_from_dir()

_all_ops = dict(_dir_ops)
_all_ops.update(_ep_ops)

for _name, _target in _all_ops.items():
    _load_op(_name, _target)

try:
    del _discover_ops_from_entry_points, _discover_ops_from_dir, _load_op
    del _ep_ops, _dir_ops, _all_ops, _name, _target
except NameError:
    pass
