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
import sys
import importlib
import logging

logger = logging.getLogger(__name__)


def _discover_ops():
    ops = {}
    current_dir = os.path.dirname(os.path.abspath(__file__))

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
                ops[sub] = sub_path

    return ops


for _name, _path in _discover_ops().items():
    _has_module_file = os.path.isfile(os.path.join(_path, _name + ".py"))
    _search_path = os.path.dirname(_path) if not _has_module_file else _path
    if _search_path not in sys.path:
        sys.path.insert(0, _search_path)

    try:
        _mod = importlib.import_module(_name)
        if hasattr(_mod, _name):
            globals()[_name] = getattr(_mod, _name)
        else:
            globals()[_name] = _mod
    except (ImportError, RuntimeError) as e:
        logger.warning("Failed to load op '%s': %s", _name, e)
    try:
        _gmod = importlib.import_module("graph_convert_%s" % _name)
        _func = "convert_%s" % _name
        if hasattr(_gmod, _func):
            globals()[_func] = getattr(_gmod, _func)
    except ImportError:
        pass


try:
    del _discover_ops, _name, _path, _mod, _gmod, _func
except NameError:
    pass
