# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Ordinary compiler options must retain their complete literal values."""

import importlib.util
from pathlib import Path


def load_file(relative, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    path = root / relative
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location("metadata_helper", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compile_include_path_containing_oom_is_forwarded(tmp_path, monkeypatch):
    helper = load_file("scripts/util/ascendc_gen_options.py", monkeypatch)
    config = tmp_path / "compile.ini"
    options = ["-I/project/room/include", "-Dcustom_oom=1", "-O2"]
    helper.gen_compile_options(str(config), "Demo", "ascend950", options)
    assert config.read_text() == "Demo,ascend950," + ";".join(options) + "\n"


def test_real_debug_options_still_route_to_opc_config(tmp_path, monkeypatch):
    helper = load_file("scripts/util/ascendc_gen_options.py", monkeypatch)
    config = tmp_path / "compile.ini"
    helper.gen_compile_options(
        str(config),
        "Demo",
        "ascend950",
        ["--oom", "--save-temp-files", "--tiling_key=1,3", "-O2"],
    )
    assert config.read_text() == "Demo,ascend950,-O2\n"
    assert (
        tmp_path / "custom_opc_options.ini"
    ).read_text() == "Demo@ascend950@--op_debug_config=oom;dump_cce@--tiling_key=1;3\n"
