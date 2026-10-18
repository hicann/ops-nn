# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Build the real merge_graph_headers target without CANN or a compiler."""

from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest


REPO = Path(__file__).resolve().parents[2]


def run(*command):
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def prototype(name, value):
    return f"REG_OP({name})\n.ATTR(value, Int, {value})\nOP_END_FACTORY_REG({name})\n"


def prepare_project(root):
    for relative in ("cmake/gen_ops_info.cmake", "scripts/util/merge_proto.py"):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / relative, destination)
    normal = root / "sample_proto.h"
    normal.write_text(prototype("SampleOp", 1), encoding="utf-8")
    extended = root / "common/inc/op_graph/op_nn_proto_extend.h"
    extended.parent.mkdir(parents=True)
    extended.write_text(prototype("ExtendedOp", 10), encoding="utf-8")
    (root / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.16)\n"
        "project(merge_fixture LANGUAGES NONE)\n"
        "set(GRAPH_PLUGIN_NAME fixture)\n"
        f'set(ASCEND_PYTHON_EXECUTABLE "{Path(sys.executable).as_posix()}")\n'
        "add_library(fixture_proto_headers INTERFACE)\n"
        'target_sources(fixture_proto_headers INTERFACE "${CMAKE_SOURCE_DIR}/sample_proto.h")\n'
        'include("${CMAKE_SOURCE_DIR}/cmake/gen_ops_info.cmake")\n'
        'file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/graph")\n'
        'merge_graph_headers(TARGET merged_proto OUT_DIR "${CMAKE_BINARY_DIR}/graph")\n',
        encoding="utf-8",
    )
    return normal, extended, root / "scripts/util/merge_proto.py"


@pytest.mark.parametrize("changed_input", ["normal", "extended", "generator"])
def test_input_update_regenerates_header_and_cpp(tmp_path, changed_input):
    source = tmp_path / "source"
    source.mkdir()
    normal, extended, generator = prepare_project(source)
    build = tmp_path / "build"
    run("cmake", "-S", str(source), "-B", str(build), "-G", "Unix Makefiles")
    run("cmake", "--build", str(build), "--target", "merged_proto")
    header = build / "graph/ops_proto_nn.h"
    cpp = build / "graph/ops_proto_nn.cpp"
    assert "Int, 1)" in header.read_text(encoding="utf-8")
    before = header.stat().st_mtime_ns
    run("cmake", "--build", str(build), "--target", "merged_proto")
    assert header.stat().st_mtime_ns == before

    time.sleep(1.1)  # Makefiles can compare timestamps at whole-second precision.
    if changed_input == "normal":
        normal.write_text(prototype("SampleOp", 2), encoding="utf-8")
        expected = "Int, 2)"
    elif changed_input == "extended":
        extended.write_text(prototype("ExtendedOp", 20), encoding="utf-8")
        expected = "Int, 20)"
    else:
        text = generator.read_text(encoding="utf-8")
        generator.write_text(
            text.replace(
                "#define OP_NN_PROTO_H_", "#define OP_NN_PROTO_H_\n// Updated generator"
            ),
            encoding="utf-8",
        )
        expected = "// Updated generator"

    run("cmake", "--build", str(build), "--target", "merged_proto")
    assert expected in header.read_text(encoding="utf-8")
    assert cpp.read_bytes() == header.read_bytes()
