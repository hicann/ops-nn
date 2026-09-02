##!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import os
import re
import sys
import logging
from dependency_parser import OpDependenciesParser


OP_CATEGORY_LIST = []
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
SOC_VERSION_MAP = {"arch22": "ascend910b", "arch35": "ascend950", "default": "default"}
COMMON_FRAMEWORK_PLUGIN_RE = re.compile(
    r"^common/src/framework/([^/]+)_onnx_plugin\.cpp$"
)
COMMON_FRAMEWORK_UT_RE = re.compile(
    r"^common/tests/ut/framework/test_[^/]+_onnx_plugin\.cpp$"
)


class UtMatcher:
    def __init__(self, soc_version):
        self.ops = set()
        self.soc_version = soc_version
        self.splited_path = None
        self.is_txt = False
        self.is_aclnn_example = False
        self.is_prototype = False
        self.is_infershape = False

    def match(self, changed_file, is_experimental=False, soc_hit=False):
        changed_file = (
            str(os.path.relpath(changed_file, os.getenv("BASE_PATH")))
            .strip()
            .replace(os.sep, "/")
        )
        self.splited_path = changed_file.split("/")
        # common 集中目录下的 onnx 插件 framework 路径：不经 category 目录校验，按文件名取算子名
        # （common/src/framework/<op>_onnx_plugin.cpp 与 common/tests/ut/framework/test_<op>_onnx_plugin.cpp
        #   以及 <category>/<op>/tests/ut/framework/test_<op>_onnx_plugin.cpp）
        if (
            not is_experimental
            and "framework" in self.splited_path
            and changed_file.endswith("_onnx_plugin.cpp")
        ):
            # common framework 插件/UT 不带 arch 子目录，仅在 default 维度触发，避免被首个 soc matcher 抢占。
            if is_common_framework_file(changed_file) and self.soc_version != "default":
                return False
            if self._ut_match(changed_file):
                return True
        if (
            not is_experimental
            and len(self.splited_path) > 2
            and self.splited_path[0] in OP_CATEGORY_LIST
        ):
            op_name = self.splited_path[1]
            if op_name == "common":
                op_name = self.splited_path[0] + ".common"
        elif (
            is_experimental
            and len(self.splited_path) > 3
            and self.splited_path[1] in OP_CATEGORY_LIST
        ):
            op_name = self.splited_path[2]
            if op_name == "common":
                op_name = self.splited_path[1] + ".common"
        else:
            return False
        if self.soc_version == "default" and soc_hit:
            self.ops.add(op_name)
            return True
        self.common_match(changed_file)
        if self._ut_match(changed_file):
            self.ops.add(op_name)
            return True
        return False

    def common_match(self, changed_file):
        self.is_txt = changed_file.find(".txt") != -1
        self.is_aclnn_example = changed_file.find("test_aclnn_") != -1
        self.is_prototype = changed_file.find("_def.cpp") != -1
        self.is_infershape = changed_file.find("_infershape.cpp") != -1

    def _ut_match(self, changed_file):
        pass


class OpApiUt(UtMatcher):
    def _ut_match(self, changed_file):
        if self.soc_version == "default":
            return (changed_file.find("op_api") != -1 and not self.is_txt) or (
                changed_file.find("op_host") != -1 and self.is_aclnn_example
            )  # 适配api ut移除op_host目录整改前逻辑
        else:
            return (
                changed_file.find("op_api/" + self.soc_version) != -1
                and not self.is_txt
            ) or (
                changed_file.find("op_host/" + self.soc_version) != -1
                and self.is_aclnn_example
            )


class OpHostUt(UtMatcher):
    def _ut_match(self, changed_file):
        # framework onnx 插件路径，按文件名取算子名：
        #   common/src/framework/<op>_onnx_plugin.cpp                       （common 插件源）
        #   common/tests/ut/framework/test_<op>_onnx_plugin.cpp             （common 插件 UT）
        #   <category>/<op>/tests/ut/framework/test_<op>_onnx_plugin.cpp    （算子级 UT）
        #   <category>/<op>/framework/<op>_onnx_plugin.cpp                 （算子级插件源）
        m = re.search(
            r"(?:^|/)(common/src/framework|[^/]+/tests/ut/framework|[^/]+/framework)/([^/]+_onnx_plugin)\.cpp$",
            changed_file,
        )
        if m:
            op = m.group(2)
            if op.startswith("test_"):
                op = op[len("test_") :]
            if op.endswith("_onnx_plugin"):
                op = op[: -len("_onnx_plugin")]
            self.ops.add(op)
            return True
        if self.soc_version == "default":
            return (
                "op_host" in self.splited_path
                and changed_file.find("config") == -1
                and not self.is_aclnn_example
                and not self.is_txt
            )
        else:
            return (
                changed_file.find("op_host/" + self.soc_version) != -1
                and changed_file.find("config") == -1
                and not self.is_aclnn_example
                and not self.is_txt
            )


class OpKernelUt(UtMatcher):
    def _ut_match(self, changed_file):
        return changed_file.find("op_kernel") != -1


class OpGraphUt(UtMatcher):
    def _ut_match(self, changed_file):
        return (
            changed_file.find("op_graph") != -1
            and changed_file.find("_proto.h") == -1
            and changed_file.find("_infer.cpp") == -1
        )


UT_MATCHERS = {
    "op_api_ut": {
        soc_version: OpApiUt(soc_version)
        for soc_version in list(SOC_VERSION_MAP.keys())
    },
    "op_host_ut": {
        soc_version: OpHostUt(soc_version)
        for soc_version in list(SOC_VERSION_MAP.keys())
    },
    "op_graph_ut": {
        soc_version: OpGraphUt(soc_version)
        for soc_version in list(SOC_VERSION_MAP.keys())
    },
}


def file_filter(path):
    key_words = [".md", ".json", ".ini", "examples"]
    for key_word in key_words:
        if key_word in path:
            return False
    return True


def is_common_framework_plugin(changed_file):
    return COMMON_FRAMEWORK_PLUGIN_RE.match(changed_file) is not None


def is_common_framework_file(changed_file):
    return (
        is_common_framework_plugin(changed_file)
        or COMMON_FRAMEWORK_UT_RE.match(changed_file) is not None
    )


def get_common_framework_plugin_op(changed_file):
    match = COMMON_FRAMEWORK_PLUGIN_RE.match(changed_file)
    if not match:
        return None
    return match.group(1)


def check_common_framework_ut(changed_file):
    op_name = get_common_framework_plugin_op(changed_file)
    if op_name is None:
        return
    base_path = os.getenv("BASE_PATH")
    ut_file = os.path.join(
        base_path,
        "common",
        "tests",
        "ut",
        "framework",
        f"test_{op_name}_onnx_plugin.cpp",
    )
    if not os.path.exists(ut_file):
        logging.error(
            "%s is a common ONNX framework plugin, but %s is not found. "
            "Please add the corresponding framework UT to avoid CI trigger gaps.",
            changed_file,
            os.path.relpath(ut_file, base_path).replace(os.sep, "/"),
        )
        exit(1)


if __name__ == "__main__":
    changed_files_info_file = sys.argv[1]
    is_experimental = sys.argv[2] == "TRUE"
    changed_files = []
    parser = OpDependenciesParser(os.getenv("BUILD_PATH"))
    OP_CATEGORY_LIST.extend(parser.get_category_list())
    if not os.path.exists(changed_files_info_file):
        logging.error(
            "[ERROR] change file is not exist, can not get file change info in this pull request."
        )
        exit(1)
    with open(changed_files_info_file) as or_f:
        changed_files = or_f.readlines()
    for changed_file in changed_files:
        if not os.path.exists(r"{}".format(changed_file.strip())):
            continue
        changed_file = (
            str(os.path.relpath(changed_file, os.getenv("BASE_PATH")))
            .strip()
            .replace(os.sep, "/")
        )
        if file_filter(changed_file) is False:
            continue
        check_common_framework_ut(changed_file)
        changed_file = os.path.join(os.getenv("BASE_PATH"), changed_file)
        for ut_matchers in UT_MATCHERS.values():
            soc_hit = False
            for _, ut_matcher in ut_matchers.items():
                if ut_matcher.match(
                    changed_file, is_experimental=is_experimental, soc_hit=soc_hit
                ):
                    soc_hit = True
                    break
            if soc_hit:
                break
    for key, matchers in UT_MATCHERS.items():
        for soc, matcher in matchers.items():
            if matcher.ops:
                (_, reverse_op_dependencies) = parser.get_dependencies_by_ops(
                    matcher.ops
                )  # 获取反向依赖
                (op_dependencies, _) = parser.get_dependencies_by_ops(
                    reverse_op_dependencies
                )  # 根据反向依赖，获取编译依赖
                compile_ops = ";".join(list(set(op_dependencies)))
                print(
                    "%s:%s:%s:%s "
                    % (
                        key,
                        ";".join(reverse_op_dependencies),
                        compile_ops,
                        SOC_VERSION_MAP[soc],
                    )
                )
