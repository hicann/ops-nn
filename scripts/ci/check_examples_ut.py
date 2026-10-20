#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""检测算子的 examples 与 ut 是否缺失或为空。

算子判定：目录下存在 op_host 或 op_kernel 子目录，op_host 下存在def文件，
且目录名不属于公共库（common、*_common 等）；无def文件为接口类算子，不参与统计。
example 判定：仅扫描暴露aclnn接口的算子（op_api目录存在aclnn_*文件，或CMakeLists声明
ACLNNTYPE aclnn/aclnn_inner）；examples 目录不存在，目录内（含子目录）无任何源码文件，
或仅有 arch35 一个子目录且def文件声明支持ascend910b/ascend910_93（纯950等非910芯片算子不统计）。
ut 判定：tests/ut 目录不存在，或目录内除 CMakeLists.txt 外无任何文件。
检测结果按类别汇总写入 build/check_examples_ut.csv，
列为: 算子路径, 算子名, OPTYPE(def文件中OP_ADD声明的值), 类别。
指定pr_filelist时（-p参数或首个位置参数为文件），仅检查pr涉及的算子，
并基于build/tmp/ops_config.txt的依赖图（同build.sh的parse_op_dependencies）做正向+反向依赖展开。
pr_filelist仅接受git diff --name-only产出的纯路径列表（仓内约定格式，
与local_build.sh/ops_st_test.sh/check_example.sh等消费方一致），不支持带状态列的格式。
"""

import argparse
import csv
import glob
import os
import re
import sys
from functools import lru_cache
from typing import List, NamedTuple, Optional, Set, Tuple

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "util")
)
from changed_ops import (
    expand_ops_with_dependencies,
    iter_op_dirs,
    parse_filelist_ops,
)

# example 内视为有效内容的文件后缀
EXAMPLE_EXTS = (".cpp", ".cc", ".py")


# 问题类别
ISSUE_EXAMPLES_MISSING = "examples缺失/为空"
ISSUE_ARCH35_ONLY = "examples仅含arch35目录"
ISSUE_UT_MISSING = "ut缺失/为空"


class OpIssue(NamedTuple):
    """算子缺失项。"""

    op_path: str
    missing_examples: bool
    arch35_only_examples: bool
    missing_ut: bool


def issue_categories(issue: OpIssue) -> List[str]:
    """返回算子命中的问题类别列表。"""
    categories = []
    if issue.missing_examples:
        categories.append(ISSUE_EXAMPLES_MISSING)
    if issue.arch35_only_examples:
        categories.append(ISSUE_ARCH35_ONLY)
    if issue.missing_ut:
        categories.append(ISSUE_UT_MISSING)
    return categories


def has_file_with_ext(root: str, exts: Tuple[str, ...]) -> bool:
    """判断目录内（递归）是否存在指定后缀的文件。"""
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith(exts):
                return True
    return False


# ut 目录内视为有效测试内容的源码后缀，README/.gitkeep/测试数据等非源码文件不算有效UT
UT_SOURCE_EXTS = (".cpp", ".cc", ".c", ".cxx", ".py")


def has_ut_content(root: str) -> bool:
    """判断 ut 目录内是否存在测试源码文件（递归）。

    仅README、测试数据JSON、占位文件等非源码内容不算有效UT，与
    ut.cmake收集test_*.cpp用例的口径对齐。
    """
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith(UT_SOURCE_EXTS):
                return True
    return False


def is_arch35_only_examples(examples_dir: str) -> bool:
    """判断 examples 目录下是否仅有 arch35 一个子目录（无顶层示例文件及其他子目录，CMakeLists.txt除外）。"""
    entries = [entry for entry in os.listdir(examples_dir) if entry != "CMakeLists.txt"]
    return (
        len(entries) == 1
        and entries[0] == "arch35"
        and os.path.isdir(os.path.join(examples_dir, "arch35"))
    )


# aclnn接口声明：CMakeLists中ACLNNTYPE取aclnn/aclnn_inner视为暴露aclnn接口
ACLNN_TYPE_RE = re.compile(r"ACLNNTYPE\s+aclnn(_inner)?\b")


def has_aclnn_entry(op_path: str) -> bool:
    """判断算子是否暴露aclnn接口（examples检测前置条件）。

    op_api目录存在aclnn_*文件，或CMakeLists声明ACLNNTYPE aclnn/aclnn_inner视为有aclnn入口；
    两者皆无的算子不扫描examples。
    """
    op_api_dir = os.path.join(op_path, "op_api")
    if os.path.isdir(op_api_dir):
        for entry in os.listdir(op_api_dir):
            if entry.startswith("aclnn_"):
                return True
    try:
        with open(
            os.path.join(op_path, "CMakeLists.txt"), encoding="utf-8", errors="ignore"
        ) as file:
            return bool(ACLNN_TYPE_RE.search(file.read()))
    except OSError:
        return False


def check_op(op_path: str) -> Optional[OpIssue]:
    """检查单个算子，无问题返回 None。"""
    missing_examples = False
    arch35_only_examples = False
    # 未暴露aclnn接口的算子不需要examples，不扫描（不报缺失、不计arch35-only）
    if has_aclnn_entry(op_path):
        missing_examples = True
        examples_dir = os.path.join(op_path, "examples")
        if os.path.isdir(examples_dir):
            missing_examples = not has_file_with_ext(examples_dir, EXAMPLE_EXTS)
            # def文件仅声明950等非910芯片的算子，仅有arch35 examples属正常，不视为缺失
            arch35_only_examples = (
                not missing_examples
                and is_arch35_only_examples(examples_dir)
                and def_supports_910(op_path)
            )

    missing_ut = True
    ut_dir = os.path.join(op_path, "tests", "ut")
    if os.path.isdir(ut_dir):
        missing_ut = not has_ut_content(ut_dir)

    if missing_examples or arch35_only_examples or missing_ut:
        return OpIssue(op_path, missing_examples, arch35_only_examples, missing_ut)
    return None


def check_all(
    repo_root: str, op_dirs: Optional[List[str]] = None
) -> Tuple[List[OpIssue], int]:
    """检查算子，op_dirs为None时全仓扫描，否则仅检查指定算子目录，返回问题列表与算子总数。"""
    issues: List[OpIssue] = []
    total = 0
    op_paths = op_dirs if op_dirs is not None else iter_op_dirs(repo_root)
    for op_path in op_paths:
        total += 1
        issue = check_op(op_path)
        if issue is not None:
            issues.append(issue)
    return issues, total


def print_report(
    issues: List[OpIssue],
    total: int,
    repo_root: str,
    enforced_paths: Optional[Set[str]] = None,
) -> None:
    """打印检测报告。enforced_paths非None时按门禁/提示分级标注（None表示--strict全量门禁）。"""
    no_examples = [i for i in issues if i.missing_examples]
    arch35_only = [
        i for i in issues if i.arch35_only_examples and not i.missing_examples
    ]
    no_ut = [i for i in issues if i.missing_ut]

    print(f"算子总数: {total}, 正常: {total - len(issues)}, 异常: {len(issues)}")
    print(f"  examples 缺失/为空: {len(no_examples)}")
    print(f"  examples 仅含arch35目录: {len(arch35_only)}")
    print(f"  ut 缺失/为空: {len(no_ut)}")
    if enforced_paths is not None:
        gate = [i for i in issues if i.op_path in enforced_paths]
        hint = [i for i in issues if i.op_path not in enforced_paths]
        print(
            f"  门禁失败(直接涉及): {len(gate)}, 依赖展开提示(不计入退出码): {len(hint)}"
        )
    print("")

    def mark(issue: OpIssue) -> str:
        if enforced_paths is None:
            return ""
        return "  [门禁]" if issue.op_path in enforced_paths else "  [提示]"

    for title, items in (
        ("缺 examples:", no_examples),
        ("examples 仅含arch35目录:", arch35_only),
        ("缺 ut:", no_ut),
    ):
        print(title)
        for issue in items:
            print(f"{mark(issue)} {os.path.relpath(issue.op_path, repo_root)}")
        print("")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="检测算子 examples 与 ut 是否缺失或为空。"
    )
    parser.add_argument(
        "repo_root",
        nargs="?",
        default=None,
        help="仓库根目录，默认为脚本所在仓库根目录；若传入的是文件则视为pr_filelist",
    )
    parser.add_argument(
        "-p",
        "--pr-filelist",
        default=None,
        dest="pr_filelist",
        help="PR变更文件列表（每行一个路径），指定后仅检查涉及的算子（含依赖展开）",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="严格模式：依赖展开算子与全仓扫描的存量缺失同样计入退出码（默认仅PR直接涉及算子门禁）",
    )
    return parser.parse_args()


def find_def_files(op_path: str) -> List[str]:
    """查找算子op_host下的def文件，优先<算子名>_def*.cpp，无则退化为任意*_def*.cpp。"""
    op_host_dir = os.path.join(op_path, "op_host")
    if not os.path.isdir(op_host_dir):
        return []
    def_files = sorted(
        glob.glob(os.path.join(op_host_dir, f"{os.path.basename(op_path)}_def*.cpp"))
    )
    if not def_files:
        def_files = sorted(glob.glob(os.path.join(op_host_dir, "*_def*.cpp")))
    return def_files


# 910系列芯片标识，def文件中声明支持这些芯片的算子，仅有arch35 examples时才视为缺失
SOC_910_PATTERNS = ('AddConfig("ascend910b"', 'AddConfig("ascend910_93"')


@lru_cache(maxsize=None)
def def_supports_910(op_path: str) -> bool:
    """判断算子def文件是否声明支持ascend910b或ascend910_93芯片。"""
    for def_file in find_def_files(op_path):
        try:
            with open(def_file, encoding="utf-8", errors="ignore") as file:
                content = file.read()
        except OSError:
            continue
        if any(pattern in content for pattern in SOC_910_PATTERNS):
            return True
    return False


@lru_cache(maxsize=None)
def get_op_type(op_path: str) -> str:
    """从算子def文件解析OP_ADD声明的OpType（大写驼峰），与func.cmake中get_op_type_from_def_files规则一致，解析失败返回空串。"""
    for def_file in find_def_files(op_path):
        try:
            with open(def_file, encoding="utf-8", errors="ignore") as file:
                content = file.read()
        except OSError:
            continue
        match = re.search(r"OP_ADD\(\s*([A-Za-z0-9_]+)", content)
        if match:
            return match.group(1)
    return ""


def write_csv(issues: List[OpIssue], csv_path: str, repo_root: str) -> int:
    """将检测到的问题按类别汇总写入csv文件，每行一条问题记录，返回行数。

    列格式: 算子(相对仓库根目录路径), 算子名, OPTYPE(def文件中OP_ADD声明的值), 类别。
    """
    rows = []
    for category in (ISSUE_EXAMPLES_MISSING, ISSUE_ARCH35_ONLY, ISSUE_UT_MISSING):
        for issue in issues:
            if category in issue_categories(issue):
                rows.append(
                    (
                        os.path.relpath(issue.op_path, repo_root),
                        os.path.basename(issue.op_path),
                        get_op_type(issue.op_path),
                        category,
                    )
                )
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["算子", "算子名", "OPTYPE", "类别"])
        writer.writerows(rows)
    return len(rows)


# CI输出白名单：算子名仅允许字母数字下划线，防止PR构造的路径将注入内容带进流水线输出变量
SAFE_OP_NAME_RE = re.compile(r"^[A-Za-z0-9_]+$")


def build_failed_output(issues: List[OpIssue], enforced_paths: Set[str]) -> str:
    """构造check_example_failed输出串：<算子名>(类别)，类别为examples/ut，多类别用+连接。

    算子名经白名单校验，不合法的剔除并告警，保证该串可安全进入CI输出变量。
    """
    ops_issues: dict = {}
    for issue in issues:
        if issue.op_path not in enforced_paths:
            continue
        name = os.path.basename(issue.op_path)
        cats = ops_issues.setdefault(name, set())
        if issue.missing_ut:
            cats.add("ut")
        if issue.missing_examples or issue.arch35_only_examples:
            cats.add("examples")
    for name in [name for name in ops_issues if not SAFE_OP_NAME_RE.match(name)]:
        print(
            f"警告: 算子名 {name!r} 含非法字符，已从check_example_failed输出中剔除",
            file=sys.stderr,
        )
        del ops_issues[name]
    return ",".join(
        f"{name}({'+'.join(sorted(cats))})" for name, cats in sorted(ops_issues.items())
    )


def main() -> int:
    """主函数。"""
    args = parse_args()
    repo_root, pr_filelist = args.repo_root, args.pr_filelist
    if repo_root and os.path.isfile(repo_root) and pr_filelist is None:
        repo_root, pr_filelist = None, repo_root
    if repo_root is None:
        repo_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    if not os.path.isdir(repo_root):
        print(f"仓库根目录不存在: {repo_root}", file=sys.stderr)
        return 2

    op_dirs = None
    direct_dirs = None
    if pr_filelist:
        if not os.path.isfile(pr_filelist):
            print(f"pr_filelist不存在: {pr_filelist}", file=sys.stderr)
            return 2
        name_prefixes, involved_dirs, involved_names = parse_filelist_ops(
            repo_root, pr_filelist
        )
        if not involved_dirs and not involved_names:
            print(f"pr_filelist未解析到算子: {pr_filelist}")
            return 0
        op_dirs, direct_dirs = expand_ops_with_dependencies(
            repo_root, involved_dirs, involved_names, name_prefixes
        )
        print(
            f"pr_filelist涉及算子(含依赖展开): {len(op_dirs)}个, "
            f"其中直接涉及: {len(direct_dirs)}个"
        )

    issues, total = check_all(repo_root, op_dirs)
    # 门禁范围：--strict时全量；PR模式默认仅直接涉及算子（依赖展开与存量缺失只提示）；
    # 全仓扫描模式默认report-only，不作为门禁
    if args.strict:
        enforced_paths = {issue.op_path for issue in issues}
    elif direct_dirs is not None:
        direct_set = set(direct_dirs)
        enforced_paths = {
            issue.op_path for issue in issues if issue.op_path in direct_set
        }
    else:
        enforced_paths = set()
    print_report(
        issues, total, repo_root, enforced_paths if pr_filelist or args.strict else None
    )
    build_dir = os.path.join(repo_root, "build")
    os.makedirs(build_dir, exist_ok=True)
    csv_path = os.path.join(build_dir, "check_examples_ut.csv")
    count = write_csv(issues, csv_path, repo_root)
    print(f"检测结果已写入: {csv_path} ({count}条)")
    if enforced_paths:
        # 供CI流水线消费的机器可读输出，与.gitcode/scripts/compile.sh的Check_Example约定一致；
        # 携带问题类别（examples/ut，arch35-only归入examples），避免UT缺失被误报为缺examples；
        # 内容经白名单过滤，可安全进入CI输出变量（流水线侧以环境变量引用，不做shell重解析）
        failed = build_failed_output(issues, enforced_paths)
        if failed:
            print(f"check_example_failed={failed}")
    return 1 if enforced_paths else 0


if __name__ == "__main__":
    sys.exit(main())
