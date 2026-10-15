#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""算子变更解析公共库：filelist解析、算子目录遍历、依赖展开，供各CI检查脚本共用。

路径约定（与parse_compile_changed_files.py的判定一致）：
- 主类目: <category>/<op>/...，category须在算子类目集合中，至少3段路径
- experimental: experimental/<category>/<op>/...，category须在算子类目集合中，至少4段路径
- 类目公共目录约定记为'<category>.common'（与build/tmp/ops_config.txt的键一致）
- pr_filelist仅接受git diff --name-only产出的纯路径列表（每行一个路径），不支持带状态列的格式
"""

import glob
import os
import re
import sys
from typing import Iterator, List, Optional, Set, Tuple

# 公共库目录名（不算子）
COMMON_LIB_NAMES = {"common", "sort_lib", "index_check", "foreach_utils", "pool_utils"}

# 非算子类目的顶级目录
NON_CATEGORY_DIRS = {
    ".git",
    "build",
    "build_out",
    "cmake",
    "common",
    "docs",
    "examples",
    "oat_reports",
    "scripts",
    "st",
    "tests",
    "third_party",
    "torch_extension",
}


def get_op_categories(repo_root: str) -> List[str]:
    """从cmake/variables.cmake的OP_CATEGORY_LIST读取算子类目列表。"""
    variables_cmake = os.path.join(repo_root, "cmake", "variables.cmake")
    with open(variables_cmake, encoding="utf-8") as file:
        for line in file:
            match = re.search(r"set\(OP_CATEGORY_LIST\s+(.*)\)", line)
            if match:
                return match.group(1).replace('"', "").split()
    return []


def resolve_changed_op(
    parts: List[str], categories: Set[str], include_experimental: bool = True
) -> Optional[Tuple[str, str, bool]]:
    """按路径段解析变更路径所属算子。

    返回(prefix, second, is_common)：prefix为'<category>'或'experimental/<category>'，
    second为算子目录名（类目公共目录时second为'common'且is_common=True）；
    不属于任何类目返回None。仅做路径段判定，不做目录存在性校验。
    include_experimental=False时与parse_compile_changed_files.py的is_experimental=FALSE语义一致，
    experimental路径返回None。
    """
    if len(parts) < 2:
        return None
    category, second = parts[0], parts[1]
    if category in categories and len(parts) >= 3:
        return (category, second, second == "common")
    if (
        include_experimental
        and category == "experimental"
        and len(parts) >= 4
        and parts[1] in categories
    ):
        # prefix须用parts[1]拼接——此处category仍为'experimental'，不能直接引用
        return (f"experimental/{parts[1]}", parts[2], parts[2] == "common")
    return None


def is_common_lib(name: str) -> bool:
    """判断目录名是否为公共库目录（不算子）。"""
    return (
        name in COMMON_LIB_NAMES
        or name.endswith("_common")
        or name.endswith("_utils")
        or name.endswith("_util")
    )


def has_def_file(op_path: str) -> bool:
    """判断算子op_host下是否存在def文件，无def文件为接口类算子。"""
    op_host_dir = os.path.join(op_path, "op_host")
    if not os.path.isdir(op_host_dir):
        return False
    return bool(
        glob.glob(os.path.join(op_host_dir, f"{os.path.basename(op_path)}_def*.cpp"))
        or glob.glob(os.path.join(op_host_dir, "*_def*.cpp"))
    )


def iter_category_dirs(repo_root: str) -> Iterator[str]:
    """遍历所有算子类目目录（含experimental下的二级类目）。"""
    for entry in sorted(os.listdir(repo_root)):
        path = os.path.join(repo_root, entry)
        if (
            entry in NON_CATEGORY_DIRS
            or entry.startswith(".")
            or not os.path.isdir(path)
        ):
            continue
        if entry == "experimental":
            for sub in sorted(os.listdir(path)):
                sub_path = os.path.join(path, sub)
                if os.path.isdir(sub_path):
                    yield sub_path
        else:
            yield path


def iter_op_dirs(repo_root: str) -> Iterator[str]:
    """遍历仓库内所有算子目录（不含公共库目录与无def文件的接口类算子）。"""
    for category in iter_category_dirs(repo_root):
        for entry in sorted(os.listdir(category)):
            op_path = os.path.join(category, entry)
            if not os.path.isdir(op_path) or is_common_lib(entry):
                continue
            if not has_def_file(op_path):
                continue
            if os.path.isdir(os.path.join(op_path, "op_host")) or os.path.isdir(
                os.path.join(op_path, "op_kernel")
            ):
                yield op_path


def parse_filelist_ops(
    repo_root: str, pr_filelist: str, categories: Optional[Set[str]] = None
) -> Tuple[dict, Set[str], Set[str]]:
    """解析pr_filelist（纯路径列表），返回(name_prefixes, involved_dirs, involved_names)。

    name_prefixes: 算子名 -> 命名空间前缀集合（'<category>'或'experimental/<category>'）；
    involved_dirs: 实际存在的算子目录绝对路径集合（跳过common与公共库目录名）；
    involved_names: 算子名集合（类目公共目录记为'<category>.common'）。
    """
    if categories is None:
        categories = set(get_op_categories(repo_root))
    name_prefixes: dict = {}
    involved_dirs: Set[str] = set()
    involved_names: Set[str] = set()
    with open(pr_filelist, encoding="utf-8") as file:
        for line in file:
            file_path = line.strip()
            if not file_path or file_path.startswith("#"):
                continue
            if os.path.isabs(file_path):
                file_path = os.path.relpath(file_path, repo_root)
            resolved = resolve_changed_op(file_path.split("/"), categories)
            if resolved is None:
                continue
            prefix, second, is_common = resolved
            if is_common:
                involved_names.add(f"{prefix.split('/')[-1]}.common")
                continue
            if is_common_lib(second):
                continue
            op_dir = os.path.join(repo_root, prefix, second)
            if not os.path.isdir(op_dir):
                continue
            involved_dirs.add(op_dir)
            involved_names.add(second)
            name_prefixes.setdefault(second, set()).add(prefix)
    return name_prefixes, involved_dirs, involved_names


def expand_ops_with_dependencies(
    repo_root: str,
    involved_dirs: Set[str],
    involved_names: Set[str],
    name_prefixes: dict,
) -> Tuple[List[str], List[str]]:
    """基于build/tmp/ops_config.txt依赖图做正向+反向依赖展开。

    返回(全部算子目录, 直接涉及算子目录)。算子名反查目录时按命名空间限定：
    优先取PR涉及算子所在的命名空间，避免主类目与experimental的同名算子互相误伤；
    命名空间未命中时仅当该算子名全仓目录唯一才回退取唯一目录，存在歧义的算子名跳过。
    依赖图不可用（未cmake configure等）时降级为仅直接涉及算子。
    """
    direct = {path for path in involved_dirs if has_def_file(path)}
    involved = set(direct)
    if not involved_names:
        return sorted(involved), sorted(direct)
    from dependency_parser import OpDependenciesParser

    try:
        parser = OpDependenciesParser(os.path.join(repo_root, "build"))
    except (RuntimeError, OSError) as err:
        print(
            f"警告: 依赖解析失败({err})，需先完成cmake configure生成build/tmp/ops_config.txt，"
            "仅检查pr_filelist直接涉及的算子",
            file=sys.stderr,
        )
        return sorted(involved), sorted(direct)
    allowed_prefixes = set().union(*name_prefixes.values()) if name_prefixes else set()
    dirs_by_name: dict = {}
    dirs_index: dict = {}
    for op_dir in iter_op_dirs(repo_root):
        rel = os.path.relpath(op_dir, repo_root)
        dirs_by_name.setdefault(
            (os.path.dirname(rel), os.path.basename(op_dir)), []
        ).append(op_dir)
        dirs_index.setdefault(os.path.basename(op_dir), []).append(op_dir)
    known_ops = set(parser.all_ops)
    known_names = [name for name in involved_names if name in known_ops]
    for name in involved_names - set(known_names):
        print(
            f"警告: {name} 不在ops_config.txt依赖图中（如PR新增算子），仅直接检查",
            file=sys.stderr,
        )
    forward, reverse = ([], [])
    try:
        forward, reverse = (
            parser.get_dependencies_by_ops(known_names) if known_names else ([], [])
        )
    except (RuntimeError, OSError) as err:
        print(
            f"警告: 依赖图遍历失败({err})，仅检查pr_filelist直接涉及的算子",
            file=sys.stderr,
        )
    expanded_names = {
        name for name in set(forward) | set(reverse) if not name.endswith(".common")
    }
    for name in expanded_names:
        resolved = []
        for prefix in sorted(allowed_prefixes):
            resolved += dirs_by_name.get((prefix, name), [])
        if not resolved:
            unique_dirs = dirs_index.get(name, [])
            if len(unique_dirs) == 1:
                resolved = unique_dirs
        involved.update(resolved)
    return sorted(involved), sorted(direct)
