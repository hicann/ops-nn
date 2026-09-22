#!/bin/bash
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
# weaken_compat_symbols.sh — 自动提取并弱化 .so 中的新接口符号
# 用法: ./weaken_compat_symbols.sh <so_file> [python3_executable]
# 依赖: nm, python3, pyelftools
#
# 注意: 该脚本作为 CMake POST_BUILD 步骤执行, 必须保证在 pyelftools 缺失或
#       无匹配符号时不阻断构建 (exit 0)。
#
# 兼容周期与下线说明：本兼容逻辑（本脚本 + weaken_dynsym.py + pass 内运行时版本守卫）
# 用于支持旧版本运行时（如 8.5.0）加载本库。兼容周期至 CANN 9.2.0（2027-06-30）止，
# 届时最低支持运行时版本已覆盖全部新增接口（SetPassName/GetOptionValue 自 9.0.0、
# Replace 三参重载自 9.2.0），此逻辑随旧版本运行时退服下线；可同步删除本脚本、
# weaken_dynsym.py、symbol.cmake 中对应 POST_BUILD 步骤及 pass 内运行时版本守卫。
SO_FILE="$1"
PYTHON_BIN="${2:-python3}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ -z "$SO_FILE" ] || [ ! -f "$SO_FILE" ]; then
    echo "weaken_compat_symbols: no valid .so file provided, skip." >&2
    exit 0
fi

if ! command -v nm >/dev/null 2>&1; then
    echo "weaken_compat_symbols: 'nm' not found, skip dynsym weakening." >&2
    exit 0
fi

echo "=== Extracting symbols to weaken from $SO_FILE ==="

# 提取所有未定义 (U) 且属于新版本接口的符号。本库以新版本头文件编译，会引用
# 新版本才提供的接口符号（CustomPassContext 新增成员方法 GetOptionValue/SetPassName/
# GetPassName、SubgraphRewriter 携带 CustomPassContext 的 Replace 重载、ES
# CompliantNodeBuilder V2 等），旧版本运行时中不存在这些符号。将其从 GLOBAL 弱化
# 为 WEAK 后，旧版本运行时 dlopen 不再失败（未解析符号返回 NULL），
# 是否真正调用由 pass 内的运行时版本守卫决定。
MANGLED_LIST=$(nm -D "$SO_FILE" 2>/dev/null | grep " U " | grep -E \
    "CustomPassContext.*(GetOptionValue|SetPassName|GetPassName)|InferShapeUtil|GraphFuseInspectorUtils|SubgraphRewriter.*CustomPassContext|PatternFusionPassV2|DecomposePassV2|CompliantNodeBuilder.*V2" \
    | awk '{print $2}')

if [ -z "$MANGLED_LIST" ]; then
    echo "No symbols to weaken (no new API references found)"
    exit 0
fi

echo "Symbols to weaken:"
for sym in $MANGLED_LIST; do
    echo "  $sym"
done
echo ""

"$PYTHON_BIN" "${SCRIPT_DIR}/weaken_dynsym.py" "$SO_FILE" $MANGLED_LIST
# pyelftools 为声明的构建依赖 (见 requirements.txt)，缺失时脚本直接报错退出
# （nm -D 读取的即为 .dynsym，符号未命中仅作为告警，不阻断构建）
