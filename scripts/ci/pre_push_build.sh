#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ============================================================================
# pre-push 本地编译门禁脚本
# 用途：在 git push 前触发本地构建与 UT，将门禁质量左移到本地，保障远程门禁成功率。
#
# 运行模式：
#   默认（快速增量模式）：按变更模块精简编译范围，耗时短但有漏检风险。
#   PRE_PUSH_FULL=1（完整对齐模式）：直接调用 scripts/ci/local_build.sh，100% 复刻 CI 门禁语义。
#
# 旁路方式：
#   git push --no-verify
#   SKIP=pre-push-build git push
#   PRE_PUSH_BUILD=0 git push
# ============================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$WORKSPACE"

# ---------------------------------------------------------------------------
# 0. 旁路控制
# ---------------------------------------------------------------------------
if [[ "${PRE_PUSH_BUILD:-1}" == "0" ]]; then
    echo "[pre-push] PRE_PUSH_BUILD=0, skipping pre-push build gate."
    exit 0
fi

if [[ -n "${SKIP:-}" && "$SKIP" == *"pre-push-build"* ]]; then
    echo "[pre-push] SKIP=pre-push-build, skipping pre-push build gate."
    exit 0
fi

# ---------------------------------------------------------------------------
# 1. 环境自检
# ---------------------------------------------------------------------------
THREAD_NUM=${THREAD_NUM:-$(grep -c ^processor /proc/cpuinfo)}
BASE_BRANCH_NAME=${BASE_BRANCH_NAME:-master}
PR_FILELIST="pr_filelist.txt"

# 检查 CANN Toolkit 环境是否安装
cann_installed=false
for _cann_dir in "${ASCEND_HOME_PATH:-}" "${ASCEND_TOOLKIT_HOME:-}" "/usr/local/Ascend/ascend-toolkit/latest"; do
    [[ -z "$_cann_dir" ]] && continue
    if [[ -f "$_cann_dir/set_env.sh" ]]; then
        cann_installed=true
        break
    fi
done
if [[ "$cann_installed" == "false" ]]; then
    echo -e "\033[31m[pre-push][WARN] 环境未检测到 CANN 包，跳过 pre-push 检查，带风险提交。\033[0m"
    exit 0
fi

# 检查关键脚本是否存在
if [[ ! -f "$WORKSPACE/scripts/ci/local_build.sh" ]]; then
    echo "[pre-push][ERROR] scripts/ci/local_build.sh not found."
    exit 1
fi

# 日志辅助：同时输出到终端和 pre_push_build.log
log_echo() {
    echo "$@" | tee -a pre_push_build.log
}

# ---------------------------------------------------------------------------
# 2. 计算变更文件清单（复用 local_build.sh 的 git diff 逻辑）
# ---------------------------------------------------------------------------
calc_changed_files() {
    local target_url="https://gitcode.com/cann/ops-nn.git"
    local remote_name=""
    for remote in $(git remote); do
        local url
        url=$(git remote get-url "$remote" 2>/dev/null || true)
        if [[ "$url" == *"$target_url"* || "$url" == *"${target_url%.git}"* ]]; then
            remote_name="$remote"
            break
        fi
    done
    if [[ -z "$remote_name" ]]; then
        if git remote | grep -q "^origin$"; then
            remote_name="origin"
        else
            log_echo "[pre-push][ERROR] No remote found for git diff calculation."
            exit 1
        fi
    fi

    log_echo "[pre-push] Remote: $remote_name ($(git remote get-url $remote_name))"
    log_echo "[pre-push] Fetching latest from $remote_name..."
    git fetch "$remote_name" --quiet --prune 2>/dev/null || true

    local remote_ref="${remote_name}/${BASE_BRANCH_NAME}"
    if ! git rev-parse --verify "$remote_ref" >/dev/null 2>&1; then
        log_echo "[pre-push][ERROR] Remote branch '$remote_ref' does not exist."
        git branch -r --list "${remote_name}/*"
        exit 1
    fi

    local merge_base target_commit
    merge_base=$(git merge-base "$remote_ref" HEAD 2>/dev/null || true)
    if [[ -z "$merge_base" ]]; then
        log_echo "[pre-push][ERROR] Could not find merge-base with $remote_ref."
        exit 1
    fi
    target_commit=$(git rev-parse HEAD)

    local changed_files
    changed_files=$(git diff --name-only "$merge_base" "$target_commit" | sort -u)
    if [[ -z "$changed_files" ]]; then
        log_echo "[pre-push][WARN] No changed files found. Please check whether the code has been committed."
        exit 0
    fi

    echo "$changed_files" > "$PR_FILELIST"
    local total
    total=$(wc -l < "$PR_FILELIST")
    log_echo "[pre-push] Changed files: $total"
    log_echo "[pre-push] --- Changed File List ---"
    while IFS= read -r f; do
        [[ -z "$f" ]] && continue
        log_echo "  $f"
    done < "$PR_FILELIST"
    log_echo "[pre-push] ---------------------------"
}

# ---------------------------------------------------------------------------
# 3. 四步识别（用于快速模式与日志展示）
# ---------------------------------------------------------------------------
identify_changes() {
    has_kernel=false
    has_host=false
    has_api=false
    has_graph=false
    has_experimental=false
    has_arch35=false
    need_jit_build=false
    md_only=true
    kernel_ops=""
    arch_list=()

    while IFS= read -r f; do
        [[ -z "$f" ]] && continue
        case "$f" in
            */op_kernel/*)       has_kernel=true; md_only=false ;;
            */op_host/*)         has_host=true;   md_only=false ;;
            */op_api/*)          has_api=true;    md_only=false ;;
            */op_graph/*)        has_graph=true;  md_only=false ;;
            */examples/*)        md_only=false ;;
            experimental/*)      has_experimental=true; md_only=false ;;
            *.md|*.json|*.ini|*.txt|*.yaml|*.yml|cmake/*|scripts/*) ;;
            *)                   md_only=false ;;
        esac
        # 目录包含 arch35 或 op_kernel 不触发 JIT，配置/脚本也不触发
        case "$f" in
            */arch35/*) ;;
            */op_kernel/*) ;;
            *.md|*.json|*.ini|*.txt|*.yaml|*.yml|cmake/*|scripts/*) ;;
            *) need_jit_build=true ;;
        esac
        # arch 识别
        case "$f" in
            */arch35/*) arch_list+=("ascend950"); has_arch35=true ;;
            */arch22/*) arch_list+=("ascend910b") ;;
            */arch20/*) arch_list+=("ascend310p") ;;
        esac
    done < "$PR_FILELIST"

    # 去重 arch_list
    if [[ ${#arch_list[@]} -gt 0 ]]; then
        mapfile -t arch_list < <(printf '%s\n' "${arch_list[@]}" | sort -u)
    fi

    log_echo "[pre-push] --- Change Identification ---"
    log_echo "[pre-push]   op_kernel   : $has_kernel"
    log_echo "[pre-push]   op_host     : $has_host"
    log_echo "[pre-push]   op_api      : $has_api"
    log_echo "[pre-push]   op_graph    : $has_graph"
    log_echo "[pre-push]   experimental: $has_experimental"
    log_echo "[pre-push]   md_only     : $md_only"
    log_echo "[pre-push]   need_jit_build: $need_jit_build"
    if [[ ${#arch_list[@]} -gt 0 ]]; then
        log_echo "[pre-push]   arch        : ${arch_list[*]}"
    else
        log_echo "[pre-push]   arch        : default (affects all SOC)"
    fi
    log_echo "[pre-push] -----------------------------"
}

# ---------------------------------------------------------------------------
# 3b. 打印本次将执行的环节与命令概览
# ---------------------------------------------------------------------------
print_pipeline() {
    local mode
    if [[ "${PRE_PUSH_FULL:-0}" == "1" ]]; then
        mode="FULL"
    else
        mode="FAST"
    fi

    log_echo ""
    log_echo "===================================================================="
    log_echo "  Pipeline Preview (mode: $mode)"
    log_echo "===================================================================="

    if [[ "$mode" == "FULL" ]]; then
        log_echo "  [1] JIT build           : bash build.sh --pkg --jit -j${THREAD_NUM}"
        log_echo "  [2] ophost UT           : bash build.sh -u --ophost -f $PR_FILELIST -j${THREAD_NUM}"
        log_echo "  [3] opapi UT            : bash build.sh -u --opapi -f $PR_FILELIST -j${THREAD_NUM}"
        if [[ "$BASE_BRANCH_NAME" == "master" ]]; then
            log_echo "  [4] opgraph UT          : bash build.sh -u --opgraph -f $PR_FILELIST -j${THREAD_NUM}"
            log_echo "  [5] opkernel UT         : bash scripts/ci/check_kernel_ut.sh $PR_FILELIST --no_cov"
        fi
        log_echo "  [6] single-op pkg       : bash scripts/ci/check_pkg.sh $PR_FILELIST -j${THREAD_NUM} --no_force"
        log_echo "  [7] ascend950 pkg       : bash scripts/ci/compile_ascend950_pkg.sh $PR_FILELIST --no_force -j${THREAD_NUM}"
        log_echo "  [8] A2 smoke            : bash scripts/ci/check_example.sh $PR_FILELIST (needs 910B)"
        log_echo "  [9] experimental pkg    : bash scripts/ci/check_experimental_pkg.sh $PR_FILELIST"
    else
        local idx=1
        if [[ "$need_jit_build" == "true" ]]; then
            log_echo "  [$idx] JIT build        : bash build.sh --pkg --jit -j${THREAD_NUM}"
            idx=$((idx+1))
        else
            log_echo "  [--] JIT build          : SKIP (arch35/op_kernel only)"
        fi
        if [[ "$has_host" == "true" ]]; then
            log_echo "  [$idx] ophost UT        : bash build.sh -u --ophost -f $PR_FILELIST -j${THREAD_NUM}"
            idx=$((idx+1))
        else
            log_echo "  [--] ophost UT          : SKIP (op_host not changed)"
        fi
        if [[ "$has_api" == "true" ]]; then
            log_echo "  [$idx] opapi UT         : bash build.sh -u --opapi -f $PR_FILELIST -j${THREAD_NUM}"
            idx=$((idx+1))
        else
            log_echo "  [--] opapi UT           : SKIP (op_api not changed)"
        fi
        if [[ "$BASE_BRANCH_NAME" == "master" ]]; then
            if [[ "$has_graph" == "true" ]]; then
                log_echo "  [$idx] opgraph UT       : bash build.sh -u --opgraph -f $PR_FILELIST -j${THREAD_NUM}"
                idx=$((idx+1))
            else
                log_echo "  [--] opgraph UT         : SKIP (op_graph not changed)"
            fi
            if [[ "$has_kernel" == "true" ]]; then
                log_echo "  [$idx] opkernel UT      : bash scripts/ci/check_kernel_ut.sh $PR_FILELIST --no_cov"
                idx=$((idx+1))
            else
                log_echo "  [--] opkernel UT        : SKIP (op_kernel not changed)"
            fi
        else
            log_echo "  [--] opgraph/opkernel UT: SKIP (non-master branch)"
        fi
        if [[ "$has_arch35" == "true" ]]; then
            local arch_flag="--no_force"
            [[ "$has_kernel" == "false" ]] && arch_flag="-force_jit"
            log_echo "  [$idx] ascend950 pkg    : bash scripts/ci/compile_ascend950_pkg.sh $PR_FILELIST ${arch_flag} -j${THREAD_NUM}"
            idx=$((idx+1))
        else
            log_echo "  [--] ascend950 pkg      : SKIP (arch35 not changed)"
        fi
        log_echo "  [--] single-op pkg      : SKIP (fast mode)"
        log_echo "  [--] A2 smoke           : SKIP (fast mode)"
        if [[ "$has_experimental" == "true" ]]; then
            log_echo "  [$idx] experimental pkg : bash scripts/ci/check_experimental_pkg.sh $PR_FILELIST"
            idx=$((idx+1))
        else
            log_echo "  [--] experimental pkg   : SKIP (experimental not changed)"
        fi
    fi
    log_echo "===================================================================="
    log_echo ""
}

# ---------------------------------------------------------------------------
# 4. 静默执行：构建/UT 输出仅写入日志文件，终端只打印阶段名与 PASS/FAIL
#    用法: run_cmd_silent <stage_name> <log_file> <extra_grep> <cmd...>
#    extra_grep 非空时，即使退出码为 0 也会在日志中搜索该关键字，命中即判失败
# ---------------------------------------------------------------------------
run_cmd_silent() {
    local stage_name=$1 log_file=$2 extra_grep=$3
    shift 3
    log_echo "[pre-push] >>> $stage_name  (log: $log_file)"
    [[ "$log_file" != "pre_push_build.log" ]] && : > "$log_file"
    local rc=0
    set +e
    "$@" >> "$log_file" 2>&1
    rc=$?
    set -e
    [[ "$log_file" != "pre_push_build.log" ]] && cat "$log_file" >> pre_push_build.log
    if [[ $rc -ne 0 ]]; then
        log_echo "[pre-push][FAILED] $stage_name (exit $rc), check log: $log_file"
        local failed_lines
        failed_lines=$(grep -E '\[  FAILED  \]|FAILED TESTS|error happened|Error:|error:' "$log_file" 2>/dev/null | tail -5)
        [[ -n "$failed_lines" ]] && log_echo "$failed_lines"
        exit $rc
    fi
    if [[ -n "$extra_grep" ]] && grep -q "$extra_grep" "$log_file" 2>/dev/null; then
        log_echo "[pre-push][FAILED] $stage_name: found '$extra_grep' in log: $log_file"
        exit 1
    fi
    log_echo "[pre-push][PASSED] $stage_name"
}

# ---------------------------------------------------------------------------
# 5. 阶段间清理
# ---------------------------------------------------------------------------
clean_build() {
    local dir
    for dir in build build_out; do
        local abs="$WORKSPACE/$dir"
        if [[ -z "$WORKSPACE" || "$abs" == "/" || ! "$abs" =~ ^/[^/] ]]; then
            log_echo "[pre-push][ERROR] clean_build refused: unsafe path '$abs' (WORKSPACE='$WORKSPACE')."
            exit 1
        fi
        if [[ -d "$abs" ]]; then
            rm -rf "$abs"
        fi
    done
}

# ---------------------------------------------------------------------------
# 6a. 完整对齐模式：直接调用 local_build.sh
# ---------------------------------------------------------------------------
run_full_mode() {
    log_echo "[pre-push] >>> FULL MODE: delegating to local_build.sh (100% CI-aligned)"
    log_echo "[pre-push] Branch: $(git rev-parse --abbrev-ref HEAD)  Base: $BASE_BRANCH_NAME"
    run_cmd_silent "FULL MODE (local_build.sh)" pre_push_build.log "" \
        bash "$WORKSPACE/scripts/ci/local_build.sh"
    log_echo "[pre-push] <<< FULL MODE: ALL PASSED"
}

# ---------------------------------------------------------------------------
# 6b. 快速增量模式：按模块精简
# ---------------------------------------------------------------------------
run_fast_mode() {
    log_echo "[pre-push] >>> FAST MODE: incremental build (default)"
    log_echo "[pre-push] NOTE: Fast mode has known coverage gaps vs CI. Run full mode before merge."
    log_echo ""

    local branch
    branch=$(git rev-parse --abbrev-ref HEAD)
    export WORKSPACE="$WORKSPACE"
    export ASCEND_3RD_LIB_PATH="$WORKSPACE/third_party"
    export BASE_PATH="$WORKSPACE"
    export BUILD_PATH="$WORKSPACE/build"

    clean_build

    if [[ "$need_jit_build" == "true" ]]; then
        run_cmd_silent "JIT build" pre_push_build.log "" \
            bash build.sh --pkg --jit -j${THREAD_NUM} --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH}
        clean_build
    fi

    if [[ "$has_host" == "true" ]]; then
        run_cmd_silent "ophost UT" ut_test.log "" \
            bash build.sh -u --ophost -f "$PR_FILELIST" --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH} -j${THREAD_NUM}
    else
        log_echo "[pre-push] op_host not changed, skipping ophost UT (fast mode)."
    fi

    if [[ "$has_api" == "true" ]]; then
        if [[ "$has_host" == "false" && "$need_jit_build" == "false" ]]; then
            run_cmd_silent "JIT build (opapi dep)" pre_push_build.log "" \
                bash build.sh --pkg --jit -j${THREAD_NUM} --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH}
        fi
        run_cmd_silent "opapi UT" ut_test.log "" \
            bash build.sh -u --opapi -f "$PR_FILELIST" --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH} -j${THREAD_NUM}
    else
        log_echo "[pre-push] op_api not changed, skipping opapi UT (fast mode)."
    fi

    if [[ "$branch" == "master" || "$BASE_BRANCH_NAME" == "master" ]]; then
        if [[ "$has_graph" == "true" ]]; then
            run_cmd_silent "opgraph UT" ut_test.log "" \
                bash build.sh -u --opgraph -f "$PR_FILELIST" --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH} -j${THREAD_NUM}
        fi
        if [[ "$has_kernel" == "true" ]]; then
            run_cmd_silent "opkernel UT" ut_kernel.log "error happened" \
                bash scripts/ci/check_kernel_ut.sh "$PR_FILELIST" --no_cov
        fi
    else
        log_echo "[pre-push] Non-master branch, skipping opgraph/opkernel UT (aligned with CI)."
    fi
    clean_build

    if [[ "$has_arch35" == "true" ]]; then
        if [[ "$has_kernel" == "false" ]]; then
            run_cmd_silent "ascend950 pkg" pre_push_build.log "" \
                bash scripts/ci/compile_ascend950_pkg.sh "$PR_FILELIST" -force_jit -j${THREAD_NUM}
        else
            run_cmd_silent "ascend950 pkg" pre_push_build.log "" \
                bash scripts/ci/compile_ascend950_pkg.sh "$PR_FILELIST" -j${THREAD_NUM} --no_force
        fi
        clean_build
    fi

    log_echo "[pre-push][SKIP] Single-op pkg / A2 smoke skipped in fast mode."
    log_echo "[pre-push][SKIP] Run full mode (PRE_PUSH_FULL=1) to cover remaining CI stages."

    if [[ "$has_experimental" == "true" ]]; then
        run_cmd_silent "experimental pkg" pre_push_build.log "" \
            bash scripts/ci/check_experimental_pkg.sh "$PR_FILELIST"
        clean_build
        log_echo "[pre-push][SKIP] experimental example skipped in fast mode."
    fi

    log_echo "[pre-push] <<< FAST MODE: DONE (with coverage gaps, see SKIP above)"
}

# ---------------------------------------------------------------------------
# 7. 主流程
# ---------------------------------------------------------------------------
main() {
    : > pre_push_build.log
    log_echo "===================================================================="
    log_echo "  Pre-Push Build & UT Gate"
    log_echo "  Mode: $([[ "${PRE_PUSH_FULL:-0}" == "1" ]] && echo "FULL (CI-aligned)" || echo "FAST (incremental)")"
    log_echo "  Workspace: $WORKSPACE"
    log_echo "  Time: $(date '+%Y-%m-%d %H:%M:%S')"
    log_echo "===================================================================="
    calc_changed_files
    identify_changes
    print_pipeline

    # 仅文档/配置/cmake/scripts 变更时，所有流程直接跳过
    if [[ "$md_only" == "true" ]]; then
        log_echo "[pre-push] Only docs/config/cmake/scripts changed, skipping all build & UT."
        log_echo "[pre-push][HINT] If scripts/ci or CMakeLists were modified, CI script changes are NOT verified locally. Review manually."
        exit 0
    fi

    # 仅 experimental 变更时，只触发 check_experimental_pkg.sh
    if [[ "$has_experimental" == "true" && "$has_host" == "false" && "$has_kernel" == "false" \
          && "$has_api" == "false" && "$has_graph" == "false"  ]]; then
        log_echo "[pre-push] Only experimental changed, running check_experimental_pkg.sh only."
        clean_build
        export ASCEND_3RD_LIB_PATH="$WORKSPACE/third_party"
        run_cmd_silent "experimental pkg" pre_push_build.log "" \
            bash scripts/ci/check_experimental_pkg.sh "$PR_FILELIST"
        exit 0
    fi

    local start_ts end_ts elapsed
    start_ts=$(date +%s)

    if [[ "${PRE_PUSH_FULL:-0}" == "1" ]]; then
        run_full_mode
    else
        run_fast_mode
    fi

    end_ts=$(date +%s)
    elapsed=$((end_ts - start_ts))
    log_echo ""
    log_echo "===================================================================="
    log_echo "  Pre-Push Gate: PASSED  (elapsed ${elapsed}s)"
    log_echo "===================================================================="
    exit 0
}

main "$@"
