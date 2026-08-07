# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ============================================================================
# Usage: bash run_kernel_ut.sh <exe_path> [SYMBOLIZE] [TIMEOUT] [ASAN_PRELOAD]
#   exe_path      : path to the kernel ut executable
#   SYMBOLIZE     : TRUE/FALSE, whether to enable addr2line symbolization (default TRUE)
#   TIMEOUT       : per-case timeout in seconds (default 120)
#   ASAN_PRELOAD  : LD_PRELOAD string for ASAN mode (optional)
set -o pipefail

EXE_PATH="$1"
ENABLE_SYMBOLIZE="${2:-TRUE}"
CASE_TIMEOUT="${3:-120}"
ASAN_PRELOAD="$4"

DOTTED_LINE="----------------------------------------------------------------"
PASS_COUNT=0
FAIL_COUNT=0
TIMEOUT_COUNT=0
FAILED_CASES=()
TIMEOUT_CASES=()

if [[ -z "$EXE_PATH" || ! -x "$EXE_PATH" ]]; then
    echo "[ERROR] Invalid executable: ${EXE_PATH}"
    exit 1
fi

# Disable core dump to avoid filling disk
ulimit -c 0 2>/dev/null || true

# Reduce CANN log verbosity to minimize output
export ASCEND_GLOBAL_LOG_LEVEL="${ASCEND_GLOBAL_LOG_LEVEL:-3}"

# When symbolization is disabled, create a fake addr2line that echoes the address.
# libcpudebug.so calls popen("addr2line -e <bin> -f -p -a -i -C 0x<addr>"),
# so the fake script must accept those flags and print the raw address.
FAKE_DIR=""
if [[ "$ENABLE_SYMBOLIZE" == "FALSE" ]]; then
    FAKE_DIR=$(mktemp -d)
    cat > "${FAKE_DIR}/addr2line" <<'EOF'
#!/bin/bash
# Fake addr2line: skip symbolization, echo last argument (the address)
for arg in "$@"; do
    last="$arg"
done
echo "??:0"
echo "${last}"
EOF
    chmod +x "${FAKE_DIR}/addr2line"
    export PATH="${FAKE_DIR}:${PATH}"
    echo "[INFO] Symbolization disabled (fake addr2line at ${FAKE_DIR})"
else
    echo "[INFO] Symbolization enabled"
fi

cleanup() {
    if [[ -n "$FAKE_DIR" && -d "$FAKE_DIR" ]]; then
        rm -rf "$FAKE_DIR"
    fi
}
trap cleanup EXIT

# List all test cases
RAW_LIST=$("$EXE_PATH" --gtest_list_tests 2>/dev/null)
if [[ $? -ne 0 ]]; then
    echo "[ERROR] Failed to list test cases from ${EXE_PATH}"
    exit 1
fi

# Parse gtest list into fully-qualified "Suite.Case" names
CASES=()
CURRENT_SUITE=""
while IFS= read -r line; do
    # Lines with no leading space are suite names ending with "."
    if [[ "$line" =~ ^[^[:space:]] ]]; then
        CURRENT_SUITE="${line}"
    elif [[ "$line" =~ ^[[:space:]]+(.+) ]]; then
        local_case="${BASH_REMATCH[1]}"
        CASES+=("${CURRENT_SUITE}${local_case}")
    fi
done <<< "$RAW_LIST"

TOTAL=${#CASES[@]}
if [[ $TOTAL -eq 0 ]]; then
    echo "[WARN] No test cases found"
    exit 0
fi

echo "$DOTTED_LINE"
echo "Running ${TOTAL} test cases (timeout=${CASE_TIMEOUT}s, symbolize=${ENABLE_SYMBOLIZE})"
echo "$DOTTED_LINE"

run_with_preload() {
    local cmd="$1"
    if [[ -n "$ASAN_PRELOAD" ]]; then
        LD_PRELOAD="${ASAN_PRELOAD}" ASAN_OPTIONS=detect_leaks=0 timeout --foreground "${CASE_TIMEOUT}" bash -c "$cmd"
    else
        timeout --foreground "${CASE_TIMEOUT}" bash -c "$cmd"
    fi
}

for case_name in "${CASES[@]}"; do
    run_cmd="export LD_LIBRARY_PATH=\"\$LD_LIBRARY_PATH\" && \"${EXE_PATH}\" --gtest_filter=\"${case_name}\""
    echo "[RUN]   ${case_name}"
    run_with_preload "$run_cmd" 2>&1
    rc=$?

    if [[ $rc -eq 124 ]]; then
        TIMEOUT_COUNT=$((TIMEOUT_COUNT + 1))
        TIMEOUT_CASES+=("$case_name")
        echo "[ERROR] Test case TIMEOUT (${CASE_TIMEOUT}s): ${case_name}"
    elif [[ $rc -eq 0 ]]; then
        PASS_COUNT=$((PASS_COUNT + 1))
        echo "[OK]    ${case_name}"
    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAILED_CASES+=("$case_name")
        echo "[FAIL]  ${case_name} (exit=${rc})"
    fi
done

echo "$DOTTED_LINE"
echo "Summary: total=${TOTAL}, pass=${PASS_COUNT}, fail=${FAIL_COUNT}, timeout=${TIMEOUT_COUNT}"
if [[ ${#FAILED_CASES[@]} -gt 0 ]]; then
    echo "Failed cases:"
    printf '  %s\n' "${FAILED_CASES[@]}"
fi
if [[ ${#TIMEOUT_CASES[@]} -gt 0 ]]; then
    echo "Timeout cases:"
    printf '  %s\n' "${TIMEOUT_CASES[@]}"
fi
echo "$DOTTED_LINE"

if [[ $FAIL_COUNT -gt 0 || $TIMEOUT_COUNT -gt 0 ]]; then
    exit 1
fi
exit 0
