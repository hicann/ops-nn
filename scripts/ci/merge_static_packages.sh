#!/bin/bash
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
set -euo pipefail

SOC="910b"
ARCH_INFO=$(uname -m)

log_info() {
    echo "=== $1 ==="
}

log_error() {
    echo "Error: $1" >&2
}

usage() {
    cat <<EOF
Usage: $(basename "$0") [--soc <soc_name>] <packages_dir> <output_dir>

Decompress multiple cann-ops-nn-static tar.gz packages and merge them.
Static library (.a) files will be merged into one.

Options:
  --soc <soc_name>  Specify the SoC name (default: 910b)

Arguments:
  packages_dir  Directory containing *.tar.gz packages
  output_dir    Output directory for merged result

Example:
  $(basename "$0") ./build_out ./merged_output
  $(basename "$0") --soc 910b ./build_out ./merged_output
EOF
    exit 1
}

remove_ascend_lower() {
    local input="$1"
    local lower_input=$(echo "$input" | tr '[:upper:]' '[:lower:]')
    local result=${lower_input#ascend}
    if [[ "$result" == "910_93" ]]; then
        result="A3"
    fi
    echo "$result"
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --soc)
                [[ $# -lt 2 ]] && { log_error "--soc requires a value"; usage; }
                SOC=$(remove_ascend_lower "$2")
                shift 2
                ;;
            -h|--help)
                usage
                ;;
            -*)
                log_error "unknown option '$1'"
                usage
                ;;
            *)
                break
                ;;
        esac
    done

    [[ $# -ne 2 ]] && usage

    PACKAGES_DIR="$1"
    OUTPUT_DIR="$2"

    [[ ! -d "$PACKAGES_DIR" ]] && { log_error "packages_dir '$PACKAGES_DIR' does not exist"; exit 1; }

    PACKAGES_DIR=$(realpath "$PACKAGES_DIR")
    mkdir -p "$OUTPUT_DIR"
    OUTPUT_DIR=$(realpath "$OUTPUT_DIR")
}

extract_packages() {
    TMPDIR="$PACKAGES_DIR/tmp"
    rm -rf "$TMPDIR"
    mkdir -p "$TMPDIR"

    log_info "Extracting packages"
    PKG_COUNT=0
    for pkg in "$PACKAGES_DIR"/*.tar.gz; do
        [[ ! -f "$pkg" ]] && continue
        echo "  Extracting: $(basename "$pkg")"
        tar -xzf "$pkg" -C "$TMPDIR"
        PKG_COUNT=$((PKG_COUNT + 1))
    done

    [[ $PKG_COUNT -eq 0 ]] && { log_error "no .tar.gz files found in '$PACKAGES_DIR'"; exit 1; }
    echo "  Total packages: $PKG_COUNT"
}

merge_include() {
    local pkg_dir="$1"
    local base_dir="$2"

    [[ -d "$pkg_dir/include" ]] || return 0
    cp -rn "$pkg_dir/include/"* "$base_dir/include/" 2>/dev/null || true
}

merge_static_lib() {
    local pkg_dir="$1"
    local base_dir="$2"
    local other_lib="$pkg_dir/lib64/libcann_nn_static.a"

    [[ -f "$other_lib" ]] || return 0

    pushd "$pkg_dir/lib64" > /dev/null

    local pkg_name
    pkg_name="$(basename "$pkg_dir")"; pkg_name="${pkg_name##*static-}"; pkg_name="${pkg_name%%_linux*}"
    local count=0

    ar x "$other_lib"
    o_files=($(ls *.o))

    for o_file in "${o_files[@]}"; do
        local name=$(basename ${o_file})
        local dst="$base_dir/lib64/$o_file"
        if [[ ! -f "$dst" ]]; then
            cp "$o_file" "$dst"
            continue
        fi

        # 非生成的资源代码，直接复制
        if [[ "$o_file" == *"op_resource.cpp.o" ]]; then
            if [[ $(stat -c%s "$o_file") -gt $(stat -c%s "$dst") ]]; then
                cp "$o_file" "$dst"
            fi
        else
            cp "$o_file" "$dst"
        fi
    done

    popd > /dev/null
}

merge_packages() {
    log_info "Merging packages"

    PKG_DIRS=()
    for dir in "$TMPDIR"/cann-ops-nn-static-*/; do
        [[ -d "$dir" ]] && PKG_DIRS+=("$dir")
    done

    if [[ ${#PKG_DIRS[@]} -eq 0 ]]; then
        log_error "no cann-ops-nn-static-* directories found after extraction"
        exit 1
    fi

    local VERSION
    VERSION=$(grep 'set_cann_package' "$(dirname "$0")/../../version.cmake" | sed 's/.*set_cann_package([^ ]* [^ ]* "\([^"]*\)".*/\1/')
    local base_name="cann-${SOC}-ops-nn-static-${VERSION}_linux-${ARCH_INFO}"
    local base_dir="$TMPDIR/$base_name"

    rm -rf "$base_dir"
    mkdir -p "$base_dir/lib64" "$base_dir/include"

    for pkg_dir in "${PKG_DIRS[@]}"; do
        echo "  Merging: $(basename "$pkg_dir")"
        merge_include "$pkg_dir" "$base_dir"
        merge_static_lib "$pkg_dir" "$base_dir"
    done

    package_result "$base_dir" "$base_name"
}

package_result() {
    local base_dir="$1"
    local base_name="$2"

    pushd "$base_dir/lib64" > /dev/null
    ar rcs libcann_nn_static.a ./*.o
    rm -f *.o
    popd > /dev/null

    pushd "$TMPDIR" > /dev/null
    tar -czf "${base_name}.tar.gz" "$base_name"
    mv "${base_name}.tar.gz" "$OUTPUT_DIR"
    popd > /dev/null

    log_info "Done: merged output in $OUTPUT_DIR"
}

main() {
    parse_args "$@"
    extract_packages
    merge_packages
}

main "$@"
