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

set -e

declare -A SOC_TO_ARCH
SOC_TO_ARCH=(["ascend910_93"]="2201" ["ascend910b"]="2201" ["ascend950"]="3510")

COMPILED_OPS=""
THREAD_NUM=8
SOC_VERSION=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --ops=*)
      COMPILED_OPS=${1#*=}
      ;;
    -j*)
      THREAD_NUM=${1#-j}
      ;;
    --soc=*)
      SOC_VERSION=${1#*=}
      ;;
  esac
  shift
done

echo "=== Build PyTorch extension ==="
echo "=== Check environment ==="
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")')
PYTHON_MAJOR=$(python3 -c 'import sys; print(sys.version_info.major)')
PYTHON_MINOR=$(python3 -c 'import sys; print(sys.version_info.minor)')
if [ "$PYTHON_MAJOR" -lt 3 ] || [ "$PYTHON_MAJOR" -eq 3 -a "$PYTHON_MINOR" -lt 8 ]; then
  echo "Warning: Build PyTorch extension failed, Python version must be >= 3.8."
  exit 0
fi
echo "Python version: $PYTHON_MAJOR.$PYTHON_MINOR"

TORCH_VERSION=$(python3 -c 'import torch; print(torch.__version__)')
TORCH_MAJOR=$(echo "$TORCH_VERSION" | cut -d. -f1 | sed 's/[^0-9]//g')
TORCH_MINOR=$(echo "$TORCH_VERSION" | cut -d. -f2 | sed 's/[^0-9]//g')
if [ "$TORCH_MAJOR" -lt 2 ] || [ "$TORCH_MAJOR" -eq 2 -a "$TORCH_MINOR" -lt 6 ]; then
  echo "Warning: Build PyTorch extension failed, PyTorch version must be >= 2.6."
  exit 0
fi
echo "PyTorch version: $TORCH_VERSION"

CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPS_NN_DIR="$(cd "$CURRENT_DIR/../.." && pwd)"
SCRIPT_DIR="$OPS_NN_DIR/build/torch_extension"
EXPERIMENTAL_DIR="$OPS_NN_DIR/experimental"
EXPERIMENTAL_TMP_DIR="$SCRIPT_DIR/experimental_tmp"
ASCEND_OPS_NN_DIR="$SCRIPT_DIR/ascend_ops_nn"
BUILD_OUT_DIR="$OPS_NN_DIR/build_out"

VALID_OPS=""
for category_dir in "$EXPERIMENTAL_DIR"/*/; do
  for op_dir in "$category_dir"*/; do
    if [ -f "$op_dir/CMakeLists.txt" ]; then
      if grep -qE "^\s*add_sources\s*\(" "$op_dir/CMakeLists.txt" 2>/dev/null; then
        op_name=$(basename "$op_dir")
        if [ -n "$VALID_OPS" ]; then
          VALID_OPS="$VALID_OPS;$op_name"
        else
          VALID_OPS="$op_name"
        fi
      fi
    fi
  done
done

if [ -z "$VALID_OPS" ]; then
  echo "Warning: PyTorch extension ops are not exist, exiting."
  exit 0
fi
echo "PyTorch extension ops: $VALID_OPS ==="

echo "THREAD_NUM: $THREAD_NUM"
export PE_BUILD_JOBS=${THREAD_NUM}

if [ -n "$SOC_VERSION" ]; then
  if [ -z "${SOC_TO_ARCH[$SOC_VERSION]+x}" ]; then
    echo "Warning: soc '$SOC_VERSION' is not supported. Only support: ascend910b, ascend910_93, ascend950."
    exit 0
  fi
  echo "NPU_ARCH: $SOC_VERSION"
else
  SOC_VERSION="ascend910b"
fi
PE_COMPILE_ARGS="--npu-arch=dav-${SOC_TO_ARCH[${SOC_VERSION}]}"
export NPU_ARCH=${SOC_VERSION}
export PE_COMPILE_ARGS=${PE_COMPILE_ARGS}

echo "=== Copy scripts files to torch_extension ==="
rm -rf "$SCRIPT_DIR"
mkdir -p "$SCRIPT_DIR"
cp -r "$CURRENT_DIR"/* "$SCRIPT_DIR"

echo "=== Copy experimental files to experimental_tmp ==="
rm -rf "$EXPERIMENTAL_TMP_DIR"
mkdir -p "$EXPERIMENTAL_TMP_DIR"

if [ -n "$COMPILED_OPS" ]; then
  echo "=== Copy specified ops: $COMPILED_OPS ==="
  found=false
  for op in ${COMPILED_OPS//;/ }; do
    if [[ ";$VALID_OPS;" != *";$op;"* ]]; then
      echo "Warning: '$op' is not PyTorch extension op, please check."
      continue
    fi
    for category_dir in "$EXPERIMENTAL_DIR"/*/; do
      category_name=$(basename "$category_dir")
      if [ -d "$category_dir$op" ]; then
        mkdir -p "$EXPERIMENTAL_TMP_DIR/$category_name"
        cp -r "$category_dir$op" "$EXPERIMENTAL_TMP_DIR/$category_name/"
        echo "Copy '$op' success."
        found=true
        break
      fi
    done
  done
  if [[ "$found" = false ]]; then
    echo "Warning: Build PyTorch extension failed."
    exit 0
  fi
else
  echo "=== Copy all ops: $VALID_OPS ==="
  for op in ${VALID_OPS//;/ }; do
    for category_dir in "$EXPERIMENTAL_DIR"/*/; do
      category_name=$(basename "$category_dir")
      if [ -d "$category_dir$op" ]; then
        mkdir -p "$EXPERIMENTAL_TMP_DIR/$category_name"
        cp -r "$category_dir$op" "$EXPERIMENTAL_TMP_DIR/$category_name/"
        echo "Copy '$op' success."
        break
      fi
    done
  done
fi

echo "=== Copy python files to ascend_ops_nn ==="
for category_dir in "$EXPERIMENTAL_TMP_DIR"/*/; do
  for op_dir in "$category_dir"*/; do
    op_name=$(basename "$op_dir")
    py_files=$(find "$op_dir" -maxdepth 1 -name '*.py' 2>/dev/null)
    if [ -n "$py_files" ]; then
      category_name=$(basename "$category_dir")
      mkdir -p "$ASCEND_OPS_NN_DIR/$category_name/$op_name"
      if [ ! -e "$ASCEND_OPS_NN_DIR/$category_name/__init.py" ]; then
        cp "$SCRIPT_DIR/pe_init.py" "$ASCEND_OPS_NN_DIR/$category_name/__init__.py"
      fi
      cp -r "$op_dir"*.py "$ASCEND_OPS_NN_DIR/$category_name/$op_name/"
      if [ ! -e "$ASCEND_OPS_NN_DIR/$category_name/$op_name/__init.py" ]; then
        cp "$SCRIPT_DIR/pe_init.py" "$ASCEND_OPS_NN_DIR/$category_name/$op_name/__init__.py"
      fi
    fi
  done
done

echo "=== Clean ==="
cd "$SCRIPT_DIR"
python3 setup.py clean

echo "=== Build wheel ==="
python3 -m build --wheel --no-isolation

# Move wheel to build_out
mkdir -p "$BUILD_OUT_DIR"
mv "$SCRIPT_DIR/dist/"*.whl "$BUILD_OUT_DIR/"

# Clean up temporary directory
rm -rf "$EXPERIMENTAL_TMP_DIR"
rm -rf "$ASCEND_OPS_NN_DIR"/*/

echo "=== Build PyTorch extension successfully ==="