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

set -e
cd "$(dirname "$0")"

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Build the project
echo "Building the project..."
python3 setup.py clean
python3 -m build --wheel --no-isolation
python3 -m pip install dist/*.whl --force-reinstall --no-deps

# Run tests
if [ -z "$NPU_ARCH" ]; then
    echo "Error: NPU_ARCH environment variable is not set"
    echo "Please set NPU_ARCH to 'Ascend950' or another architecture"
    exit 1
fi
echo "Running tests for NPU architecture: ${NPU_ARCH}"
echo "Running tests..."
if [ "$NPU_ARCH" = "Ascend950" ]; then
    pytest tests/conv3d_custom/ascend950 -v -s
else
    pytest tests/conv3d_custom/ascend910b -v -s
fi
echo "execute samples success"
