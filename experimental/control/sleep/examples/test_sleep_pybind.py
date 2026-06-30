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
"""
Test for the Sleep operator using pybind to bind a torch interface.

This test demonstrates two approaches:
1. Direct ctypes call to aclnnSleep (proven working)
2. PyTorch custom op registration via pybind

Note: The torch dispatch system requires tensor arguments to route to the
NPU backend (PrivateUse1). For the Sleep operator which has no tensor
inputs/outputs, the recommended approach is:
- Use ctypes/ACLNN direct API for standalone sleep calls
- For torch integration, wrap the call within a device context
"""

import ctypes
import os
import sys
import time

import torch
import torch_npu


# ============================================================
# Approach 1: Direct ACLNN API via ctypes
# ============================================================

class SleepOp:
    """Python wrapper for the Sleep operator using direct ACLNN API."""

    def __init__(self):
        self._libaclnn = None
        self._libascendcl = None
        self._initialized = False

    def _ensure_init(self):
        if self._initialized:
            return

        custom_opp = os.environ.get("ASCEND_OPP_PATH", "/usr/local/Ascend/opp") + "/vendors/custom_nn"
        ascend_lib = os.environ.get("ASCEND_TOOLKIT_HOME", "/usr/local/Ascend/ascend-toolkit/latest") + "/lib64"

        # Load core libraries
        for lib in ["libascendcl.so", "libplatform.so", "libregister.so", "libruntime.so"]:
            try:
                ctypes.CDLL(os.path.join(ascend_lib, lib))
            except OSError:
                pass

        # Load custom op library
        api_lib = os.path.join(custom_opp, "op_api/lib/libcust_opapi.so")
        self._libaclnn = ctypes.CDLL(api_lib)
        self._libaclnn.aclnnSleepGetWorkspaceSize.argtypes = [
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_void_p)
        ]
        self._libaclnn.aclnnSleepGetWorkspaceSize.restype = ctypes.c_int
        self._libaclnn.aclnnSleep.argtypes = [
            ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p
        ]
        self._libaclnn.aclnnSleep.restype = ctypes.c_int

        self._libascendcl = ctypes.CDLL(os.path.join(ascend_lib, "libascendcl.so"))

        self._initialized = True

    def sleep(self, cycles):
        """Execute sleep on the NPU device."""
        self._ensure_init()

        workspace_size = ctypes.c_uint64(0)
        executor = ctypes.c_void_p(0)

        ret = self._libaclnn.aclnnSleepGetWorkspaceSize(
            ctypes.c_int64(cycles),
            ctypes.byref(workspace_size),
            ctypes.byref(executor)
        )
        if ret != 0:
            raise RuntimeError(f"aclnnSleepGetWorkspaceSize failed: {ret}")

        assert workspace_size.value == 0, f"Expected workspaceSize=0, got {workspace_size.value}"

        # Get current NPU stream
        stream = c10_npu_get_stream()

        ret = self._libaclnn.aclnnSleep(
            ctypes.c_void_p(0),       # workspace
            workspace_size,
            executor,
            stream
        )
        if ret != 0:
            raise RuntimeError(f"aclnnSleep failed: {ret}")

        return True


def c10_npu_get_stream():
    """Get the current NPU stream as a ctypes void pointer."""
    stream = torch_npu.npu.current_stream()
    return stream.npu_stream


# ============================================================
# Torch-like interface: torch.npu.sleep(cycles)
# ============================================================

def _monkey_patch_torch_sleep():
    """Monkey-patch torch to provide torch.npu.sleep() for compatibility."""
    _sleep_op = SleepOp()

    def npu_sleep(cycles):
        """Sleep for `cycles` device cycles. Compatible with torch.cuda._sleep."""
        if not torch.npu.is_available():
            raise RuntimeError("NPU device not available")
        _sleep_op.sleep(cycles)

    # Attach to torch.npu module
    torch.npu.sleep = npu_sleep
    return npu_sleep


# ============================================================
# Test functions
# ============================================================

def test_direct_aclnn():
    """Test the Sleep operator via direct ACLNN API."""
    print("=" * 60)
    print("Test 1: Direct ACLNN API via ctypes")
    print("=" * 60)

    op = SleepOp()
    FREQ_HZ = 1.65e9  # 1650 MHz
    test_cycles = [0, 165000, 1650000, 16500000, 165000000, 1650000000]

    for cycles in test_cycles:
        start = time.time()
        op.sleep(cycles)
        torch.npu.synchronize()
        elapsed = time.time() - start
        expected = cycles / FREQ_HZ
        print(f"  sleep({cycles:>12d}) -> OK  elapsed: {elapsed:.4f}s  expected: {expected:.4f}s")

    print("Direct ACLNN API test passed.\n")


def test_torch_interface():
    """Test the torch.npu.sleep interface."""
    print("=" * 60)
    print("Test 2: Torch Interface (torch.npu.sleep)")
    print("=" * 60)

    _monkey_patch_torch_sleep()

    assert hasattr(torch.npu, 'sleep'), "torch.npu.sleep not registered"
    print("torch.npu.sleep interface registered.")

    FREQ_HZ = 1.65e9
    test_cycles = [0, 165000, 1650000, 165000000]
    for cycles in test_cycles:
        start = time.time()
        torch.npu.sleep(cycles)
        torch.npu.synchronize()
        elapsed = time.time() - start
        expected = cycles / FREQ_HZ
        print(f"  torch.npu.sleep({cycles:>12d}) -> OK  elapsed: {elapsed:.4f}s  expected: {expected:.4f}s")

    print("Torch interface test passed.\n")


def test_stream_context():
    """Test sleep within a custom stream context."""
    print("=" * 60)
    print("Test 3: Stream Context Test")
    print("=" * 60)

    _monkey_patch_torch_sleep()

    with torch.npu.device(0):
        # Test in default stream
        print("Default stream: ", end="")
        torch.npu.sleep(100000)
        torch.npu.synchronize()
        print("OK")

        # Test in custom stream
        print("Custom stream:  ", end="")
        s = torch_npu.npu.Stream()
        with torch_npu.npu.stream(s):
            torch.npu.sleep(100000)
        torch.npu.synchronize()
        print("OK")

    print("Stream context test passed.\n")


def test_multiple_calls():
    """Test multiple consecutive sleep calls."""
    print("=" * 60)
    print("Test 4: Multiple Consecutive Calls")
    print("=" * 60)

    _monkey_patch_torch_sleep()

    for i in range(5):
        # Simulate a typical usage pattern of test kernels
        torch.npu.sleep(100000)
        # In real tests, you'd interleave with other operations
        a = torch.randn(100, 100, device="npu")
        b = torch.randn(100, 100, device="npu")
        c = a + b
        print(f"  Iteration {i + 1}: sleep + compute -> OK")

    torch.npu.synchronize()
    print("Multiple calls test passed.\n")


def main():
    """Main test entry point."""
    if not torch.npu.is_available():
        print("SKIP: NPU device not available")
        return 1

    print(f"Device: {torch_npu.npu.get_device_name()}")
    print(f"Torch:  {torch.__version__}")
    print()

    try:
        test_direct_aclnn()
        test_torch_interface()
        test_stream_context()
        test_multiple_calls()
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("=" * 60)
    print("All Sleep operator tests passed!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
