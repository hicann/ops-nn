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
Test for the Sleep operator using ACLNN C API via ctypes.

This script directly tests the installed aclnnSleep API.
"""

import ctypes
import os
import sys
import time


def load_libraries():
    """Load required CANN and custom op libraries."""
    cann_lib = os.environ.get("ASCEND_TOOLKIT_HOME", "/usr/local/Ascend/ascend-toolkit/latest") + "/lib64"
    custom_opp_path = os.environ.get("ASCEND_OPP_PATH", "/usr/local/Ascend/opp") + "/vendors/custom_nn"

    # Load core CANN libraries first
    libs_to_load = [
        "libascendcl.so",
        "libplatform.so",
        "libregister.so",
        "libtiling_api.so",
        "libruntime.so",
    ]

    for lib in libs_to_load:
        path = os.path.join(cann_lib, lib)
        if os.path.exists(path):
            ctypes.CDLL(path)

    # Load custom operator API library
    api_lib_path = os.path.join(custom_opp_path, "op_api/lib/libcust_opapi.so")
    lib = ctypes.CDLL(api_lib_path)

    return lib


def test_sleep():
    """Test the aclnnSleep operator."""

    print("=" * 60)
    print("Sleep Operator ACLNN API Test")
    print("=" * 60)

    # Load libraries
    print("\nLoading libraries...")
    libaclnn = load_libraries()
    print("Libraries loaded successfully.")

    # Set up function signatures
    libaclnn.aclnnSleepGetWorkspaceSize.argtypes = [
        ctypes.c_int64,                          # cycles
        ctypes.POINTER(ctypes.c_uint64),         # workspaceSize
        ctypes.POINTER(ctypes.c_void_p)          # executor
    ]
    libaclnn.aclnnSleepGetWorkspaceSize.restype = ctypes.c_int

    libaclnn.aclnnSleep.argtypes = [
        ctypes.c_void_p,     # workspace
        ctypes.c_uint64,     # workspaceSize
        ctypes.c_void_p,     # executor
        ctypes.c_void_p      # stream
    ]
    libaclnn.aclnnSleep.restype = ctypes.c_int

    # ACL initialization
    libascendcl = ctypes.CDLL("libascendcl.so")

    ret = libascendcl.aclInit(None)
    assert ret == 0, f"aclInit failed: {ret}"
    print("aclInit: OK")

    ret = libascendcl.aclrtSetDevice(0)
    assert ret == 0, f"aclrtSetDevice failed: {ret}"
    print("Set device 0: OK")

    stream = ctypes.c_void_p(0)
    libascendcl.aclrtCreateStream.restype = ctypes.c_int
    libascendcl.aclrtCreateStream.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    ret = libascendcl.aclrtCreateStream(ctypes.byref(stream))
    assert ret == 0, f"aclrtCreateStream failed: {ret}"
    print("Create stream: OK")

    # freq = 1650 MHz, expected time = cycles / 1.65e9
    FREQ_HZ = 1.65e9
    test_cycles = [0, 165000, 1650000, 16500000, 165000000, 1650000000]
    print(f"\n--- Test Cases (freq = {FREQ_HZ/1e6:.0f} MHz) ---")

    for cycles in test_cycles:
        workspace_size = ctypes.c_uint64(0)
        executor = ctypes.c_void_p(0)

        ret = libaclnn.aclnnSleepGetWorkspaceSize(
            ctypes.c_int64(cycles),
            ctypes.byref(workspace_size),
            ctypes.byref(executor)
        )
        assert ret == 0, f"GetWorkspaceSize failed for cycles={cycles}: {ret}"
        assert workspace_size.value == 0, f"Expected workspaceSize=0, got {workspace_size.value}"

        workspace_addr = ctypes.c_void_p(0)
        if workspace_size.value > 0:
            libascendcl.aclrtMalloc.restype = ctypes.c_int
            libascendcl.aclrtMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint64, ctypes.c_int]
            ret = libascendcl.aclrtMalloc(ctypes.byref(workspace_addr), workspace_size.value, 0x400)
            assert ret == 0, f"Malloc failed: {ret}"

        start = time.time()
        ret = libaclnn.aclnnSleep(workspace_addr, workspace_size, executor, stream)
        assert ret == 0, f"Sleep failed for cycles={cycles}: {ret}"

        libascendcl.aclrtSynchronizeStream.restype = ctypes.c_int
        libascendcl.aclrtSynchronizeStream.argtypes = [ctypes.c_void_p]
        ret = libascendcl.aclrtSynchronizeStream(stream)
        assert ret == 0, f"Synchronize failed: {ret}"

        elapsed = time.time() - start
        expected = cycles / FREQ_HZ
        print(f"  sleep({cycles:>12d}) -> OK  elapsed: {elapsed:.4f}s  expected: {expected:.4f}s")

    # Cleanup
    print("\n--- Cleanup ---")
    libascendcl.aclrtDestroyStream.restype = ctypes.c_int
    libascendcl.aclrtDestroyStream.argtypes = [ctypes.c_void_p]
    ret = libascendcl.aclrtDestroyStream(stream)
    print(f"Destroy stream: OK")

    libascendcl.aclrtResetDevice.restype = ctypes.c_int
    libascendcl.aclrtResetDevice.argtypes = [ctypes.c_int32]
    ret = libascendcl.aclrtResetDevice(0)
    print(f"Reset device: OK")

    libascendcl.aclFinalize.restype = ctypes.c_int
    ret = libascendcl.aclFinalize()
    print(f"aclFinalize: OK")

    print("\n" + "=" * 60)
    print("All Sleep operator tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_sleep()
