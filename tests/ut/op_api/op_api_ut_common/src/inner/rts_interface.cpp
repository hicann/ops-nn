/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdlib>
#include <iostream>
#include <map>

#include "op_api_ut_common/inner/rts_interface.h"
#include "op_api_ut_common/inner/types.h"

using namespace std;

#define DEVICE_ID 0

void* MallocDeviceMemory(unsigned long size) { return nullptr; }

void FreeDeviceMemory(void* device_mem_ptr) { return; }

int MemcpyToDevice(const void* host_mem, void* dev_mem, unsigned long size) { return 0; }

int MemcpyFromDevice(void* host_mem, const void* dev_mem, unsigned long size) { return 0; }

bool IsFileExistsAccessable(string& name) { return (access(name.c_str(), F_OK) != -1); }

int RtsInit() { return 0; }

void RtsUnInit() { return; }

int SynchronizeStream() { return 0; }
