/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#ifndef OPS_TEST_UTILS_AICPU_TEST_UTISL_H_
#define OPS_TEST_UTILS_AICPU_TEST_UTISL_H_
#include <random>
#include <vector>
#include <math.h>
#include <iostream>
#include "Eigen/Core"

uint64_t CalTotalElements(std::vector<std::vector<int64_t>> &shapes,
                          uint32_t index);

bool CompareResult(Eigen::half output[], Eigen::half expect_output[],
                   uint64_t num);

bool CompareResultAllClose(float output[], float expect_output[], uint64_t num);

template <typename T>
bool CompareResult(T output[], T expect_output[], uint64_t num) {
  bool result = true;
  for (uint64_t i = 0; i < num; ++i) {
    if (output[i] != expect_output[i]) {
      std::cout << "output[" << i << "] = ";
      std::cout << output[i];
      std::cout << ", expect_output[" << i << "] = ";
      std::cout << expect_output[i] << std::endl;
      result = false;
    }
  }
  return result;
}

template <typename T>
void SetRandomValue(T input[], uint64_t num, float min = 0.0,
                    float max = 10.0) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(min, max);
  for (uint64_t i = 0; i < num; ++i) {
    input[i] = static_cast<T>(dis(gen));
  }
}

#define RUN_KERNEL(node_def, device_type, expect)                 \
  string node_def_str;                                            \
  node_def->SerializeToString(node_def_str);                      \
  CpuKernelContext ctx(device_type);                              \
  EXPECT_EQ(ctx.Init(node_def.get()), KERNEL_STATUS_OK);          \
  uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx); \
  EXPECT_EQ(ret, expect)

#define RUN_KERNEL_WITHBLOCK(node_def, device_type, expect, blkInfo)       \
  string node_def_str;                                            \
  node_def->SerializeToString(node_def_str);                      \
  auto blockNum = CpuKernelUtils::CreateAttrValue();              \
  blockNum->SetInt(blkInfo->block_num);                            \
  node_def->AddAttrs("block_num", blockNum.get());                \
  auto blockId = CpuKernelUtils::CreateAttrValue();               \
  blockId->SetInt(blkInfo->block_id);                              \
  node_def->AddAttrs("block_id", blockId.get());                  \
  CpuKernelContext ctx(device_type);                              \
  EXPECT_EQ(ctx.Init(node_def.get()), KERNEL_STATUS_OK);          \
  uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx); \
  EXPECT_EQ(ret, expect)

#endif