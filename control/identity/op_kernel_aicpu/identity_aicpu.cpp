/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "identity_aicpu.h"

#include "securec.h"
#include "log.h"
#include "status.h"
#include "utils/kernel_util.h"

namespace {
constexpr uint32_t kIdentityInputNum = 1;
constexpr uint32_t kIdentityOutputNum = 1;
const char *const kIdentity = "Identity";
}

namespace aicpu {
uint32_t IdentityCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kIdentityInputNum, kIdentityOutputNum),
                      "[%s] check params failed.", kIdentity);

  // parse params
  void *input_data = ctx.Input(0)->GetData();
  void *output_data = ctx.Output(0)->GetData();
  uint64_t input_size = ctx.Input(0)->GetDataSize();
  uint64_t output_size = ctx.Output(0)->GetDataSize();
  if (output_size < input_size) {
    KERNEL_LOG_WARN("[%s] output size [%ld] less than input size [%ld].",
                    kIdentity, output_size, input_size);
    input_size = output_size;
  } else if (output_size > input_size) {
    KERNEL_LOG_ERROR("[%s] output size [%ld] greater than input size [%ld].",
                     kIdentity, output_size, input_size);
    return KERNEL_STATUS_PARAM_INVALID;
  }

  // do copy
  if (output_data != input_data) {
    auto cpret = BiggerMemCpy(output_data, output_size, input_data, input_size);
    KERNEL_CHECK_FALSE(
        cpret, KERNEL_STATUS_INNER_ERROR,
        "[%s] memcpy_s to output failed, destMax [%ld], count [%ld].",
        kIdentity, output_size, input_size);
  }
  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(kIdentity, IdentityCpuKernel);
}  // namespace aicpu
