/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef MAT_MUL_V3_ARCH31_L1_POLICY_H
#define MAT_MUL_V3_ARCH31_L1_POLICY_H

#include <stdint.h>

namespace matmul_v3_arch31 {
constexpr int32_t kAl1StripMarker = 0x414C3101;
constexpr int32_t kSplitKAl1Marker = 0x414C3102;
constexpr int64_t kL1Bytes = 512 * 1024;
constexpr int64_t kBPipelineReserveBytes = 128 * 1024;
constexpr int64_t kAl1StripBudgetBytes = kL1Bytes - kBPipelineReserveBytes;
constexpr int64_t kFp16Bytes = 2;
constexpr int64_t kCubeAlign = 16;
constexpr int64_t kMaxStripRows = 128;
constexpr int64_t kAl1BaseN = 128;
constexpr int64_t kAl1BaseK = 64;
constexpr int64_t kMinReuseNTiles = 4;
constexpr int64_t kSplitKUbBytes = 64 * 1024;
} // namespace matmul_v3_arch31

#endif
