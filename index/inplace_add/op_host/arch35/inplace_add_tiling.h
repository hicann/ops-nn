/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_INPLACE_ADD_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_INPLACE_ADD_H_

#include <cstdint>

namespace optiling {
// 只保留 TilingPrepare4InplaceAdd 真正写入、Tiling4InplaceAdd 真正读取的两项。
struct InplaceAddCompileInfo {
    int64_t coreNum{1};
    int64_t ubSize{1};
};
} // namespace optiling

#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_INPLACE_ADD_H_
