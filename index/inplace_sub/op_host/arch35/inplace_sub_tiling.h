/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_INPLACE_SUB_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_INPLACE_SUB_H_

#include <cstdint>

namespace optiling {
struct InplaceSubCompileInfo {
    int64_t core_num{1};
    int64_t ub_size{1};
    int64_t var_size{1};
    int64_t indices_size{1};
    int64_t support_atomic{1};
};
} // namespace optiling

#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_INPLACE_SUB_H_
