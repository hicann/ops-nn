/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_MATH_DEV_TESTS_UT_COMMON_TILING_PARSE_CONTEXT_FAKER_H
#define OPS_MATH_DEV_TESTS_UT_COMMON_TILING_PARSE_CONTEXT_FAKER_H

#include <vector>
#include "kernel_run_context_holder.h"

namespace gert {
class TilingParseContextFaker : public OpTilingParseContextBuilder, public KernelRunContextHolder {
public:
    TilingParseContextFaker() = default;
    TilingParseContextFaker& operator=(TilingParseContextFaker&&);
    TilingParseContextFaker(TilingParseContextFaker&&);

    TilingParseContextFaker& SetOpType(const std::string opType);

    TilingParseContextFaker& KernelIONum(size_t inputNum, size_t outputNum);

    TilingParseContextFaker& Inputs(const std::vector<void*>& inputs);

    TilingParseContextFaker& Outputs(const std::vector<void*>& outputs);

    KernelRunContextHolder Build();
};

using KernelRunContextFaker = TilingParseContextFaker;
} // namespace gert
#endif // OPS_MATH_DEV_TESTS_UT_COMMON_TILING_PARSE_CONTEXT_FAKER_H
