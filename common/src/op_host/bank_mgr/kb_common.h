/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef RUNTIME_KB_COMMON_UTILS_KB_COMMON_H_
#define RUNTIME_KB_COMMON_UTILS_KB_COMMON_H_
#include <memory>
#include "kb_log.h"

namespace RuntimeKb {
template <typename T, typename... Args>
inline std::shared_ptr<T> MakeShared(Args&&... args)
{
    try {
        return std::make_shared<T>(std::forward<Args>(args)...);
    } catch (const std::bad_alloc&) {
        CANNKB_LOGE("Make shared failed");
        return nullptr;
    }
    return nullptr;
}

#define RTKB_CHECK(cond, logFunc, returnExpr) \
    do {                                      \
        if ((cond)) {                         \
            logFunc;                          \
            returnExpr;                       \
        }                                     \
    } while (false)
} // namespace RuntimeKb
#endif
