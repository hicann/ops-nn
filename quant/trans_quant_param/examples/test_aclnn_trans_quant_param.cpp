/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include "acl/acl.h"
#include "aclnnop/aclnn_trans_quant_param.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int main()
{
    float scaleArray[3] = {1.0, 1.0, 1.0};
    uint64_t scaleSize = 3;
    float offsetArray[3] = {1.0, 1.0, 1.0};
    uint64_t offsetSize = 3;
    uint64_t* result = nullptr;
    uint64_t resultSize = 0;
    auto ret = aclnnTransQuantParam(scaleArray, scaleSize, offsetArray, offsetSize, &result, &resultSize);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransQuantParam failed. ERROR: %d\n", ret); return ret);
    for (auto i = 0; i < resultSize; i++) {
        LOG_PRINT("result[%d] is: %ld\n", i, result[i]);
    }
    free(result);
    return 0;
}
