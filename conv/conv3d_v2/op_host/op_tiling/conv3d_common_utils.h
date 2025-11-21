/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file conv3d_common_utils.h
 * \brief Common utility functions for conv3d operations
 */

#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_CONV3D_COMMON_UTILS_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_CONV3D_COMMON_UTILS_H

namespace Conv3dCommon {

// Common utility function for multiplication with overflow check
template <typename T>
bool MulWithOverflowCheck(T &res, T a, T b)
{
    if (a == 0 || b == 0) {
        res = 0;
        return false;
    }
    T tmpRes = a * b;
    if (tmpRes / a != b) {
        return true;
    }
    res = tmpRes;
    return false;
}

// 调用时控制传参个数，避免栈溢出
template <typename T, typename... Args>
bool MulWithOverflowCheck(T &res, T a, T b, Args... args)
{
    T tmp;
    return MulWithOverflowCheck(tmp, a, b) || MulWithOverflowCheck(res, tmp, args...);
}

} // namespace Conv3dCommon

#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_CONV3D_COMMON_UTILS_H