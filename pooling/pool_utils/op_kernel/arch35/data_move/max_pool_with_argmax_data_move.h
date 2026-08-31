/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file max_pool_with_argmax_data_move.h
 * \brief MaxPoolWithArgmax 系列算子共用的 UB 内结果回写接口。
 */

#ifndef POOL_UTILS_ARCH35_DATA_MOVE_MAX_POOL_WITH_ARGMAX_DATA_MOVE_H_
#define POOL_UTILS_ARCH35_DATA_MOVE_MAX_POOL_WITH_ARGMAX_DATA_MOVE_H_

#include <cstdint>

#include "kernel_operator.h"

namespace PoolUtils {
namespace DataMove {

template <typename T1, typename T2>
__aicore__ inline void CopyResultToUb(__ubuf__ T1* maxValueLocal, __ubuf__ T2* argmaxLocal, __ubuf__ T1* maxValueHelp,
                                      __ubuf__ T2* argmaxHelp)
{
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<T1> vreg0;
        AscendC::Reg::RegTensor<T2> argmaxUpdateVreg;
        AscendC::Reg::MaskReg pregAllT1 = AscendC::Reg::CreateMask<T1, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::MaskReg pregAllT2 = AscendC::Reg::CreateMask<T2, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::LoadAlign(vreg0, maxValueHelp);
        AscendC::Reg::LoadAlign(argmaxUpdateVreg, argmaxHelp);
        AscendC::Reg::StoreAlign(maxValueLocal, vreg0, pregAllT1);
        AscendC::Reg::StoreAlign(argmaxLocal, argmaxUpdateVreg, pregAllT2);
    }
}

} // namespace DataMove
} // namespace PoolUtils

#endif // POOL_UTILS_ARCH35_DATA_MOVE_MAX_POOL_WITH_ARGMAX_DATA_MOVE_H_
