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
 * \file adaptive_pool3d_parall_pool_data_move.h
 * \brief AdaptiveAvgPool3D/AdaptiveMaxPool3D parallel pool 模板共用的输入搬入接口。
 */

#ifndef POOL_UTILS_ARCH35_DATA_MOVE_ADAPTIVE_POOL3D_PARALL_POOL_DATA_MOVE_H_
#define POOL_UTILS_ARCH35_DATA_MOVE_ADAPTIVE_POOL3D_PARALL_POOL_DATA_MOVE_H_

#include <cstdint>

#include "op_kernel/math_util.h"
#include "kernel_operator.h"

namespace PoolUtils {
namespace DataMove {

/*
 * 功能：按 nc/d 两层 loop mode 把当前窗口覆盖的输入块搬入 UB，行内按 ubAlignNum 对齐。
 */
template <typename T>
__aicore__ inline void CopyInput(AscendC::TQue<AscendC::QuePosition::VECIN, 1>& inputQue,
                                 const AscendC::GlobalTensor<T>& xGm, uint32_t ncNum, uint32_t diDataLen,
                                 uint32_t hiDataLen, uint32_t wiDataLen, int64_t xOffset, uint32_t ubAlignNum,
                                 int64_t wIn, int64_t inHW, int64_t inDHW)
{
    AscendC::LocalTensor<T> xLocal = inputQue.template AllocTensor<T>();

    uint32_t wiDataAlign = Ops::Base::CeilAlign(wiDataLen, ubAlignNum);
    AscendC::DataCopyExtParams paramsIn = {
        static_cast<uint16_t>(hiDataLen), static_cast<uint32_t>(wiDataLen * sizeof(T)),
        static_cast<uint32_t>((wIn - wiDataLen) * sizeof(T)), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    AscendC::DataCopyPadExtParams<T> padParams = {false, 0, 0, 0};

    AscendC::LoopModeParams loopModeParams;
    loopModeParams.loop1Size = diDataLen;
    loopModeParams.loop2Size = ncNum;
    loopModeParams.loop1SrcStride = inHW * sizeof(T);
    loopModeParams.loop2SrcStride = inDHW * sizeof(T);
    loopModeParams.loop1DstStride = hiDataLen * wiDataAlign * sizeof(T);
    loopModeParams.loop2DstStride = diDataLen * hiDataLen * wiDataAlign * sizeof(T);

    AscendC::SetLoopModePara(loopModeParams, AscendC::DataCopyMVType::OUT_TO_UB);
    AscendC::DataCopyPad(xLocal, xGm[xOffset], paramsIn, padParams);
    AscendC::ResetLoopModePara(AscendC::DataCopyMVType::OUT_TO_UB);
    inputQue.EnQue(xLocal);
}

} // namespace DataMove
} // namespace PoolUtils

#endif // POOL_UTILS_ARCH35_DATA_MOVE_ADAPTIVE_POOL3D_PARALL_POOL_DATA_MOVE_H_
