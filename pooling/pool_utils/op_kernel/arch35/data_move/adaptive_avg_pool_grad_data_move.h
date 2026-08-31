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
 * \file adaptive_avg_pool_grad_data_move.h
 * \brief AdaptiveAvgPool2DGrad/AdaptiveAvgPool3DGrad 共用的输出梯度 gather/scatter 与输入转置接口。
 */

#ifndef POOL_UTILS_ARCH35_DATA_MOVE_ADAPTIVE_AVG_POOL_GRAD_DATA_MOVE_H_
#define POOL_UTILS_ARCH35_DATA_MOVE_ADAPTIVE_AVG_POOL_GRAD_DATA_MOVE_H_

#include <cstdint>
#include <type_traits>

#include "kernel_operator.h"
#include "pool_utils/arch35/data_move/adaptive_pool_transpose_data_move.h"

namespace PoolUtils {
namespace DataMove {

constexpr AscendC::Reg::CastTrait ADAPTIVE_GRAD_CAST_I64I32 = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::NO_SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND,
};

/*
 * 功能：按输出梯度索引把 UB 中的输出梯度 gather 到寄存器。
 * 说明：INDEX 为 int64_t 时先降位到 int32_t 并做 Pack，再按 uint32_t 索引 gather。
 */
template <typename INDEX, typename COMPUTE_TYPE>
__aicore__ inline void GatherCopyGradUb2Reg(AscendC::Reg::RegTensor<INDEX>& gradOutputUBIdx,
                                            AscendC::Reg::RegTensor<COMPUTE_TYPE>& gradOutputUbValue,
                                            __ubuf__ COMPUTE_TYPE* yAddr, uint32_t& maskCount)
{
    uint32_t maskCountTemp = maskCount;
    AscendC::Reg::MaskReg pregU32 = AscendC::Reg::UpdateMask<uint32_t>(maskCountTemp);
    if constexpr (std::is_same<INDEX, int64_t>::value) {
        AscendC::Reg::MaskReg allMask = AscendC::Reg::CreateMask<INDEX, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::RegTensor<int32_t> gradOutputUBIdxI32;
        AscendC::Reg::Cast<int32_t, int64_t, ADAPTIVE_GRAD_CAST_I64I32>(gradOutputUBIdxI32, gradOutputUBIdx, allMask);
        AscendC::Reg::Pack((AscendC::Reg::RegTensor<uint32_t>&)gradOutputUBIdxI32,
                           (AscendC::Reg::RegTensor<int64_t>&)gradOutputUBIdxI32);
        AscendC::Reg::Gather(gradOutputUbValue, yAddr, (AscendC::Reg::RegTensor<uint32_t>&)gradOutputUBIdxI32, pregU32);
    } else {
        AscendC::Reg::Gather(gradOutputUbValue, yAddr, (AscendC::Reg::RegTensor<uint32_t>&)gradOutputUBIdx, pregU32);
    }
}

/*
 * 功能：按输出梯度索引把寄存器中的累加结果 scatter 回 UB。
 * 说明：INDEX 为 int64_t 时先降位到 int32_t 并做 Pack，再按 uint32_t 索引 scatter。
 */
template <typename INDEX, typename COMPUTE_TYPE>
__aicore__ inline void ScatterCopyGradReg2Ub(AscendC::Reg::RegTensor<INDEX>& gradOutputUBIdx,
                                             AscendC::Reg::RegTensor<COMPUTE_TYPE>& gradOutputUbValue,
                                             __ubuf__ COMPUTE_TYPE* yAddr, uint32_t& maskCount)
{
    uint32_t maskCountTemp = maskCount;
    AscendC::Reg::MaskReg pregU32 = AscendC::Reg::UpdateMask<uint32_t>(maskCountTemp);
    if constexpr (std::is_same<INDEX, int64_t>::value) {
        AscendC::Reg::MaskReg allMask = AscendC::Reg::CreateMask<INDEX, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::RegTensor<int32_t> gradOutputUBIdxI32;
        AscendC::Reg::Cast<int32_t, int64_t, ADAPTIVE_GRAD_CAST_I64I32>(gradOutputUBIdxI32, gradOutputUBIdx, allMask);
        AscendC::Reg::Pack((AscendC::Reg::RegTensor<uint32_t>&)gradOutputUBIdxI32,
                           (AscendC::Reg::RegTensor<int64_t>&)gradOutputUBIdxI32);
        AscendC::Reg::Scatter(yAddr, gradOutputUbValue, (AscendC::Reg::RegTensor<uint32_t>&)gradOutputUBIdxI32,
                              pregU32);
    } else {
        AscendC::Reg::Scatter(yAddr, gradOutputUbValue, (AscendC::Reg::RegTensor<uint32_t>&)gradOutputUBIdx, pregU32);
    }
}

/*
 * 功能：把已搬入的输入按类型选择 B32/B16 转置后送入转置队列。
 * 说明：转置后插入 V_S 同步，保证后续标量读取到转置结果。
 */
template <typename T, int32_t BUFFER_DEPTH>
__aicore__ inline void TransInput(AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_DEPTH>& inputQue,
                                  AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_DEPTH>& transQue, uint32_t rowNum,
                                  uint32_t colNum, uint32_t ubBlockSize)
{
    AscendC::LocalTensor<T> srcLocal = inputQue.template DeQue<T>();
    AscendC::LocalTensor<T> dstLocal = transQue.template AllocTensor<T>();

    if constexpr (AscendC::IsSameType<T, float>::value) {
        TransposeB32<T>(dstLocal, srcLocal, rowNum, colNum, ubBlockSize);
    } else {
        TransposeB16<T>(dstLocal, srcLocal, rowNum, colNum, ubBlockSize);
    }

    event_t eventIDVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_S));
    AscendC::SetFlag<AscendC::HardEvent::V_S>(eventIDVToS);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(eventIDVToS);

    inputQue.FreeTensor(srcLocal);
    transQue.EnQue(dstLocal);
}

} // namespace DataMove
} // namespace PoolUtils

#endif // POOL_UTILS_ARCH35_DATA_MOVE_ADAPTIVE_AVG_POOL_GRAD_DATA_MOVE_H_
