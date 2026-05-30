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
 * \file conv_bp_filter_sub_func_load_common.h
 * \brief Common definitions for conv3d backprop filter load functions
 */

#ifndef CONV3D_BP_FILTER_SUB_FUNC_LOAD_COMMON_H
#define CONV3D_BP_FILTER_SUB_FUNC_LOAD_COMMON_H

namespace ConvolutionBackpropFunc {

using AscendC::Dn2NzParams;
using AscendC::Nd2NzParams;

struct Out2L1ScalarParams {
    // to L1A
    bool isLoad2L1A{false};
    bool isLastMAL1{false};
    bool isFreeAL1{false};
    uint64_t out2A1SrcAddr{0};

    // to L1B
    bool isLoad2L1B{false};
    bool isFreeBL1{false};
    uint64_t out2B1SrcAddr{0};
    uint32_t singleShapeHi{0};
    uint32_t singleShapeWi{0};
};

struct B1HiCopyParams {
    uint32_t b1SrcHi{0};
    uint32_t hiCopyLen{0};
    uint32_t padUp{0};
};

__aicore__ inline uint64_t Ceil(uint64_t a, uint32_t b)
{
    if (b == 0) {
        // 处理除数为0的情况，直接返回a
        return a;
    }

    return (a + b - 1) / b;
}

template <class Intf>
__aicore__ inline void InitZeroValue(Intf* self, const LocalTensor<typename Intf::SrcT>& buf)
{
    uint32_t len = buf.GetSize() * sizeof(typename Intf::SrcT);
    constexpr int32_t SHIFT_BITS_5 = 5;

    AscendC::InitConstValueParams<typename Intf::SrcT> initConstValueParams;
    initConstValueParams.repeatTimes = 1;
    initConstValueParams.blockNum = len >> SHIFT_BITS_5; // 除以blockSize
    initConstValueParams.dstGap = 0;
    initConstValueParams.initValue = (typename Intf::SrcT)(0);
    InitConstValue(buf, initConstValueParams);

    PipeBarrier<PIPE_MTE2>();
}

} // namespace ConvolutionBackpropFunc

#endif
