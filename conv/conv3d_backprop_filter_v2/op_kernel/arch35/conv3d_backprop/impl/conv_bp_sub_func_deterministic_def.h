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
 * \file conv_bp_sub_func_deterministic_def.h
 * \brief 确定性计算的定义层: 常量、数据结构与纯数学工具, 不含流水与搬运调用
 */

#ifndef CONV3D_BP_FILTER_SUB_FUNC_DETERMINISTIC_DEF_H
#define CONV3D_BP_FILTER_SUB_FUNC_DETERMINISTIC_DEF_H
constexpr uint8_t SREG_PROC_NUM = 64;

namespace ConvolutionBackpropFunc {
constexpr uint8_t SYNC_MODE0 = 0;

constexpr uint8_t SYNC_MODE2 = 2;

constexpr uint8_t SYNC_MODE4 = 4;

constexpr uint8_t DST_DTYPE_BYTES = 4;

constexpr uint8_t ONE_BLK_SHIFT_SIZE = 5;

constexpr uint32_t VECTOR_UB_SIZE_HALF = AscendC::TOTAL_UB_SIZE >> 1; // UB大小的一半

constexpr uint32_t FLOAT_SHIFT_SIZE = 2;

constexpr uint32_t CUT_FOUR = 4;

constexpr uint32_t RELATED_CORE_NUM = 2;

constexpr uint32_t DOUBLE = 2;

constexpr uint32_t PINGPONG_NUM = 2;

constexpr uint64_t CUBE_WORKSPACE = AscendC::TOTAL_L0C_SIZE >> 2; // 2: sizeof(float)

constexpr uint64_t HALF_CUBE_WORKSPACE = CUBE_WORKSPACE >> 1;

constexpr uint64_t QUARTER_CUBE_WORKSPACE = CUBE_WORKSPACE >> 2;

constexpr FixpipeConfig CFG_NZ = {CO2Layout::NZ, true};

static constexpr uint16_t SYNC_AIV_AIC_DET_FLAG = 6;

struct DeterMinisticShape {
    uint32_t mSize[CUT_FOUR] = {0, 0, 0, 0};
    uint32_t nSize[CUT_FOUR] = {0, 0, 0, 0};
    uint64_t mnSize[CUT_FOUR] = {0, 0, 0, 0};
    uint64_t addrOffset[CUT_FOUR] = {0, 0, 0, 0};
    uint32_t usedMSize[CUT_FOUR] = {0, 0, 0, 0};
    uint32_t usedNSize[CUT_FOUR] = {0, 0, 0, 0};
    bool isNTail[CUT_FOUR] = {false, false, false, false};
};

struct CutDeterMinisticMNSize {
    uint32_t curMSize = 0;
    uint32_t curNSize = 0;
    uint32_t usedMSize = 0;
    uint32_t usedNSize = 0;
    bool isNTail = false;
};

template <typename DstT>
struct ScatterDeterCtx {
    uint64_t baseCin;
    uint64_t coutNum;
    __ubuf__ uint32_t* indexPtr;
    __ubuf__ DstT* srcPtr;
    __ubuf__ DstT* dstPtr;
    uint32_t c0PerReg;
    uint32_t sreg;
    uint32_t tailSreg;
};

static __aicore__ inline bool IsDivisible2(const uint64_t a) { return a == ((a >> 1) << 1); }
static __aicore__ inline void DivTwoNumersInHalf(uint64_t& a, uint64_t& b)
{
    // 能对半切就切，不能就切大的数
    if (IsDivisible2(b)) {
        b = b >> 1;
    } else if (IsDivisible2(a)) {
        a = a >> 1;
    } else if (b > a) {
        b = (b + 1) >> 1;
    } else {
        a = (a + 1) >> 1;
    }
    // 最小要为1， 不能是0
    a = a < 1 ? 1 : a;
    b = b < 1 ? 1 : b;
}
} // namespace ConvolutionBackpropFunc

#endif
