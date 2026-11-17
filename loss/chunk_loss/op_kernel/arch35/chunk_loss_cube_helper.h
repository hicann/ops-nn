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
 * \file chunk_loss_cube_helper.h
 * \brief ChunkLoss arch35 Cube 基础原语封装
 *
 * 封装 Nd2Nz、LoadData、Mmad、Fixpipe 等 AscendC Cube 接口。
 */
#ifndef CHUNKLOSS_ARCH35_CUBE_HELPER_H_
#define CHUNKLOSS_ARCH35_CUBE_HELPER_H_

#include "chunk_loss_common_arch35.h"

namespace chunkloss {

constexpr uint32_t kCubeBlock = 16;
constexpr uint32_t kC0Fp16 = 16; // 32B / sizeof(half)

CHUNKLOSS_DEVICE uint32_t CeilDivU(uint32_t a, uint32_t b) { return (a + b - 1u) / b; }

CHUNKLOSS_DEVICE uint32_t CeilAlignU(uint32_t a, uint32_t b) { return CeilDivU(a, b) * b; }

CHUNKLOSS_DEVICE uint32_t MmadMFloor(uint32_t m) { return m < 2u ? 2u : m; }

/** BSA load_data_gm_2_l1: RowMajor [m,k] GM -> NZ L1. */
template <typename T>
CHUNKLOSS_DEVICE void LoadGm2L1(const AscendC::LocalTensor<T>& dstL1, const AscendC::GlobalTensor<T>& srcGm,
                                int32_t mSize, int32_t kSize, int32_t mSizeAlign, int32_t /*kSizeAlign*/,
                                int32_t srcStride)
{
    AscendC::Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.nValue = mSize;
    nd2nzPara.dValue = kSize;
    nd2nzPara.srcDValue = srcStride;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzC0Stride = mSizeAlign;
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.dstNzMatrixStride = 0;
    AscendC::DataCopy(dstL1, srcGm, nd2nzPara);
}

/** BSA load_data_l1_2_l0_nz: left no-transpose / right transpose. */
template <typename T>
CHUNKLOSS_DEVICE void LoadL1ToL0Nz(const AscendC::LocalTensor<T>& dstL0, const AscendC::LocalTensor<T>& srcL1,
                                   int32_t mSizeAlign, int32_t kSizeAlign)
{
    AscendC::LoadData2DParamsV2 p;
    p.mStartPosition = 0;
    p.kStartPosition = 0;
    p.mStep = mSizeAlign / 16;
    p.kStep = kSizeAlign / static_cast<int32_t>(kC0Fp16);
    p.srcStride = mSizeAlign / 16;
    p.dstStride = mSizeAlign / 16;
    p.ifTranspose = false;
    AscendC::LoadData(dstL0, srcL1, p);
}

/**
 * BSA load_data_l1_2_l0_zn (FP16/BF16): left transpose / right no-transpose.
 * mFracStart/mFracCount select a row-fractal window inside parent mSizeAlign (srcStride).
 */
template <typename T>
CHUNKLOSS_DEVICE void LoadL1ToL0Zn(const AscendC::LocalTensor<T>& dstL0, const AscendC::LocalTensor<T>& srcL1,
                                   int32_t mSizeAlign, int32_t kSizeAlign, int32_t mFracStart, int32_t mFracCount)
{
    AscendC::LoadData2DParamsV2 p;
    p.kStartPosition = 0;
    p.ifTranspose = true;
    p.mStep = 1;
    p.kStep = kSizeAlign / static_cast<int32_t>(kC0Fp16);
    p.srcStride = mSizeAlign / 16;
    p.dstStride = 1;
    for (int32_t i = 0; i < mFracCount; ++i) {
        p.mStartPosition = mFracStart + i;
        AscendC::LoadData(dstL0[i * kSizeAlign * 16], srcL1, p);
    }
}

template <typename TC, typename TA, typename TB>
CHUNKLOSS_DEVICE void MmadUnit(const AscendC::LocalTensor<TC>& l0c, const AscendC::LocalTensor<TA>& l0a,
                               const AscendC::LocalTensor<TB>& l0b, uint32_t m, uint32_t n, uint32_t k, bool initC,
                               uint8_t unitFlag)
{
    AscendC::MmadParams mp;
    mp.m = MmadMFloor(m);
    // BSA FP16: madParams.n = nProcess (actual). Pad fractals must not be computed.
    mp.n = n < 1u ? 1u : n;
    mp.k = k;
    mp.cmatrixSource = false;
    mp.cmatrixInitVal = initC;
    mp.unitFlag = unitFlag;
    AscendC::Mmad(l0c, l0a, l0b, mp);
}

constexpr AscendC::FixpipeConfig kFixpipeRowMajorGm = {AscendC::CO2Layout::ROW_MAJOR, false};
constexpr AscendC::FixpipeConfig kFixpipeRowMajorUb = {AscendC::CO2Layout::ROW_MAJOR, true};

template <typename COut, typename CAcc>
CHUNKLOSS_DEVICE void FixpipeGmRowMajor(AscendC::GlobalTensor<COut> gm, const AscendC::LocalTensor<CAcc>& l0c,
                                        uint32_t m, uint32_t n, uint32_t dstStride, uint8_t unitFlag)
{
    AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> p;
    // BSA MM12→UB uses nProcessAlign; DQ/DK GM uses actual. For ZW tile GM (dstStride
    // usually kTileN) prefer nAlign so L0C extent matches Load; for tight dH strides
    // callers pass actual n and dstStride==fullH.
    p.nSize = (dstStride >= CeilAlignU(n < 1u ? 1u : n, kCubeBlock)) ? CeilAlignU(n < 1u ? 1u : n, kCubeBlock) :
                                                                       (n < 1u ? 1u : n);
    p.mSize = m;
    p.srcStride = CeilAlignU(m, kCubeBlock);
    p.dstStride = dstStride;
    p.unitFlag = unitFlag;
    p.reluEn = false;
    if constexpr (sizeof(COut) == sizeof(float)) {
        p.quantPre = QuantMode_t::NoQuant;
    } else {
        p.quantPre = QuantMode_t::F322F16;
    }
    AscendC::Fixpipe<COut, CAcc, kFixpipeRowMajorGm>(gm, l0c, p);
}

/** BSA MM12 Fixpipe→UB with dualDstCtl=1 (1AIC:2AIV). */
CHUNKLOSS_DEVICE void FixpipeUbSplitM(const AscendC::LocalTensor<float>& ub, const AscendC::LocalTensor<float>& l0c,
                                      uint32_t m, uint32_t n, uint32_t dstStride, uint8_t unitFlag)
{
    AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> p;
    p.nSize = CeilAlignU(n, kCubeBlock);
    p.mSize = CeilAlignU(m, 2u);
    p.srcStride = CeilAlignU(m, kCubeBlock);
    p.dstStride = dstStride;
    p.quantPre = QuantMode_t::NoQuant;
    p.reluEn = false;
    p.unitFlag = unitFlag;
    p.dualDstCtl = 1;
    AscendC::Fixpipe<float, float, kFixpipeRowMajorUb>(ub, l0c, p);
}

} // namespace chunkloss

#endif
