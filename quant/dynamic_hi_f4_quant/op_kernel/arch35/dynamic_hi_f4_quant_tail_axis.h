/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dynamic_hi_f4_quant_tail_axis.h
 * \brief axis=-1 路径：定长 tile 按展平流切分 + 按 tile 分核，
 *        单 __VEC_SCOPE__ 内 6 相位 VF 链 + 多深度缓冲环。
 */
#ifndef DYNAMIC_HIF4_QUANT_TAIL_AXIS_HIFLOAT4_H
#define DYNAMIC_HIF4_QUANT_TAIL_AXIS_HIFLOAT4_H

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"
#include "dynamic_hi_f4_quant_struct.h"

namespace DynamicHiF4Quant {
using namespace AscendC;

constexpr int64_t DQ_HIFP4_GROUP_ELE_NUM = 64;
constexpr int64_t DQ_HIFP4_GROUP_BYTE_NUM = 32; // 64 * 0.5 byte
constexpr int64_t DQ_HIFP4_SCALE_ALIGN = 256;
constexpr int64_t DQ_HIFP4_MAXPER8_ALIGN = 128;
constexpr int64_t DQ_HIFP4_SCALETMP_ALIGN = 256;
constexpr int64_t DQ_HIFP4_CAST_COPYOUT_PACK4_NUM = 4;
// 输入队列深度(编译器自动流水预取 MTE2)
constexpr int32_t INPUT_QUE_DEPTH = 4;
// y 输出槽与 maxper4 工作区共用：maxper4 生命期止于 VF 链前段，ComputeOut 覆写为量化结果
constexpr int32_t OUT_QUE_DEPTH = 2;
// scale 攒批搬出：队列深度 = 2 bank x 4 tile，每满 4 tile 或本核末 tile 下发批搬
constexpr int32_t SCALE_QUE_DEPTH = 8;
constexpr int32_t SCALE_TILES_PER_BANK = 4;
constexpr int64_t DIGIT_TWO = 2;
constexpr int64_t DIGIT_FOUR = 4;
constexpr int64_t DIGIT_EIGHT = 8;
constexpr int64_t BLOCK_SIZE = 64;
template <typename ST, typename DT, uint32_t useQueNum>
class DynamicHiF4QuantTailAxis {
public:
    __aicore__ inline DynamicHiF4QuantTailAxis(TPipe* pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR scale,
                                const DynamicHiF4QuantTilingData* __restrict tilingData);
    __aicore__ inline void Process();

protected:
    __aicore__ inline void CopyIn(const int64_t& gmOffset);
    __aicore__ inline void ComputeAll(int64_t regLoop, int64_t maxPer8RegLoop, int64_t eaRegLoop, int64_t scaleRegLoop,
                                      int64_t outRegLoop, __ubuf__ uint16_t* scaleSlotAddr);
    __aicore__ inline void ComputeMaxPer4(uint16_t regLoop, __ubuf__ float* xAddr1, __ubuf__ uint16_t* maxPer4BufAddr);
    __aicore__ inline void ComputeMaxPer8(uint16_t maxPer8RegLoop, __ubuf__ uint16_t* maxPer4BufReadAddr,
                                          __ubuf__ uint16_t* max64RawBufAddr, __ubuf__ uint16_t* maxPer8BufAddr);
    __aicore__ inline void ComputeEa(uint16_t eaRegLoop, __ubuf__ uint16_t* max64RawReadAddr,
                                     __ubuf__ uint16_t* e8BufAddr, __ubuf__ uint16_t* eaRecBufAddr);
    __aicore__ inline void ComputeTmpScale(uint16_t regLoop, __ubuf__ uint16_t* maxPer4BufRead2Addr,
                                           __ubuf__ uint16_t* maxPer8BufReadAddr, __ubuf__ uint16_t* e8BufRead1Addr,
                                           __ubuf__ uint32_t* e1p8BufAddr, __ubuf__ uint8_t* e1p16BufAddr,
                                           __ubuf__ uint16_t* tmpScaleAddr);
    __aicore__ inline void ComputeScale(uint16_t scaleRegLoop, __ubuf__ uint8_t* e8BufRead2Addr,
                                        __ubuf__ uint8_t* e1p8BufReadAddr, __ubuf__ uint8_t* e1p16BufReadAddr,
                                        __ubuf__ uint16_t* scaleAddr);
    __aicore__ inline void ComputeOut(uint16_t outRegLoop, __ubuf__ ST* xAddr2, __ubuf__ uint16_t* tmpScaleReadAddr,
                                      __ubuf__ int32_t* yAddr);
    __aicore__ inline void CopyOutY(const int64_t& gmOffset);
    __aicore__ inline void FlushScaleBatch(int64_t tileBegin, int64_t lastLocalTile, int64_t groupsPerTile,
                                           int64_t lastTileIdx, int64_t tailGroups);

private:
protected:
    const DynamicHiF4QuantTilingData* __restrict tilingData_{nullptr};
    TPipe* pipe_{nullptr};
    int64_t blockIdx_{0};
    // 由 scale_alg 决定的 Eb/Ec 比较阈值
    // scale_alg=0(默认): Eb阈值4, Ec阈值2; scale_alg=1: Eb阈值3.75, Ec阈值1.875
    bfloat16_t e8Thresh_ = static_cast<bfloat16_t>(4.0F);
    bfloat16_t e16Thresh_ = static_cast<bfloat16_t>(2.0F);

    TQue<QuePosition::VECIN, INPUT_QUE_DEPTH> inQueueX_;
    TQue<QuePosition::VECOUT, OUT_QUE_DEPTH> outQueue_;
    TQue<QuePosition::VECOUT, SCALE_QUE_DEPTH> outScaleQueue_;

    TBuf<TPosition::VECCALC> bufE8_;
    TBuf<TPosition::VECCALC> buf1E8_;
    TBuf<TPosition::VECCALC> buf1E16_;
    TBuf<TPosition::VECCALC> bufTmpScale_;
    TBuf<TPosition::VECCALC> bufMaxPer8_;
    TBuf<TPosition::VECCALC> bufMax64Raw_; // 压缩后的 64 组最大值(bf16), ComputeEa 的输入
    TBuf<TPosition::VECCALC> bufEaRec_;    // Ea 倒数(bf16)暂存, E2B_B16 装载即广播

    GlobalTensor<ST> xGm_;
    GlobalTensor<uint8_t> yGm_;
    GlobalTensor<float> scaleGm_;

    DataCopyExtParams dataCopyInParams_;
    DataCopyPadExtParams<ST> padParams_;
    DataCopyExtParams dataCopyOutParams_;
    DataCopyExtParams dataCopyOutScaleParams_;

    int64_t sysRegLen_ = 256;
    int64_t gmCopyInRegStep_ = 0;
    int64_t gmCopyOutRegStep_ = 0;
    int64_t gmCopyOutScaleRegStep_ = 0;
    int64_t gmCopyInRegStep2_ = 0;
    int64_t gmCopyTmpE1_ = 0;
    int64_t gmCopyTmpE8_ = 0;
    int64_t gmCopyTmpE16_ = 0;
    int64_t gmCopyTmpBufE8Step_ = 0;
    int64_t gmCopyTmpBufOtherStep_ = 0;

    int64_t bufE8Buffer_ = 0;
    int64_t buf1E8Buffer_ = 0;
    int64_t buf1E16Buffer_ = 0;
    int64_t maxPer8Buffer_ = 0;
    int64_t tmpScaleBuffer_ = 0;
    int64_t max64RawBuffer_ = 0;
    int64_t eaRecBuffer_ = 0;

    // 各 VF 阶段每迭代处理的 group 数(与 host 侧对齐约束联动)，满块/尾块循环数在 Process 内派生
    static constexpr int64_t HIFP4_OUTPUT_REG_ONELOOP_GROUP_NUM = 8;         // ComputeMaxPer4/ComputeTmpScale
    static constexpr int64_t HIFP4_OUTPUT_SCALE_REG_ONELOOP_GROUP_NUM = 256; // ComputeEa/ComputeScale
    static constexpr int64_t HIFP4_MAXPER8_REG_ONELOOP_GROUP_NUM = 16;       // ComputeMaxPer8
    static constexpr int64_t HIFP4_OUT_REG_ONELOOP_GROUP_NUM = 2;            // ComputeOut(每 128 元素寄存器)
    int64_t blockSize_ = 0;
    constexpr static Reg::CastTrait castTrait_ = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                  Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    constexpr static Reg::CastTrait castTraitReciprocalCast = {Reg::RegLayout::ZERO, Reg::SatMode::SAT,
                                                               Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
};

template <typename ST, typename DT, uint32_t useQueNum>
__aicore__ inline void DynamicHiF4QuantTailAxis<ST, DT, useQueNum>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR scale, const DynamicHiF4QuantTilingData* __restrict tilingData)
{
    tilingData_ = tilingData;
    blockIdx_ = GetBlockIdx();
    blockSize_ = BLOCK_SIZE;

    // GM 按全局 tile 序号寻址
    xGm_.SetGlobalBuffer((__gm__ ST*)x);
    pipe_->InitBuffer(inQueueX_, INPUT_QUE_DEPTH, tilingData_->alignGroupNum * sizeof(ST) * DQ_HIFP4_GROUP_ELE_NUM);

    yGm_.SetGlobalBuffer((__gm__ uint8_t*)y);
    pipe_->InitBuffer(outQueue_, OUT_QUE_DEPTH, tilingData_->alignGroupNum * DQ_HIFP4_GROUP_BYTE_NUM);

    scaleGm_.SetGlobalBuffer((__gm__ float*)scale);
    // scale 银行每 tile 一个槽位，按 256 对齐保证 ComputeScale 末轮全寄存器写跨度覆盖
    int64_t alignTo256UbFormer = Ops::Base::CeilAlign(static_cast<int64_t>(tilingData_->alignGroupNum),
                                                      static_cast<int64_t>(DQ_HIFP4_SCALE_ALIGN));
    pipe_->InitBuffer(outScaleQueue_, SCALE_QUE_DEPTH, alignTo256UbFormer * sizeof(float));

    // e8/e1p8/e1p16 按 256 对齐，保证 256-group 粒度全寄存器读写跨度覆盖
    bufE8Buffer_ = Ops::Base::CeilAlign(static_cast<int64_t>(tilingData_->alignGroupNum),
                                        static_cast<int64_t>(DQ_HIFP4_SCALETMP_ALIGN));
    buf1E8Buffer_ = Ops::Base::CeilAlign(static_cast<int64_t>(tilingData_->alignGroupNum),
                                         static_cast<int64_t>(DQ_HIFP4_SCALETMP_ALIGN));
    buf1E16Buffer_ = Ops::Base::CeilAlign(static_cast<int64_t>(tilingData_->alignGroupNum),
                                          static_cast<int64_t>(DQ_HIFP4_SCALETMP_ALIGN));
    maxPer8Buffer_ = Ops::Base::CeilAlign(static_cast<int64_t>(tilingData_->alignGroupNum),
                                          static_cast<int64_t>(DQ_HIFP4_MAXPER8_ALIGN));
    tmpScaleBuffer_ = Ops::Base::CeilAlign(static_cast<int64_t>(tilingData_->alignGroupNum),
                                           static_cast<int64_t>(DQ_HIFP4_SCALETMP_ALIGN));
    max64RawBuffer_ = Ops::Base::CeilAlign(static_cast<int64_t>(tilingData_->alignGroupNum),
                                           static_cast<int64_t>(DQ_HIFP4_SCALETMP_ALIGN));
    eaRecBuffer_ = Ops::Base::CeilAlign(static_cast<int64_t>(tilingData_->alignGroupNum),
                                        static_cast<int64_t>(DQ_HIFP4_SCALETMP_ALIGN));

    pipe_->InitBuffer(bufE8_, bufE8Buffer_ * sizeof(int16_t));
    pipe_->InitBuffer(buf1E8_, buf1E8Buffer_ * sizeof(int32_t));
    pipe_->InitBuffer(buf1E16_, buf1E16Buffer_ * sizeof(int16_t));
    pipe_->InitBuffer(bufMaxPer8_, maxPer8Buffer_ * 8 * sizeof(int16_t));
    pipe_->InitBuffer(bufTmpScale_, tmpScaleBuffer_ * 32 * sizeof(int16_t));
    pipe_->InitBuffer(bufMax64Raw_, max64RawBuffer_ * sizeof(uint16_t));
    pipe_->InitBuffer(bufEaRec_, eaRecBuffer_ * sizeof(uint16_t));

    dataCopyInParams_.blockCount = 1;
    dataCopyInParams_.blockLen = 0;
    dataCopyInParams_.srcStride = 0;
    dataCopyInParams_.dstStride = 0;

    dataCopyOutParams_.blockCount = 1;
    dataCopyOutParams_.blockLen = 0;
    dataCopyOutParams_.srcStride = 0;
    dataCopyOutParams_.dstStride = 0;

    dataCopyOutScaleParams_.blockCount = 1;
    dataCopyOutScaleParams_.blockLen = 0;
    dataCopyOutScaleParams_.srcStride = 0;
    dataCopyOutScaleParams_.dstStride = 0;

    padParams_.isPad = false;
    padParams_.leftPadding = 0;
    padParams_.rightPadding = 0;

    gmCopyInRegStep_ = sysRegLen_ / sizeof(float) * 2;
    gmCopyOutRegStep_ = sysRegLen_ / sizeof(int32_t) / DQ_HIFP4_CAST_COPYOUT_PACK4_NUM;
    gmCopyOutScaleRegStep_ = sysRegLen_ / sizeof(float);
    gmCopyInRegStep2_ = sysRegLen_ / sizeof(ST);
    gmCopyTmpE1_ = sysRegLen_ * DIGIT_FOUR / blockSize_ / DIGIT_TWO;
    gmCopyTmpE8_ = sysRegLen_ * DIGIT_FOUR / DIGIT_EIGHT;
    gmCopyTmpE16_ = sysRegLen_ * DIGIT_FOUR / DIGIT_FOUR;
    gmCopyTmpBufE8Step_ = sysRegLen_ * DIGIT_TWO;
    gmCopyTmpBufOtherStep_ = sysRegLen_;

    // 根据 scale_alg 选择 Eb/Ec 比较阈值
    e8Thresh_ = (tilingData_->scaleAlg == 1) ? static_cast<bfloat16_t>(3.75F) : static_cast<bfloat16_t>(4.0F);
    e16Thresh_ = (tilingData_->scaleAlg == 1) ? static_cast<bfloat16_t>(1.875F) : static_cast<bfloat16_t>(2.0F);
}

template <typename ST, typename DT, uint32_t useQueNum>
__aicore__ inline void DynamicHiF4QuantTailAxis<ST, DT, useQueNum>::CopyIn(const int64_t& gmOffset)
{
    auto xLocalIn = inQueueX_.template AllocTensor<ST>();
    DataCopyPad(xLocalIn, xGm_[gmOffset], dataCopyInParams_, padParams_, true);
    inQueueX_.EnQue(xLocalIn);
}

template <typename ST, typename DT, uint32_t useQueNum>
__aicore__ inline void DynamicHiF4QuantTailAxis<ST, DT, useQueNum>::ComputeAll(int64_t regLoop, int64_t maxPer8RegLoop,
                                                                               int64_t eaRegLoop, int64_t scaleRegLoop,
                                                                               int64_t outRegLoop,
                                                                               __ubuf__ uint16_t* scaleSlotAddr)
{
    LocalTensor<ST> x = inQueueX_.template DeQue<ST>();
    LocalTensor<uint8_t> y = outQueue_.template AllocTensor<uint8_t>();

    LocalTensor<uint16_t> e8buf = bufE8_.Get<uint16_t>();
    LocalTensor<uint8_t> e1p8buf = buf1E8_.Get<uint8_t>();
    LocalTensor<uint8_t> e1p16buf = buf1E16_.Get<uint8_t>();
    LocalTensor<uint16_t> tmpscale = bufTmpScale_.Get<uint16_t>();
    LocalTensor<uint16_t> maxper8 = bufMaxPer8_.Get<uint16_t>();
    LocalTensor<uint16_t> max64raw = bufMax64Raw_.Get<uint16_t>();
    LocalTensor<uint16_t> earec = bufEaRec_.Get<uint16_t>();

    auto xAddr1 = (__ubuf__ float*)x.GetPhyAddr();
    auto xAddr2 = (__ubuf__ ST*)x.GetPhyAddr();
    auto yAddr = (__ubuf__ int32_t*)y.GetPhyAddr();
    auto scaleAddr = scaleSlotAddr;

    auto e8BufAddr = (__ubuf__ uint16_t*)e8buf.GetPhyAddr();
    auto e8BufAddr2 = (__ubuf__ uint8_t*)e8buf.GetPhyAddr();
    auto e1p8BufAddr = (__ubuf__ uint32_t*)e1p8buf.GetPhyAddr();
    auto e1p16BufAddr = (__ubuf__ uint8_t*)e1p16buf.GetPhyAddr();
    auto tmpScaleAddr = (__ubuf__ uint16_t*)tmpscale.GetPhyAddr();
    // maxper4 工作区与 y 输出槽共用：生命期止于 VF 链前段，ComputeOut 覆写为量化结果
    auto maxPer4BufAddr = (__ubuf__ uint16_t*)y.GetPhyAddr();
    auto maxPer8BufAddr = (__ubuf__ uint16_t*)maxper8.GetPhyAddr();
    auto max64RawBufAddr = (__ubuf__ uint16_t*)max64raw.GetPhyAddr();
    auto eaRecBufAddr = (__ubuf__ uint16_t*)earec.GetPhyAddr();

    auto e8BufRead2Addr = e8BufAddr2;
    auto e1p8BufReadAddr = (__ubuf__ uint8_t*)e1p8buf.GetPhyAddr();
    auto e1p16BufReadAddr = e1p16BufAddr;
    auto tmpScaleReadAddr = tmpScaleAddr;
    auto maxPer4BufReadAddr = maxPer4BufAddr;
    auto maxPer4BufRead2Addr = maxPer4BufAddr;
    auto maxPer8BufReadAddr = maxPer8BufAddr;
    auto max64RawReadAddr = max64RawBufAddr;

    // 6 个相位共用同一 __VEC_SCOPE__（相位间 UB 读写依赖由向量核按序执行保证）
    {
        ComputeMaxPer4(static_cast<uint16_t>(regLoop), xAddr1, maxPer4BufAddr);
        ComputeMaxPer8(static_cast<uint16_t>(maxPer8RegLoop), maxPer4BufReadAddr, max64RawBufAddr, maxPer8BufAddr);
        ComputeEa(static_cast<uint16_t>(eaRegLoop), max64RawReadAddr, e8BufAddr, eaRecBufAddr);
        ComputeTmpScale(static_cast<uint16_t>(regLoop), maxPer4BufRead2Addr, maxPer8BufReadAddr, e8BufAddr, e1p8BufAddr,
                        e1p16BufAddr, tmpScaleAddr);
        ComputeScale(static_cast<uint16_t>(scaleRegLoop), e8BufRead2Addr, e1p8BufReadAddr, e1p16BufReadAddr, scaleAddr);
        ComputeOut(static_cast<uint16_t>(outRegLoop), xAddr2, tmpScaleReadAddr, yAddr);
    }

    outQueue_.template EnQue(y);
    inQueueX_.template FreeTensor(x);
}

template <typename ST, typename DT, uint32_t useQueNum>
__aicore__ inline void DynamicHiF4QuantTailAxis<ST, DT, useQueNum>::ComputeMaxPer4(uint16_t regLoop,
                                                                                   __ubuf__ float* xAddr1,
                                                                                   __ubuf__ uint16_t* maxPer4BufAddr)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<ST> vregIn1, vregIn2, vregIn3, vregIn4, vregMax1, vregMax2, vregOut;
        Reg::RegTensor<bfloat16_t> vregOutbf16;
        Reg::RegTensor<uint16_t> expMaskBF16;
        Reg::MaskReg maskAll = Reg::CreateMask<uint8_t>();
        Reg::Duplicate(expMaskBF16, 0x7fff);
        static constexpr Reg::CastTrait castTraitHalf2Bf16 = {Reg::RegLayout::UNKNOWN, Reg::SatMode::UNKNOWN,
                                                              Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        // 2 条双目载入各走独立指针（相差 1 记录，步长 2 记录）
        __ubuf__ float* xStream0 = xAddr1;
        __ubuf__ float* xStream1 = xAddr1 + gmCopyInRegStep_;
        for (uint16_t loopIdx = 0; loopIdx < regLoop; loopIdx++) {
            Reg::DataCopy<float, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B32>(
                (Reg::RegTensor<float>&)vregIn1, (Reg::RegTensor<float>&)vregIn2, xStream0,
                gmCopyInRegStep_ * DIGIT_TWO);
            Reg::And((Reg::RegTensor<uint16_t>&)vregIn1, (Reg::RegTensor<uint16_t>&)vregIn1, expMaskBF16, maskAll);
            Reg::And((Reg::RegTensor<uint16_t>&)vregIn2, (Reg::RegTensor<uint16_t>&)vregIn2, expMaskBF16, maskAll);
            Reg::DataCopy<float, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B32>(
                (Reg::RegTensor<float>&)vregIn3, (Reg::RegTensor<float>&)vregIn4, xStream1,
                gmCopyInRegStep_ * DIGIT_TWO);
            Reg::And((Reg::RegTensor<uint16_t>&)vregIn3, (Reg::RegTensor<uint16_t>&)vregIn3, expMaskBF16, maskAll);
            Reg::And((Reg::RegTensor<uint16_t>&)vregIn4, (Reg::RegTensor<uint16_t>&)vregIn4, expMaskBF16, maskAll);
            Reg::Max(vregMax1, vregIn1, vregIn2, maskAll);
            Reg::Max(vregMax2, vregIn3, vregIn4, maskAll);
            Reg::DeInterleave(vregMax1, vregMax2, vregMax1, vregMax2);
            Reg::Max(vregOut, vregMax1, vregMax2, maskAll);
            if constexpr (IsSameType<ST, half>::value) {
                Reg::Cast<bfloat16_t, ST, castTraitHalf2Bf16>(vregOutbf16, vregOut, maskAll);
                Reg::DataCopy<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(
                    maxPer4BufAddr, (Reg::RegTensor<uint16_t>&)vregOutbf16, gmCopyInRegStep2_, maskAll);
            } else {
                Reg::DataCopy<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(
                    maxPer4BufAddr, (Reg::RegTensor<uint16_t>&)vregOut, gmCopyInRegStep2_, maskAll);
            }
        }
    }
    return;
}

template <typename ST, typename DT, uint32_t useQueNum>
__aicore__ inline void DynamicHiF4QuantTailAxis<ST, DT, useQueNum>::ComputeMaxPer8(
    uint16_t maxPer8RegLoop, __ubuf__ uint16_t* maxPer4BufReadAddr, __ubuf__ uint16_t* max64RawBufAddr,
    __ubuf__ uint16_t* maxPer8BufAddr)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<bfloat16_t> vreg0Maxper4, vreg1Maxper4, vreg0Maxper8, vreg0Maxper64, vreg1Maxper64;
        Reg::UnalignReg ureg1;
        Reg::MaskReg maskAll = Reg::CreateMask<uint8_t>();
        // 本循环只算 max8/max64；max64 由 ReduceMaxWithDataBlock 压缩暂存(每寄存器 8 有效值)，
        // Ea 的 Mul(1/7)+Cast(E6M2) 移到压缩数据上的 ComputeEa 执行
        for (uint16_t loopIdx = 0; loopIdx < maxPer8RegLoop; loopIdx++) {
            Reg::DataCopy<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(
                (Reg::RegTensor<uint16_t>&)vreg0Maxper4, (Reg::RegTensor<uint16_t>&)vreg1Maxper4, maxPer4BufReadAddr,
                gmCopyInRegStep2_ * DIGIT_TWO);
            Reg::Max(vreg0Maxper8, vreg0Maxper4, vreg1Maxper4, maskAll);
            Reg::Interleave(vreg0Maxper4, vreg1Maxper4, vreg0Maxper4, vreg1Maxper4);

            Reg::ReduceMaxWithDataBlock<uint16_t>((Reg::RegTensor<uint16_t>&)vreg0Maxper64,
                                                  (Reg::RegTensor<uint16_t>&)vreg0Maxper4, maskAll);
            Reg::ReduceMaxWithDataBlock<uint16_t>((Reg::RegTensor<uint16_t>&)vreg1Maxper64,
                                                  (Reg::RegTensor<uint16_t>&)vreg1Maxper4, maskAll);

            Reg::DataCopy<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(
                maxPer8BufAddr, (Reg::RegTensor<uint16_t>&)vreg0Maxper8, gmCopyInRegStep2_, maskAll);
            Reg::DataCopyUnAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(
                max64RawBufAddr, (Reg::RegTensor<uint16_t>&)vreg0Maxper64, ureg1, 8);
            Reg::DataCopyUnAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(
                max64RawBufAddr, (Reg::RegTensor<uint16_t>&)vreg1Maxper64, ureg1, 8);
        }
        Reg::DataCopyUnAlignPost(max64RawBufAddr, ureg1, 0);
    }
    return;
}

template <typename ST, typename DT, uint32_t useQueNum>
__aicore__ inline void DynamicHiF4QuantTailAxis<ST, DT, useQueNum>::ComputeEa(uint16_t eaRegLoop,
                                                                              __ubuf__ uint16_t* max64RawReadAddr,
                                                                              __ubuf__ uint16_t* e8BufAddr,
                                                                              __ubuf__ uint16_t* eaRecBufAddr)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<bfloat16_t> vreg0Max64, vreg1Max64;
        Reg::RegTensor<uint16_t> vreg0E6m2, vreg1E6m2, vregBf16E6M2ScaleBias;
        Reg::MaskReg maskAll = Reg::CreateMask<uint8_t>();
        Reg::Duplicate(vregBf16E6M2ScaleBias, 0x3e12); // ≈1/7 (Ea = Vmax/7 后取 E6M2)
        static constexpr Reg::CastTrait castTraitE6M2 = {Reg::RegLayout::ZERO, Reg::SatMode::SAT,
                                                         Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        // 在压缩后的 max64 上换算 Ea，fp8 码字写入 e8 缓冲；
        // ReciprocalCast(E6M2→bf16) 移至 ComputeTmpScale，经 e8 缓冲内存往返物化后执行
        __ubuf__ uint16_t* m64Stream0 = max64RawReadAddr;
        __ubuf__ uint16_t* m64Stream1 = max64RawReadAddr + gmCopyInRegStep2_;
        __ubuf__ uint16_t* e8Stream0 = e8BufAddr;
        __ubuf__ uint16_t* e8Stream1 = e8BufAddr + gmCopyInRegStep2_;
        for (uint16_t loopIdx = 0; loopIdx < eaRegLoop; loopIdx++) {
            Reg::DataCopy<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_NORM>(
                (Reg::RegTensor<uint16_t>&)vreg0Max64, m64Stream0, gmCopyInRegStep2_ * DIGIT_TWO);
            Reg::DataCopy<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_NORM>(
                (Reg::RegTensor<uint16_t>&)vreg1Max64, m64Stream1, gmCopyInRegStep2_ * DIGIT_TWO);
            Reg::Mul(vreg0Max64, vreg0Max64, (Reg::RegTensor<bfloat16_t>&)vregBf16E6M2ScaleBias, maskAll);
            Reg::Mul(vreg1Max64, vreg1Max64, (Reg::RegTensor<bfloat16_t>&)vregBf16E6M2ScaleBias, maskAll);
            Reg::Cast<fp8_e6m2_t, bfloat16_t, castTraitE6M2>((Reg::RegTensor<fp8_e6m2_t>&)vreg0E6m2, vreg0Max64,
                                                             maskAll); // 只有前8位才是有效值
            Reg::Cast<fp8_e6m2_t, bfloat16_t, castTraitE6M2>((Reg::RegTensor<fp8_e6m2_t>&)vreg1E6m2, vreg1Max64,
                                                             maskAll); // 只有前8位才是有效值
            Reg::DataCopy<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(e8Stream0, (Reg::RegTensor<uint16_t>&)vreg0E6m2,
                                                                        gmCopyInRegStep2_ * DIGIT_TWO, maskAll);
            Reg::DataCopy<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(e8Stream1, (Reg::RegTensor<uint16_t>&)vreg1E6m2,
                                                                        gmCopyInRegStep2_ * DIGIT_TWO, maskAll);
        }
    }
    return;
}

template <typename ST, typename DT, uint32_t useQueNum>
__aicore__ inline void DynamicHiF4QuantTailAxis<ST, DT, useQueNum>::ComputeTmpScale(
    uint16_t regLoop, __ubuf__ uint16_t* maxPer4BufRead2Addr, __ubuf__ uint16_t* maxPer8BufReadAddr,
    __ubuf__ uint16_t* e8BufRead1Addr, __ubuf__ uint32_t* e1p8BufAddr, __ubuf__ uint8_t* e1p16BufAddr,
    __ubuf__ uint16_t* tmpScaleAddr)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<bfloat16_t> vreg0Maxper4, vreg0Maxper8, vreg0Maxper64, vb16_EaRecBf16, vregTmpScale;
        Reg::RegTensor<bfloat16_t> vregNum1, vregNumhalf;
        Reg::MaskReg vb1E8Comp, vb1E8CompTmp, vb1E16Comp, maskAll = Reg::CreateMask<uint8_t>();
        Reg::UnalignReg ureg1;

        Reg::Duplicate(vregNum1, 1.0F);
        Reg::Duplicate(vregNumhalf, static_cast<bfloat16_t>(0.5F));
        // Ea 的 fp8 码字经 E2B_B16 从 e8 缓冲读回(每 8 group 广播)，内存往返物化量化后再取倒数
        for (uint16_t loopIdx = 0; loopIdx < regLoop; loopIdx++) {
            Reg::DataCopy<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_E2B_B16>(
                (Reg::RegTensor<uint16_t>&)vreg0Maxper64, e8BufRead1Addr, 8);
            Reg::DataCopy<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_US_B16>(
                (Reg::RegTensor<uint16_t>&)vreg0Maxper8, maxPer8BufReadAddr, 64);
            Reg::DataCopy<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_NORM>(
                (Reg::RegTensor<uint16_t>&)vreg0Maxper4, maxPer4BufRead2Addr, 128);

            Reg::ReciprocalCast<bfloat16_t, fp8_e6m2_t, castTraitReciprocalCast>(
                (Reg::RegTensor<bfloat16_t>&)vb16_EaRecBf16, (Reg::RegTensor<fp8_e6m2_t>&)vreg0Maxper64, maskAll);
            Reg::Mul(vreg0Maxper8, vreg0Maxper8, vb16_EaRecBf16, maskAll);
            Reg::CompareScalar<bfloat16_t, CMPMODE::GE>(vb1E8Comp, vreg0Maxper8, e8Thresh_, maskAll);
            Reg::Select(vreg0Maxper8, vregNumhalf, vregNum1, vb1E8Comp);
            Reg::Mul(vreg0Maxper4, vreg0Maxper4, vb16_EaRecBf16, maskAll);
            Reg::Mul(vreg0Maxper4, vreg0Maxper4, vreg0Maxper8, maskAll);
            Reg::CompareScalar<bfloat16_t, CMPMODE::GE>(vb1E16Comp, vreg0Maxper4, e16Thresh_, maskAll);
            Reg::Select(vreg0Maxper4, vregNumhalf, vregNum1, vb1E16Comp);
            Reg::Mul(vreg0Maxper4, vreg0Maxper4, vreg0Maxper8, maskAll);
            Reg::Mul(vregTmpScale, vreg0Maxper4, vb16_EaRecBf16, maskAll);

            Reg::DataCopy<uint16_t, Reg::StoreDist::DIST_INTLV_B16>(tmpScaleAddr + loopIdx * gmCopyTmpBufOtherStep_,
                                                                    (Reg::RegTensor<uint16_t>&)vregTmpScale,
                                                                    (Reg::RegTensor<uint16_t>&)vregTmpScale, maskAll);

            Reg::DataCopyUnAlign<uint32_t>(e1p8BufAddr, vb1E8Comp, ureg1);

            Reg::DataCopy<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::MaskDist::DIST_PACK>(e1p16BufAddr,
                                                                                                 vb1E16Comp, 16);
        }
        Reg::DataCopyUnAlignPost<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE>(e1p8BufAddr, ureg1, 0);
    }
    return;
}

template <typename ST, typename DT, uint32_t useQueNum>
__aicore__ inline void DynamicHiF4QuantTailAxis<ST, DT, useQueNum>::ComputeScale(uint16_t scaleRegLoop,
                                                                                 __ubuf__ uint8_t* e8BufRead2Addr,
                                                                                 __ubuf__ uint8_t* e1p8BufReadAddr,
                                                                                 __ubuf__ uint8_t* e1p16BufReadAddr,
                                                                                 __ubuf__ uint16_t* scaleAddr)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<uint8_t> vreg0E8, vreg1E8, vreg1E16First, vreg1E16Second;
        Reg::MaskReg maskAll = Reg::CreateMask<uint8_t>();
        // e1p16 的 2 条载入各走独立指针（相差 1 块，步长 2 块）
        __ubuf__ uint8_t* e16Stream0 = e1p16BufReadAddr;
        __ubuf__ uint8_t* e16Stream1 = e1p16BufReadAddr + gmCopyTmpBufOtherStep_;
        for (uint16_t loopIdx = 0; loopIdx < scaleRegLoop; loopIdx++) {
            Reg::DataCopy<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DS_B8>(
                vreg0E8, e8BufRead2Addr, gmCopyTmpBufE8Step_);
            Reg::DataCopy<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_NORM>(
                vreg1E8, e1p8BufReadAddr, gmCopyTmpBufOtherStep_);
            Reg::DataCopy<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_NORM>(
                vreg1E16First, e16Stream0, gmCopyTmpBufOtherStep_ * DIGIT_TWO);
            Reg::DataCopy<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_NORM>(
                vreg1E16Second, e16Stream1, gmCopyTmpBufOtherStep_ * DIGIT_TWO);

            Reg::Interleave<uint8_t>(vreg0E8, vreg1E8, vreg0E8, vreg1E8);
            Reg::DataCopy<uint16_t, Reg::StoreDist::DIST_INTLV_B16>(scaleAddr + loopIdx * gmCopyTmpBufE8Step_,
                                                                    (Reg::RegTensor<uint16_t>&)vreg0E8,
                                                                    (Reg::RegTensor<uint16_t>&)vreg1E16First, maskAll);
            Reg::DataCopy<uint16_t, Reg::StoreDist::DIST_INTLV_B16>(
                scaleAddr + loopIdx * gmCopyTmpBufE8Step_ + gmCopyTmpBufOtherStep_, (Reg::RegTensor<uint16_t>&)vreg1E8,
                (Reg::RegTensor<uint16_t>&)vreg1E16Second, maskAll);
        }
    }
    return;
}

template <typename ST, typename DT, uint32_t useQueNum>
__aicore__ inline void DynamicHiF4QuantTailAxis<ST, DT, useQueNum>::ComputeOut(uint16_t outRegLoop, __ubuf__ ST* xAddr2,
                                                                               __ubuf__ uint16_t* tmpScaleReadAddr,
                                                                               __ubuf__ int32_t* yAddr)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<ST> vregOut0, vregOut1;
        Reg::RegTensor<bfloat16_t> vregOutbf16_0, vregOutbf16_1;
        Reg::RegTensor<fp4x2_e1m2_t> vbOut0, vbOut1;
        Reg::RegTensor<uint16_t> vregScaleTmp0, vregScaleTmp1;
        Reg::MaskReg maskAll = Reg::CreateMask<uint8_t>();
        static constexpr Reg::CastTrait castTraitHalf2Bf16 = {Reg::RegLayout::UNKNOWN, Reg::SatMode::UNKNOWN,
                                                              Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        // 每迭代 2 条独立 load->mul->cast->store 链，循环次数减半；奇数收尾复用 stream0
        __ubuf__ ST* xStream0 = xAddr2;
        __ubuf__ ST* xStream1 = xAddr2 + gmCopyInRegStep2_;
        __ubuf__ uint16_t* tsStream0 = tmpScaleReadAddr;
        __ubuf__ uint16_t* tsStream1 = tmpScaleReadAddr + gmCopyOutScaleRegStep_;
        __ubuf__ int32_t* yStream0 = yAddr;
        __ubuf__ int32_t* yStream1 = yAddr + gmCopyOutRegStep_;
        for (uint16_t loopIdx = 0; loopIdx < outRegLoop / DIGIT_TWO; loopIdx++) {
            // 先批量发 4 条载入，再各自计算/存储
            if constexpr (IsSameType<ST, half>::value) {
                Reg::DataCopy<ST, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_NORM>(
                    vregOut0, xStream0, gmCopyInRegStep2_ * DIGIT_TWO);
                Reg::DataCopy<ST, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_NORM>(
                    vregOut1, xStream1, gmCopyInRegStep2_ * DIGIT_TWO);
                Reg::Cast<bfloat16_t, ST, castTraitHalf2Bf16>((Reg::RegTensor<bfloat16_t>&)vregOutbf16_0, vregOut0,
                                                              maskAll);
                Reg::Cast<bfloat16_t, ST, castTraitHalf2Bf16>((Reg::RegTensor<bfloat16_t>&)vregOutbf16_1, vregOut1,
                                                              maskAll);
            } else {
                Reg::DataCopy<ST, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_NORM>(
                    vregOutbf16_0, xStream0, gmCopyInRegStep2_ * DIGIT_TWO);
                Reg::DataCopy<ST, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_NORM>(
                    vregOutbf16_1, xStream1, gmCopyInRegStep2_ * DIGIT_TWO);
            }
            Reg::DataCopy<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_US_B16>(
                vregScaleTmp0, tsStream0, gmCopyOutScaleRegStep_ * DIGIT_TWO);
            Reg::DataCopy<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_US_B16>(
                vregScaleTmp1, tsStream1, gmCopyOutScaleRegStep_ * DIGIT_TWO);
            Reg::Mul((Reg::RegTensor<bfloat16_t>&)vregOutbf16_0, (Reg::RegTensor<bfloat16_t>&)vregOutbf16_0,
                     (Reg::RegTensor<bfloat16_t>&)vregScaleTmp0, maskAll);
            Reg::Mul((Reg::RegTensor<bfloat16_t>&)vregOutbf16_1, (Reg::RegTensor<bfloat16_t>&)vregOutbf16_1,
                     (Reg::RegTensor<bfloat16_t>&)vregScaleTmp1, maskAll);
            Reg::Cast<fp4x2_e1m2_t, bfloat16_t, castTrait_>(vbOut0, (Reg::RegTensor<bfloat16_t>&)vregOutbf16_0,
                                                            maskAll);
            Reg::Cast<fp4x2_e1m2_t, bfloat16_t, castTrait_>(vbOut1, (Reg::RegTensor<bfloat16_t>&)vregOutbf16_1,
                                                            maskAll);
            Reg::DataCopy<int32_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_PACK4_B32>(
                yStream0, (Reg::RegTensor<int32_t>&)vbOut0, gmCopyOutRegStep_ * DIGIT_TWO, maskAll);
            Reg::DataCopy<int32_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_PACK4_B32>(
                yStream1, (Reg::RegTensor<int32_t>&)vbOut1, gmCopyOutRegStep_ * DIGIT_TWO, maskAll);
        }
        // 奇数收尾：复用 stream0
        if (outRegLoop % DIGIT_TWO == 1) {
            if constexpr (IsSameType<ST, half>::value) {
                Reg::DataCopy<ST, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_NORM>(
                    vregOut0, xStream0, gmCopyInRegStep2_ * DIGIT_TWO);
                Reg::Cast<bfloat16_t, ST, castTraitHalf2Bf16>((Reg::RegTensor<bfloat16_t>&)vregOutbf16_0, vregOut0,
                                                              maskAll);
            } else {
                Reg::DataCopy<ST, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_NORM>(
                    vregOutbf16_0, xStream0, gmCopyInRegStep2_ * DIGIT_TWO);
            }
            Reg::DataCopy<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_US_B16>(
                vregScaleTmp0, tsStream0, gmCopyOutScaleRegStep_ * DIGIT_TWO);
            Reg::Mul((Reg::RegTensor<bfloat16_t>&)vregOutbf16_0, (Reg::RegTensor<bfloat16_t>&)vregOutbf16_0,
                     (Reg::RegTensor<bfloat16_t>&)vregScaleTmp0, maskAll);
            Reg::Cast<fp4x2_e1m2_t, bfloat16_t, castTrait_>(vbOut0, (Reg::RegTensor<bfloat16_t>&)vregOutbf16_0,
                                                            maskAll);
            Reg::DataCopy<int32_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_PACK4_B32>(
                yStream0, (Reg::RegTensor<int32_t>&)vbOut0, gmCopyOutRegStep_ * DIGIT_TWO, maskAll);
        }
    }
    return;
}

template <typename ST, typename DT, uint32_t useQueNum>
__aicore__ inline void DynamicHiF4QuantTailAxis<ST, DT, useQueNum>::CopyOutY(const int64_t& gmOffset)
{
    auto yLocalOut = outQueue_.template DeQue<uint8_t>();
    DataCopyPad(yGm_[gmOffset], yLocalOut, dataCopyOutParams_);
    outQueue_.FreeTensor(yLocalOut);
}

template <typename ST, typename DT, uint32_t useQueNum>
__aicore__ inline void DynamicHiF4QuantTailAxis<ST, DT, useQueNum>::FlushScaleBatch(
    int64_t tileBegin, int64_t lastLocalTile, int64_t groupsPerTile, int64_t lastTileIdx, int64_t tailGroups)
{
    // 本批 tile 数(满批 SCALE_TILES_PER_BANK 个)与批内首 tile 的全局序号
    const int64_t batchTileNum = lastLocalTile % SCALE_TILES_PER_BANK + 1;
    const int64_t firstGlobalTile = tileBegin + lastLocalTile + 1 - batchTileNum;
    // 全局尾 tile 若落在本批：单独以短 blockLen 截断搬出；其余满 tile 合并为一条连续搬出
    const bool batchContainsTail = (firstGlobalTile + batchTileNum - 1 == lastTileIdx);
    const int64_t fullTiles = batchContainsTail ? batchTileNum - 1 : batchTileNum;
    // 批内槽地址连续；逐槽 DeQue/Free 仅走队列状态机，批搬数据在各自 EnQue 时已写完
    for (int64_t i = 0; i < batchTileNum; ++i) {
        auto scaleLocalOut = outScaleQueue_.template DeQue<float>();
        if (i == 0 && fullTiles > 0) {
            // 满块连续批搬：nBurst=1，lenBurst=满块数 x 2KB，src/dst 均紧邻无 stride
            dataCopyOutScaleParams_.blockCount = 1;
            dataCopyOutScaleParams_.blockLen = fullTiles * groupsPerTile * sizeof(float);
            dataCopyOutScaleParams_.srcStride = 0;
            dataCopyOutScaleParams_.dstStride = 0;
            DataCopyPad(scaleGm_[firstGlobalTile * groupsPerTile], scaleLocalOut, dataCopyOutScaleParams_);
        }
        if (batchContainsTail && i == fullTiles) {
            // 尾 tile 短块：blockLen 可非 32B 整数倍，截断不污染 GM
            dataCopyOutScaleParams_.blockCount = 1;
            dataCopyOutScaleParams_.blockLen = tailGroups * sizeof(float);
            dataCopyOutScaleParams_.srcStride = 0;
            dataCopyOutScaleParams_.dstStride = 0;
            DataCopyPad(scaleGm_[lastTileIdx * groupsPerTile], scaleLocalOut, dataCopyOutScaleParams_);
        }
        outScaleQueue_.FreeTensor(scaleLocalOut);
    }
}

template <typename ST, typename DT, uint32_t useQueNum>
__aicore__ inline void DynamicHiF4QuantTailAxis<ST, DT, useQueNum>::Process()
{
    if (blockIdx_ >= tilingData_->usedCoreNum) {
        return;
    }

    // 均分核：头/尾核连续分配，块起址连续衔接
    int64_t tileNum = 0;
    int64_t tileBegin = 0;
    if (blockIdx_ < tilingData_->headCoreNum) {
        tileNum = tilingData_->blockPerHeadCore;
        tileBegin = blockIdx_ * tileNum;
    } else {
        tileNum = tilingData_->blockPerTailCore;
        tileBegin = tilingData_->headCoreNum * tilingData_->blockPerHeadCore +
                    (blockIdx_ - tilingData_->headCoreNum) * tileNum;
    }
    // 定长 tile 按展平流切分，仅全局最后一个 tile 可能不足(tailGroups)
    const int64_t groupsPerTile = tilingData_->alignGroupNum;
    const int64_t lastTileIdx = (tilingData_->groupNum + groupsPerTile - 1) / groupsPerTile - 1;
    const int64_t tailGroups = tilingData_->groupNum - lastTileIdx * groupsPerTile;

    // 满块/尾块两套循环数一次性预计算（除数为编译期常数）
    const int64_t innerLoopFull = (groupsPerTile + HIFP4_OUTPUT_REG_ONELOOP_GROUP_NUM - 1) /
                                  HIFP4_OUTPUT_REG_ONELOOP_GROUP_NUM;
    const int64_t innerLoopTail = (tailGroups + HIFP4_OUTPUT_REG_ONELOOP_GROUP_NUM - 1) /
                                  HIFP4_OUTPUT_REG_ONELOOP_GROUP_NUM;
    const int64_t max8LoopFull = (groupsPerTile + HIFP4_MAXPER8_REG_ONELOOP_GROUP_NUM - 1) /
                                 HIFP4_MAXPER8_REG_ONELOOP_GROUP_NUM;
    const int64_t max8LoopTail = (tailGroups + HIFP4_MAXPER8_REG_ONELOOP_GROUP_NUM - 1) /
                                 HIFP4_MAXPER8_REG_ONELOOP_GROUP_NUM;
    // ComputeEa 与 ComputeScale 同为 256 group/迭代, 循环数一致
    const int64_t scaleLoopFull = (groupsPerTile + HIFP4_OUTPUT_SCALE_REG_ONELOOP_GROUP_NUM - 1) /
                                  HIFP4_OUTPUT_SCALE_REG_ONELOOP_GROUP_NUM;
    const int64_t scaleLoopTail = (tailGroups + HIFP4_OUTPUT_SCALE_REG_ONELOOP_GROUP_NUM - 1) /
                                  HIFP4_OUTPUT_SCALE_REG_ONELOOP_GROUP_NUM;
    const int64_t outLoopFull = (groupsPerTile + HIFP4_OUT_REG_ONELOOP_GROUP_NUM - 1) / HIFP4_OUT_REG_ONELOOP_GROUP_NUM;
    const int64_t outLoopTail = (tailGroups + HIFP4_OUT_REG_ONELOOP_GROUP_NUM - 1) / HIFP4_OUT_REG_ONELOOP_GROUP_NUM;

    for (int64_t t = 0; t < tileNum; ++t) {
        const int64_t globalTile = tileBegin + t;
        const bool isTailTile = (globalTile == lastTileIdx);
        const int64_t groupNum = isTailTile ? tailGroups : groupsPerTile;

        // GM 绝对偏移按全局 tile 序号计算
        const int64_t xGmOffset = globalTile * groupsPerTile * DQ_HIFP4_GROUP_ELE_NUM;
        const int64_t yGmOffset = globalTile * groupsPerTile * DQ_HIFP4_GROUP_BYTE_NUM;

        dataCopyInParams_.blockLen = groupNum * DQ_HIFP4_GROUP_ELE_NUM * sizeof(ST);
        dataCopyOutParams_.blockLen = groupNum * DQ_HIFP4_GROUP_BYTE_NUM;

        // CopyIn 先行，再 Alloc scale 银行槽
        CopyIn(xGmOffset);
        LocalTensor<float> scaleOut = outScaleQueue_.template AllocTensor<float>();
        ComputeAll(isTailTile ? innerLoopTail : innerLoopFull, isTailTile ? max8LoopTail : max8LoopFull,
                   isTailTile ? scaleLoopTail : scaleLoopFull, isTailTile ? scaleLoopTail : scaleLoopFull,
                   isTailTile ? outLoopTail : outLoopFull, (__ubuf__ uint16_t*)scaleOut.GetPhyAddr());
        outScaleQueue_.template EnQue(scaleOut);
        CopyOutY(yGmOffset);

        // scale 每满 4 个 tile 或本核末 tile，批量集中搬出
        const int64_t slotInBank = t % SCALE_TILES_PER_BANK;
        if (slotInBank == SCALE_TILES_PER_BANK - 1 || t == tileNum - 1) {
            FlushScaleBatch(tileBegin, t, groupsPerTile, lastTileIdx, tailGroups);
        }
    }
}

} // namespace DynamicHiF4Quant
#endif
