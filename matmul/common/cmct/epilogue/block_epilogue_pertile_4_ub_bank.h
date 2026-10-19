/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file block_epilogue_pertile_4_ub_bank.h
 * \brief DAV_9201/9202(960DT/960PR) 专用 perblock(pertile) epilogue：
 *        9201/9202 的 UB 为 4-bank 交错布局且 AIV:AIC 为 1:1，累加/搬出按 4-bank stride 组织，
 *        单 AIV 承接完整 baseM（与 DAV_3510 的 BlockEpiloguePertile 双子块切分不同）。
 *        算法主流程（per-tile/per-block 缩放、bias 搬入与累加、类型转换、CopyOut、事件同步）与
 *        block_epilogue_pertile.h 保持一致，布局差异集中在 UbBank4Layout 中维护，
 *        后续 9201/9202 与 3510 的公共主流程可基于该 layout 抽象做进一步合并。
 */

#pragma once
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "../utils/common_utils.h"
#include "../utils/layout_utils.h"
#include "../utils/quant_batch_matmul_constant.h"
#include "../utils/tensor_utils.h"
#include "block_epilogue_pertile.h"

namespace Cmct {
namespace Gemm {
namespace Block {

using namespace Cmct::Gemm::QuantBatchMatmul;

namespace {
// 与 common/inc/op_kernel/platform.h 的 platform::GetVRegSize 语义保持一致；
// cmct 公共头不依赖算子侧 include 路径，此处内联实现。
__aicore__ inline uint32_t GetVRegSizeLocal()
{
#if __CCE_AICORE__ == 310
    return AscendC::VECTOR_REG_WIDTH;
#else
    return 256U;
#endif
}
} // namespace

/*!
 * \brief DAV_9201/9202 UB 4-bank 交错布局参数。
 *        UB 由 4 个 256B sub-bank 交错组成（UB_BANK0~3_START_ADRR），相邻 bank 组间隔 1KB：
 *        - l0c 输入区（mmRes）：相邻两行间隔 1 个 bank 组（1KB）；
 *        - 累加区（mmAdd）：同一行第 2 个 ND 与第 1 个 ND 间隔 1 个 bank 组（1KB），
 *          相邻两行间隔 2 个 bank 组（2KB，读写 stride 拉开以避免 bank 冲突）。
 *        所有访问 mmRes/mmAdd 的代码统一从该 layout 取 stride，不再散落魔法数字。
 */
struct UbBank4Layout {
    static constexpr uint32_t bankNum = 4U;                         // 4: 4 个 256B sub-bank
    static constexpr uint32_t bankGroupBytes = UB_FOUR_BANK_STRIDE; // 1KB: 一个 4-bank 组
    // 同一行内第 2 个 ND 相对第 1 个 ND 的元素偏移（1KB / sizeof(T)）
    // 注：须标注 __aicore__ 方可在 device 函数中调用（同 math_utils.h 的 CeilDiv 等工具函数）
    template <class T>
    __aicore__ inline constexpr static uint32_t NdStrideElems()
    {
        return bankGroupBytes / static_cast<uint32_t>(sizeof(T));
    }
    // 累加区相邻两行的元素偏移（2KB / sizeof(T)）
    template <class T>
    __aicore__ inline constexpr static uint32_t RowStrideElems()
    {
        return NdStrideElems<T>() * 2U;
    }
};

struct PerBlockUBParam4UbBank {
    bool CopyOutWithSplitN = false;
    uint16_t ndNum;
    uint64_t singleM;
    uint64_t singleN;
    uint64_t validM;
    uint32_t validN;
    uint64_t offsetScaleM;
    uint64_t offsetScaleN;
    uint64_t offsetY;
    uint64_t offsetBias;
};

QBMM_BLOCK_EPILOGUE_PERTILE_CLASS_LOCAL_PARAMS
class BlockEpiloguePertile4UbBank {
public:
    __aicore__ inline BlockEpiloguePertile4UbBank()
    {
        if ASCEND_IS_AIV {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(1);
        }
    }

    __aicore__ inline ~BlockEpiloguePertile4UbBank()
    {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(1);
        if ASCEND_IS_AIV {
            if (isBias_) {
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(biasPingPongID_);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(biasPingPongID_ ^ 1);
            }
        }
    }

    struct Arguments {
        GM_ADDR outGmAddr{nullptr};
        GM_ADDR x2ScaleGmAddr{nullptr};
        GM_ADDR x1ScaleGmAddr{nullptr};
        GM_ADDR biasGmAddr{nullptr};
        uint32_t baseM;
        uint32_t baseN;
        uint32_t baseK;
        uint32_t groupSizeM = 1U;
        uint32_t groupSizeN = 128U;
        uint32_t groupSizeK = 128U;
        uint32_t isBias = 0;
        Arguments() = default;
    };

    // params
    using Params = Arguments;

    using YType = DataTypeOut_;
    using CType = DataTypeIn_;
    using BiasType = DataTypeBias_;
    using X2ScaleType = DataTypeX2Scale_;
    using X1ScaleType = DataTypeX1Scale_;
    using LayoutX1Scale = LayoutX1Scale_;
    using LayoutX2Scale = LayoutX2Scale_;

    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t>; // m,n,k
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t>;

    static constexpr bool transA = TagToTrans<LayoutX1Scale>::value;
    static constexpr bool transB = TagToTrans<LayoutX2Scale>::value;

public:
    __aicore__ inline void Init(const Params* params);
    __aicore__ inline void operator()(const TupleShape& actualSingleShape, const BlockCoord& blockCoord);
    __aicore__ inline void UpdateGlobalAddr(const BlockCoord& baseOffset);
    __aicore__ inline void UpdateParamsForNextProblem(const TupleShape& problemShape);
    __aicore__ inline auto GetL0c2UbPingTensor();
    __aicore__ inline auto GetL0c2UbPongTensor();

private:
    __aicore__ inline void ProcessAivSingleKPerTile(int64_t x1ScaleOffset, __gm__ X2ScaleType* x2ScaleAddr);

    __aicore__ inline void ProcessAivSingleKPerBlock(int64_t x1ScaleOffset, __gm__ X2ScaleType* x2ScaleAddr);
    // 按 cube 侧 ping-pong id 选择本次 K 循环的累加缓冲（cur/initial）与 L0C→UB 输入缓冲
    __aicore__ inline void GetMmAccBuffers(__ubuf__ CType*& mmAccCur, __ubuf__ CType*& mmAccInitial,
                                           __ubuf__ CType*& mmRes);
    __aicore__ inline void CopyInBias();
    template <class T>
    __aicore__ inline __ubuf__ T* CopyInX1Scale(uint64_t srcOffset, uint64_t m, uint64_t k);
    template <class T>
    __aicore__ inline T CopyInX1ScalePerblock(__gm__ T* src, uint64_t offset);
    template <class T>
    __aicore__ inline T CopyInX2Scale(__gm__ T* src, uint64_t offset);
    template <class T>
    __aicore__ inline __ubuf__ T* GetX1ScaleUbAddrPerGroup(int64_t x1ScaleOffset, uint64_t kOffset, uint64_t kElem);
    template <bool isFirstKLoop>
    __aicore__ inline void AivPerTensorPertile(__ubuf__ X1ScaleType* x1ScaleUbAddr, X2ScaleType x2Scale, uint64_t kElem,
                                               uint64_t x1ScaleKRem);
    template <bool isFirstKLoop>
    __aicore__ inline void AivPerTensorPerblock(X1ScaleType scaleMul);
    template <bool isFirstKLoop, uint32_t ndNum>
    __aicore__ inline void AivPerTensor(__ubuf__ CType* accCur, __ubuf__ CType* accIntial, __ubuf__ CType* l0cOut,
                                        float scaleScalar, uint16_t mSize, uint16_t nSize, uint32_t nSrcUbAligned);
    template <bool isFirstKLoop, uint32_t ndNum>
    __aicore__ inline void AivPerTensor(__ubuf__ CType* accRes, __ubuf__ CType* accIntial, __ubuf__ CType* l0cOut,
                                        __ubuf__ X1ScaleType* x1scale, X2ScaleType x2scale, uint16_t mSize,
                                        uint16_t nSize, uint32_t nSrcUbAligned, uint64_t kSize,
                                        uint64_t x1ScaleKIdxInCache);
    template <uint32_t ndNum>
    __aicore__ inline void AddBias(__ubuf__ CType* mmAdd, __ubuf__ BiasType* bias, uint16_t mSize);
    __aicore__ inline void AivPostProcess(const AscendC::LocalTensor<CType>& mmAddUb);
    __aicore__ inline void CopyB32Out(const AscendC::LocalTensor<CType>& mmAddUb);
    template <uint32_t ndNum>
    __aicore__ inline void CastToB16AndCopyOut(const AscendC::LocalTensor<CType>& mmAddUb);
    __aicore__ inline void CopyOut(const AscendC::LocalTensor<YType>& ubRes, uint16_t blkCount, uint32_t blkLen,
                                   uint32_t srcStride, uint32_t dstStride, uint64_t yOffset);
    __aicore__ inline void UpdatePerBlockUBValidMN();
    __aicore__ inline void UpdatePerBlockUBParam();
    __aicore__ inline void WaitForCube(uint16_t crossPingPongID)
    {
        AscendC::CrossCoreWaitFlag<QBMM_AIC_SYNC_AIV_MODE, PIPE_V>(QBMM_AIV_SYNC_AIC_FLAG + crossPingPongID);
    }
    __aicore__ inline void NotifyCube(uint16_t crossPingPongID)
    {
        AscendC::CrossCoreSetFlag<QBMM_AIC_SYNC_AIV_MODE, PIPE_V>(QBMM_AIC_SYNC_AIV_FLAG + crossPingPongID);
    }
    __aicore__ inline void NotifyV2Mte3(uint16_t id)
    {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIdV2Mte3[id]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIdV2Mte3[id]);
    }

    AscendC::GlobalTensor<YType> cGlobal_;
    AscendC::GlobalTensor<X1ScaleType> x1ScaleGlobal_;
    __gm__ X1ScaleType* x1ScaleGlobalPerblock_;
    __gm__ X2ScaleType* x2ScaleGlobal_;
    AscendC::GlobalTensor<BiasType> biasGlobal_;
    AscendC::LocalTensor<CType> mmResPing_;
    AscendC::LocalTensor<CType> mmResPong_;
    AscendC::LocalTensor<YType> ubResPing_;
    AscendC::LocalTensor<YType> ubResPong_;
    AscendC::LocalTensor<CType> mmAddUbPing_;
    AscendC::LocalTensor<CType> mmAddUbPong_;
    AscendC::LocalTensor<X1ScaleType> x1ScaleUbPing_;
    AscendC::LocalTensor<X1ScaleType> x1ScaleUbPong_;
    AscendC::LocalTensor<BiasType> biasUbPing_;
    AscendC::LocalTensor<BiasType> biasUbPong_;

private:
    const Params* params_;
    PerBlockUBParam4UbBank ubParams_;
    TupleShape problemShape_{};
    TupleShape actualSingleShape_{};
    BlockCoord baseOffset_{0, 0, 0, 0};
    BlockCoord blockCoord_{0, 0, 0, 0};

    uint64_t scaleM_ = 0;
    uint64_t scaleN_ = 0;
    uint64_t scaleK_ = 0;
    uint32_t subBlockIdx_;
    uint16_t crossPingPongID_ = 0;
    uint16_t x1ScalePingPongID_ = 0;
    // bias 双缓冲事件 id 借用 2/3（0/1 已被 x1Scale ping-pong 使用），初始值 2 表示首块用 Ping
    uint16_t biasPingPongID_ = 2;
    uint64_t kCopySizeX1Scale_ = 0;
    bool isPertile_ = false;
    bool isBias_ = false;
    uint64_t fp32NumPerVfLen_ = GetVRegSizeLocal() / sizeof(CType);

    event_t eventIdV2Mte2[2] = {EVENT_ID0, EVENT_ID1};
    event_t eventIdV2Mte3[2] = {EVENT_ID0, EVENT_ID1};
    event_t eventIdMTE32V_ = EVENT_ID0;
};

QBMM_BLOCK_EPILOGUE_PERTILE_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpiloguePertile4UbBank<QBMM_BLOCK_EPILOGUE_PERTILE_FUNC_LOCAL_PARAMS>::Init(
    const Params* params)
{
    if ASCEND_IS_AIC {
        return;
    }

    params_ = params;
    subBlockIdx_ = AscendC::GetSubBlockIdx();
    (void)GetL0c2UbPingTensor();
    (void)GetL0c2UbPongTensor();
    uint32_t elems = params_->baseM * params_->baseN;
    mmAddUbPing_ = AscendC::LocalTensor<CType>(AscendC::TPosition::VECCALC, UB_BANK2_START_ADRR,
                                               elems * UB_SUB_BANK_NUM - UB_BANK2_START_ADRR / sizeof(CType));
    mmAddUbPong_ = AscendC::LocalTensor<CType>(AscendC::TPosition::VECCALC, UB_BANK1_START_ADRR,
                                               elems * UB_SUB_BANK_NUM - UB_BANK1_START_ADRR / sizeof(CType));

    kCopySizeX1Scale_ = 64; // 64: 9201/9202 UB 较大，一次搬64个基本块对应的x1scale 的数据量
    uint32_t afterAddOffset = elems * UbBank4Layout::bankNum * sizeof(CType) + UB_BANK3_START_ADRR;
    if constexpr (!AscendC::IsSameType<CType, YType>::value) {
        ubResPing_ = AscendC::LocalTensor<YType>(AscendC::TPosition::VECCALC, afterAddOffset, elems);
        ubResPong_ = ubResPing_[elems / QBMM_BUFFER_NUM];
    }

    isPertile_ = params_->groupSizeM == 1;
    uint32_t x1ScaleUbOffset = (AscendC::IsSameType<CType, YType>::value) ? afterAddOffset :
                                                                            afterAddOffset + elems * sizeof(YType);
    if (isPertile_) {
        uint32_t x1ScaleElems = params_->baseM * kCopySizeX1Scale_;
        x1ScaleUbPing_ = AscendC::LocalTensor<X1ScaleType>(AscendC::TPosition::VECCALC, x1ScaleUbOffset, x1ScaleElems);
        x1ScaleUbPong_ = AscendC::LocalTensor<X1ScaleType>(
            AscendC::TPosition::VECCALC, x1ScaleUbOffset + x1ScaleElems * sizeof(X1ScaleType), x1ScaleElems);
        // bias 双缓冲放在 x1Scale 双缓冲之后，与 BlockEpiloguePertile 保持一致：
        // 仅 pertile(G-B, groupSizeM==1) 分支支持 bias，perblock(B-B) 分支不支持；
        // 带 bias 的 G-B/B-B 场景由 host 侧 QuantBatchMatmulV3Checker::CheckInputValidInPerblockMode
        // 统一拦截（"when the quantization mode is G-B or B-B, bias must be null"），kernel 侧不会收到
        // perblock + bias 的组合，此处保留与 3510 相同的防御性逻辑。
        if (params_->isBias == 1) {
            isBias_ = true;
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(biasPingPongID_);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(biasPingPongID_ ^ 1);
            uint32_t afterX1ScaleUbOffset = x1ScaleUbOffset +
                                            2 * (params_->baseM * kCopySizeX1Scale_) * sizeof(X1ScaleType);
            biasUbPing_ = AscendC::LocalTensor<BiasType>(AscendC::TPosition::VECCALC, afterX1ScaleUbOffset,
                                                         PER_BLOCK_SIZE);
            biasUbPong_ = AscendC::LocalTensor<BiasType>(
                AscendC::TPosition::VECCALC, afterX1ScaleUbOffset + PER_BLOCK_SIZE * sizeof(BiasType), PER_BLOCK_SIZE);
        }
    }
}

QBMM_BLOCK_EPILOGUE_PERTILE_CLASS_LOCAL_PARAMS
__aicore__ inline auto BlockEpiloguePertile4UbBank<QBMM_BLOCK_EPILOGUE_PERTILE_FUNC_LOCAL_PARAMS>::GetL0c2UbPingTensor()
{
    uint32_t elems = params_->baseM * params_->baseN;
    mmResPing_ = AscendC::LocalTensor<CType>(AscendC::TPosition::VECCALC, 0, elems * UB_SUB_BANK_NUM);
    return mmResPing_;
}

QBMM_BLOCK_EPILOGUE_PERTILE_CLASS_LOCAL_PARAMS
__aicore__ inline auto BlockEpiloguePertile4UbBank<QBMM_BLOCK_EPILOGUE_PERTILE_FUNC_LOCAL_PARAMS>::GetL0c2UbPongTensor()
{
    uint32_t elems = params_->baseM * params_->baseN;
    mmResPong_ = AscendC::LocalTensor<CType>(
        AscendC::TPosition::VECCALC, UB_BANK3_START_ADRR,
        elems * UB_SUB_BANK_NUM - UB_BANK3_START_ADRR / sizeof(CType)); // 64 stride for fp32 DB
    return mmResPong_;
}

QBMM_BLOCK_EPILOGUE_PERTILE_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpiloguePertile4UbBank<
    QBMM_BLOCK_EPILOGUE_PERTILE_FUNC_LOCAL_PARAMS>::UpdateParamsForNextProblem(const TupleShape& problemShape)
{
    problemShape_ = problemShape;

    scaleM_ = Cmct::Gemm::CeilDiv(Get<MNK_M>(problemShape_), params_->groupSizeM);
    scaleN_ = Cmct::Gemm::CeilDiv(Get<MNK_N>(problemShape_), params_->groupSizeN);
    scaleK_ = Cmct::Gemm::CeilDiv(Get<MNK_K>(problemShape_), params_->groupSizeK);
}

QBMM_BLOCK_EPILOGUE_PERTILE_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpiloguePertile4UbBank<QBMM_BLOCK_EPILOGUE_PERTILE_FUNC_LOCAL_PARAMS>::UpdateGlobalAddr(
    const BlockCoord& baseOffset)
{
    if ASCEND_IS_AIV {
        x1ScaleGlobal_.SetGlobalBuffer((__gm__ X1ScaleType*)params_->x1ScaleGmAddr + Get<X1SCALE_IDX>(baseOffset));
        x1ScaleGlobalPerblock_ = (__gm__ X1ScaleType*)params_->x1ScaleGmAddr + Get<X1SCALE_IDX>(baseOffset);
        x2ScaleGlobal_ = (__gm__ X2ScaleType*)params_->x2ScaleGmAddr + Get<X2SCALE_IDX>(baseOffset);
        cGlobal_.SetGlobalBuffer((__gm__ YType*)params_->outGmAddr + Get<Y_IDX>(baseOffset));
        if (isBias_) {
            biasGlobal_.SetGlobalBuffer((__gm__ BiasType*)params_->biasGmAddr + Get<BIAS_IDX>(baseOffset));
        }
    }
}

QBMM_BLOCK_EPILOGUE_PERTILE_CLASS_LOCAL_PARAMS
template <class T>
__aicore__ inline __ubuf__ T* BlockEpiloguePertile4UbBank<QBMM_BLOCK_EPILOGUE_PERTILE_FUNC_LOCAL_PARAMS>::CopyInX1Scale(
    uint64_t srcOffset, uint64_t m, uint64_t k)
{
    AscendC::DataCopyExtParams x1ScaleGm2UbParams;
    AscendC::DataCopyPadExtParams<X1ScaleType> padParams;
    if constexpr (transA) {
        x1ScaleGm2UbParams.blockCount = k;
        x1ScaleGm2UbParams.blockLen = m * sizeof(T);
        x1ScaleGm2UbParams.srcStride = (scaleM_ - m) * sizeof(T);
    } else {
        x1ScaleGm2UbParams.blockCount = m;
        x1ScaleGm2UbParams.blockLen = k * sizeof(T);
        x1ScaleGm2UbParams.srcStride = (scaleK_ - k) * sizeof(T);
    }
    auto x1ScaleUb = x1ScalePingPongID_ == 0 ? &x1ScaleUbPing_ : &x1ScaleUbPong_;
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(x1ScalePingPongID_);
    AscendC::DataCopyPad(*x1ScaleUb, x1ScaleGlobal_[srcOffset], x1ScaleGm2UbParams, padParams);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(x1ScalePingPongID_);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(x1ScalePingPongID_);
    return reinterpret_cast<__ubuf__ T*>(x1ScaleUb->GetPhyAddr());
}

QBMM_BLOCK_EPILOGUE_PERTILE_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpiloguePertile4UbBank<QBMM_BLOCK_EPILOGUE_PERTILE_FUNC_LOCAL_PARAMS>::CopyInBias()
{
    AscendC::DataCopyExtParams biasGm2UbParams;
    AscendC::DataCopyPadExtParams<BiasType> padParams;
    // bias 按 VF 段（fp32NumPerVfLen_ 个/段）搬入 UB，段内不足部分 pad 0，与 AddBias 的分段一致
    uint32_t validN0 = Min(ubParams_.validN, static_cast<uint32_t>(fp32NumPerVfLen_));
    biasGm2UbParams.blockCount = 1;
    biasGm2UbParams.blockLen = validN0 * sizeof(BiasType);
    biasGm2UbParams.srcStride = 0;
    biasGm2UbParams.dstStride = 0;
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(biasPingPongID_);
    auto biasUb = biasPingPongID_ == 2 ? biasUbPing_ : biasUbPong_;
    AscendC::DataCopyPad(biasUb, biasGlobal_[ubParams_.offsetBias], biasGm2UbParams, padParams);
    if (ubParams_.validN > fp32NumPerVfLen_) {
        biasGm2UbParams.blockLen = (ubParams_.validN - fp32NumPerVfLen_) * sizeof(BiasType);
        AscendC::DataCopyPad(biasUb[fp32NumPerVfLen_], biasGlobal_[ubParams_.offsetBias + fp32NumPerVfLen_],
                             biasGm2UbParams, padParams);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(biasPingPongID_);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(biasPingPongID_);
}

QBMM_BLOCK_EPILOGUE_PERTILE_CLASS_LOCAL_PARAMS
template <class T>
__aicore__ inline T BlockEpiloguePertile4UbBank<QBMM_BLOCK_EPILOGUE_PERTILE_FUNC_LOCAL_PARAMS>::CopyInX1ScalePerblock(
    __gm__ T* src, uint64_t offset)
{
    if constexpr (transA) {
        return src[offset * scaleM_];
    } else {
        return src[offset];
    }
}

QBMM_BLOCK_EPILOGUE_PERTILE_CLASS_LOCAL_PARAMS
template <class T>
__aicore__ inline T BlockEpiloguePertile4UbBank<QBMM_BLOCK_EPILOGUE_PERTILE_FUNC_LOCAL_PARAMS>::CopyInX2Scale(
    __gm__ T* src, uint64_t offset)
{
    if constexpr (transB) {
        return src[offset];
    } else {
        return src[offset * scaleN_];
    }
}

QBMM_BLOCK_EPILOGUE_PERTILE_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpiloguePertile4UbBank<QBMM_BLOCK_EPILOGUE_PERTILE_FUNC_LOCAL_PARAMS>::UpdatePerBlockUBValidMN()
{
    int64_t actualN = Get<MNK_N>(actualSingleShape_);
    ubParams_.validM = ubParams_.singleM;
    ubParams_.validN = Min(ubParams_.singleN, static_cast<uint64_t>(actualN));
    ubParams_.ndNum = CeilDiv(ubParams_.validN, fp32NumPerVfLen_);
}

QBMM_BLOCK_EPILOGUE_PERTILE_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpiloguePertile4UbBank<QBMM_BLOCK_EPILOGUE_PERTILE_FUNC_LOCAL_PARAMS>::UpdatePerBlockUBParam()
{
    ubParams_.singleN = Align(Get<MNK_N>(actualSingleShape_), static_cast<uint64_t>(AscendC::BLOCK_CUBE));
    ubParams_.singleM = CeilDiv(Get<MNK_M>(actualSingleShape_), AscendC::GetTaskRation());
    UpdatePerBlockUBValidMN();
    ubParams_.offsetY = Get<Y_IDX>(blockCoord_);
    ubParams_.offsetBias = Get<BIAS_IDX>(blockCoord_);
}

QBMM_BLOCK_EPILOGUE_PERTILE_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpiloguePertile4UbBank<QBMM_BLOCK_EPILOGUE_PERTILE_FUNC_LOCAL_PARAMS>::operator()(
    const TupleShape& actualSingleShape, const BlockCoord& blockCoord)
{
    actualSingleShape_ = actualSingleShape;
    blockCoord_ = blockCoord;
    UpdatePerBlockUBParam();
    int64_t x1ScaleOffset = Get<X1SCALE_IDX>(blockCoord_);
    int64_t x2ScaleOffset = Get<X2SCALE_IDX>(blockCoord_);
    __gm__ X2ScaleType* x2ScaleAddr = x2ScaleGlobal_ + x2ScaleOffset;
    if (isPertile_) {
        ProcessAivSingleKPerTile(x1ScaleOffset, x2ScaleAddr);
    } else {
        ProcessAivSingleKPerBlock(x1ScaleOffset, x2ScaleAddr);
    }
}

QBMM_BLOCK_EPILOGUE_PERTILE_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpiloguePertile4UbBank<QBMM_BLOCK_EPILOGUE_PERTILE_FUNC_LOCAL_PARAMS>::GetMmAccBuffers(
    __ubuf__ CType*& mmAccCur, __ubuf__ CType*& mmAccInitial, __ubuf__ CType*& mmRes)
{
    // cube 侧按 crossPingPongID_ 交替写入 mmResPing_/mmResPong_，AIV 侧同 id 的累加缓冲为 cur，对侧为 initial
    if (crossPingPongID_ == 0) {
        mmAccCur = reinterpret_cast<__ubuf__ CType*>(mmAddUbPing_.GetPhyAddr());
        mmAccInitial = reinterpret_cast<__ubuf__ CType*>(mmAddUbPong_.GetPhyAddr());
        mmRes = reinterpret_cast<__ubuf__ CType*>(mmResPing_.GetPhyAddr());
    } else {
        mmAccCur = reinterpret_cast<__ubuf__ CType*>(mmAddUbPong_.GetPhyAddr());
        mmAccInitial = reinterpret_cast<__ubuf__ CType*>(mmAddUbPing_.GetPhyAddr());
        mmRes = reinterpret_cast<__ubuf__ CType*>(mmResPong_.GetPhyAddr());
    }
}

QBMM_BLOCK_EPILOGUE_PERTILE_CLASS_LOCAL_PARAMS
template <class T>
__aicore__ inline __ubuf__ T*
BlockEpiloguePertile4UbBank<QBMM_BLOCK_EPILOGUE_PERTILE_FUNC_LOCAL_PARAMS>::GetX1ScaleUbAddrPerGroup(
    int64_t x1ScaleOffset, uint64_t kOffset, uint64_t kElem)
{
    uint64_t scaleX1GmOffset;
    if constexpr (transA) {
        scaleX1GmOffset = x1ScaleOffset + kOffset * scaleM_;
    } else {
        scaleX1GmOffset = x1ScaleOffset + kOffset;
    }
    return CopyInX1Scale<X1ScaleType>(scaleX1GmOffset, ubParams_.validM, kElem);
}

QBMM_BLOCK_EPILOGUE_PERTILE_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpiloguePertile4UbBank<QBMM_BLOCK_EPILOGUE_PERTILE_FUNC_LOCAL_PARAMS>::ProcessAivSingleKPerTile(
    int64_t x1ScaleOffset, __gm__ X2ScaleType* x2ScaleAddr)
{
    const uint16_t x1ScaleKElem = Min(kCopySizeX1Scale_, scaleK_);
    uint64_t kElem;
    __ubuf__ X1ScaleType* x1ScaleUbAddr;
    X2ScaleType x2Scale;
    for (uint64_t kb = 0, kOffset = 0; kb < Get<MNK_K>(problemShape_); kb += params_->baseK, kOffset++) {
        x2Scale = CopyInX2Scale<X2ScaleType>(x2ScaleAddr, kOffset);
        uint64_t x1ScaleKRem = kOffset % x1ScaleKElem;
        if (x1ScaleKRem == 0) {
            kElem = Min(static_cast<uint64_t>(x1ScaleKElem), scaleK_ - kOffset);
            x1ScaleUbAddr = GetX1ScaleUbAddrPerGroup<X1ScaleType>(x1ScaleOffset, kOffset, kElem);
        }
        WaitForCube(crossPingPongID_);
        if (kb == 0) {
            AivPerTensorPertile<true>(x1ScaleUbAddr, x2Scale, kElem, x1ScaleKRem);
        } else {
            AivPerTensorPertile<false>(x1ScaleUbAddr, x2Scale, kElem, x1ScaleKRem);
        }
        if (x1ScaleKRem == x1ScaleKElem - 1 || kOffset == scaleK_ - 1) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(x1ScalePingPongID_);
            x1ScalePingPongID_ = (x1ScalePingPongID_ + 1) & 1;
        }
        NotifyCube(crossPingPongID_);
        crossPingPongID_ = (crossPingPongID_ + 1) & 1;
    }
    if (isBias_) {
        CopyInBias();
    }
    AivPostProcess(crossPingPongID_ == 0 ? mmAddUbPong_ : mmAddUbPing_);
}

QBMM_BLOCK_EPILOGUE_PERTILE_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpiloguePertile4UbBank<QBMM_BLOCK_EPILOGUE_PERTILE_FUNC_LOCAL_PARAMS>::ProcessAivSingleKPerBlock(
    int64_t x1ScaleOffset, __gm__ X2ScaleType* x2ScaleAddr)
{
    auto x1ScaleAddr = x1ScaleGlobalPerblock_ + x1ScaleOffset;
    X2ScaleType x2Scale;
    for (uint64_t kb = 0, kOffset = 0; kb < Get<MNK_K>(problemShape_); kb += params_->baseK, kOffset++) {
        x2Scale = CopyInX2Scale<X2ScaleType>(x2ScaleAddr, kOffset);
        X1ScaleType x1Scale = CopyInX1ScalePerblock(x1ScaleAddr, kOffset);
        X1ScaleType scaleMul = x1Scale * x2Scale;
        WaitForCube(crossPingPongID_);
        AscendC::PipeBarrier<PIPE_V>();
        if (kb == 0) {
            AivPerTensorPerblock<true>(scaleMul);
        } else {
            AivPerTensorPerblock<false>(scaleMul);
        }
        NotifyCube(crossPingPongID_);
        crossPingPongID_ = (crossPingPongID_ + 1) & 1;
    }
    // 与 BlockEpiloguePertile 保持一致：perblock(B-B) 分支不支持 bias，
    // 带 bias 的场景已被 host 侧 CheckInputValidInPerblockMode 拦截，不会进入该分支
    AivPostProcess(crossPingPongID_ == 0 ? mmAddUbPong_ : mmAddUbPing_);
}

QBMM_BLOCK_EPILOGUE_PERTILE_CLASS_LOCAL_PARAMS
template <bool isFirstKLoop>
__aicore__ inline void BlockEpiloguePertile4UbBank<QBMM_BLOCK_EPILOGUE_PERTILE_FUNC_LOCAL_PARAMS>::AivPerTensorPertile(
    __ubuf__ X1ScaleType* x1ScaleUbAddr, X2ScaleType x2Scale, uint64_t kElem, uint64_t x1ScaleKRem)
{
    __ubuf__ CType* accCur;
    __ubuf__ CType* accInitial;
    __ubuf__ CType* mmRes;
    GetMmAccBuffers(accCur, accInitial, mmRes);
    if (ubParams_.ndNum == 1) {
        AivPerTensor<isFirstKLoop, 1U>(accCur, accInitial, mmRes, x1ScaleUbAddr, x2Scale, ubParams_.validM,
                                       ubParams_.validN, ubParams_.singleN, kElem, x1ScaleKRem);
    } else {
        AivPerTensor<isFirstKLoop, 2U>(accCur, accInitial, mmRes, x1ScaleUbAddr, x2Scale, ubParams_.validM,
                                       ubParams_.validN, ubParams_.singleN, kElem, x1ScaleKRem);
    }
}

QBMM_BLOCK_EPILOGUE_PERTILE_CLASS_LOCAL_PARAMS
template <bool isFirstKLoop>
__aicore__ inline void BlockEpiloguePertile4UbBank<QBMM_BLOCK_EPILOGUE_PERTILE_FUNC_LOCAL_PARAMS>::AivPerTensorPerblock(
    X1ScaleType scaleMul)
{
    __ubuf__ CType* accCur;
    __ubuf__ CType* accInitial;
    __ubuf__ CType* mmRes;
    GetMmAccBuffers(accCur, accInitial, mmRes);
    if (ubParams_.ndNum == 1) {
        AivPerTensor<isFirstKLoop, 1U>(accCur, accInitial, mmRes, scaleMul, ubParams_.validM, ubParams_.validN,
                                       ubParams_.singleN);
    } else {
        AivPerTensor<isFirstKLoop, 2U>(accCur, accInitial, mmRes, scaleMul, ubParams_.validM, ubParams_.validN,
                                       ubParams_.singleN);
    }
}

QBMM_BLOCK_EPILOGUE_PERTILE_CLASS_LOCAL_PARAMS
template <bool isFirstKLoop, uint32_t ndNum>
__aicore__ inline void BlockEpiloguePertile4UbBank<QBMM_BLOCK_EPILOGUE_PERTILE_FUNC_LOCAL_PARAMS>::AivPerTensor(
    __ubuf__ CType* accCur, __ubuf__ CType* accIntial, __ubuf__ CType* l0cOut, float scaleScalar, uint16_t mSize,
    uint16_t nSize, uint32_t nSrcUbAligned)
{
    (void)nSize;
    (void)nSrcUbAligned;
    // 4-bank 布局：l0c 输入区相邻行 1KB；累加区相邻行 2KB、同行第 2 个 ND 偏移 1KB（均以元素计）
    uint32_t l0cRowStride = UbBank4Layout::NdStrideElems<CType>();
    uint32_t accRowStride = UbBank4Layout::RowStrideElems<CType>();
    uint32_t accNdStride = UbBank4Layout::NdStrideElems<CType>();
    __VEC_SCOPE__
    {
        AscendC::Reg::MaskReg maskN = AscendC::Reg::CreateMask<CType, AscendC::Reg::MaskPattern::ALL>();

        // for m 循环内部的for n循环被展开，最多执行两次，这基于假设：tiling设置时，baseN最大为128.
        for (uint16_t mIdx = 0; mIdx < mSize; mIdx++) {
            AscendC::Reg::RegTensor<CType> l0cOutReg;
            AscendC::Reg::RegTensor<CType> accReg;

            // 从UB 读取数据的stride, 1KB
            AscendC::Reg::AddrReg mmResAddrReg = AscendC::Reg::CreateAddrReg<CType>(mIdx, l0cRowStride);
            AscendC::Reg::LoadAlign<CType, AscendC::Reg::LoadDist::DIST_NORM>(l0cOutReg, l0cOut, mmResAddrReg);
            AscendC::Reg::Muls(accReg, l0cOutReg, scaleScalar, maskN);

            // 2:将结果写到UB的stride为2KB
            AscendC::Reg::AddrReg accResAddrReg = AscendC::Reg::CreateAddrReg<CType>(mIdx, accRowStride);
            if constexpr (isFirstKLoop) {
                AscendC::Reg::StoreAlign<CType, AscendC::Reg::StoreDist::DIST_NORM_B32>(accCur, accReg, accResAddrReg,
                                                                                        maskN);
            } else {
                AscendC::Reg::RegTensor<CType> accInitReg;
                AscendC::Reg::LoadAlign<CType, AscendC::Reg::LoadDist::DIST_NORM>(accInitReg, accIntial, accResAddrReg);
                AscendC::Reg::Add(accReg, accInitReg, accReg, maskN);
                AscendC::Reg::StoreAlign<CType, AscendC::Reg::StoreDist::DIST_NORM_B32>(accCur, accReg, accResAddrReg,
                                                                                        maskN);
            }

            if constexpr (ndNum == 1) {
                continue;
            }

            AscendC::Reg::LoadAlign<CType, AscendC::Reg::LoadDist::DIST_NORM>(l0cOutReg, l0cOut + mSize * l0cRowStride,
                                                                              mmResAddrReg);
            AscendC::Reg::Muls(accReg, l0cOutReg, scaleScalar, maskN);
            if constexpr (isFirstKLoop) {
                AscendC::Reg::StoreAlign<CType, AscendC::Reg::StoreDist::DIST_NORM_B32>(accCur + accNdStride, accReg,
                                                                                        accResAddrReg, maskN);
            } else {
                AscendC::Reg::RegTensor<CType> accInitReg;
                AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(accInitReg, accIntial + accNdStride,
                                                                                  accResAddrReg);
                AscendC::Reg::Add(accReg, accInitReg, accReg, maskN);
                AscendC::Reg::StoreAlign<CType, AscendC::Reg::StoreDist::DIST_NORM_B32>(accCur + accNdStride, accReg,
                                                                                        accResAddrReg, maskN);
            }
        }
    }
}

QBMM_BLOCK_EPILOGUE_PERTILE_CLASS_LOCAL_PARAMS
template <bool isFirstKLoop, uint32_t ndNum>
__aicore__ inline void BlockEpiloguePertile4UbBank<QBMM_BLOCK_EPILOGUE_PERTILE_FUNC_LOCAL_PARAMS>::AivPerTensor(
    __ubuf__ CType* accRes, __ubuf__ CType* accIntial, __ubuf__ CType* l0cOut, __ubuf__ X1ScaleType* x1scale,
    X2ScaleType x2scale, uint16_t mSize, uint16_t nSize, uint32_t nSrcUbAligned, uint64_t kSize,
    uint64_t x1ScaleKIdxInCache)
{
    (void)nSize;
    (void)nSrcUbAligned;
    uint16_t alignX1scaleM = CeilAlign(mSize, UB_ALIGN_SIZE / sizeof(X1ScaleType));
    uint16_t alignX1scaleK = CeilAlign(kSize, UB_ALIGN_SIZE / sizeof(X1ScaleType));

    // 4-bank 布局：l0c 输入区相邻行 1KB；累加区相邻行 2KB、同行第 2 个 ND 偏移 1KB（均以元素计）
    uint32_t l0cRowStride = UbBank4Layout::NdStrideElems<CType>();
    uint32_t accRowStride = UbBank4Layout::RowStrideElems<CType>();
    uint32_t accNdStride = UbBank4Layout::NdStrideElems<CType>();
    __VEC_SCOPE__
    {
        AscendC::Reg::MaskReg maskN = AscendC::Reg::CreateMask<CType, AscendC::Reg::MaskPattern::ALL>();
        for (uint16_t mIdx = 0; mIdx < mSize; mIdx++) {
            AscendC::Reg::RegTensor<X1ScaleType> x1scaleReg, muledScaleReg;
            // 更新x1Scale
            if constexpr (transA) {
                AscendC::Reg::DataCopy<X1ScaleType, AscendC::Reg::LoadDist::DIST_BRC_B32>(
                    x1scaleReg, x1scale + alignX1scaleM * x1ScaleKIdxInCache + mIdx);
            } else {
                AscendC::Reg::DataCopy<X1ScaleType, AscendC::Reg::LoadDist::DIST_BRC_B32>(
                    x1scaleReg, x1scale + alignX1scaleK * mIdx + x1ScaleKIdxInCache);
            }
            AscendC::Reg::Muls(muledScaleReg, x1scaleReg, x2scale, maskN);

            AscendC::Reg::RegTensor<CType> l0cOutReg;
            AscendC::Reg::RegTensor<CType> accReg;

            // 加载从l0c 搬入UB 的数据到 RegTensor， 搬运的数据长度为VL， vector core一次可处理的数据长度，即256B；
            AscendC::Reg::AddrReg mmResAddrReg = AscendC::Reg::CreateAddrReg<CType>(mIdx, l0cRowStride);
            AscendC::Reg::AddrReg accResAddeReg = AscendC::Reg::CreateAddrReg<CType>(mIdx, accRowStride); // 每次间隔2KB

            AscendC::Reg::LoadAlign<CType, AscendC::Reg::LoadDist::DIST_NORM>(l0cOutReg, l0cOut, mmResAddrReg);

            if constexpr (isFirstKLoop) {
                AscendC::Reg::Mul(accReg, l0cOutReg, muledScaleReg, maskN);
                AscendC::Reg::StoreAlign<CType, AscendC::Reg::StoreDist::DIST_NORM_B32>(accRes, accReg, accResAddeReg,
                                                                                        maskN);
            } else {
                AscendC::Reg::LoadAlign<CType, AscendC::Reg::LoadDist::DIST_NORM>(accReg, accIntial, accResAddeReg);
                AscendC::Reg::MulAddDst(accReg, l0cOutReg, muledScaleReg, maskN);
                AscendC::Reg::StoreAlign<CType, AscendC::Reg::StoreDist::DIST_NORM_B32>(accRes, accReg, accResAddeReg,
                                                                                        maskN);
            }

            if constexpr (ndNum == 1) {
                continue;
            }

            AscendC::Reg::LoadAlign<CType, AscendC::Reg::LoadDist::DIST_NORM>(l0cOutReg, l0cOut + mSize * l0cRowStride,
                                                                              mmResAddrReg);
            if constexpr (isFirstKLoop) {
                AscendC::Reg::Mul(accReg, l0cOutReg, muledScaleReg, maskN);
                AscendC::Reg::StoreAlign<CType, AscendC::Reg::StoreDist::DIST_NORM_B32>(accRes + accNdStride, accReg,
                                                                                        accResAddeReg, maskN);
            } else {
                AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(accReg, accIntial + accNdStride,
                                                                                  accResAddeReg);
                AscendC::Reg::MulAddDst(accReg, l0cOutReg, muledScaleReg, maskN);
                AscendC::Reg::StoreAlign<CType, AscendC::Reg::StoreDist::DIST_NORM_B32>(accRes + accNdStride, accReg,
                                                                                        accResAddeReg, maskN);
            }
        }
    }
}

QBMM_BLOCK_EPILOGUE_PERTILE_CLASS_LOCAL_PARAMS
template <uint32_t ndNum>
__aicore__ inline void BlockEpiloguePertile4UbBank<QBMM_BLOCK_EPILOGUE_PERTILE_FUNC_LOCAL_PARAMS>::AddBias(
    __ubuf__ CType* mmAdd, __ubuf__ BiasType* bias, uint16_t mSize)
{
    AscendC::PipeBarrier<PIPE_V>();
    // 4-bank 布局：累加区相邻行 2KB、同行第 2 个 ND 偏移 1KB（均以元素计）
    uint32_t accNdStride = UbBank4Layout::NdStrideElems<CType>();
    uint32_t accRowStride = UbBank4Layout::RowStrideElems<CType>();
    uint32_t nSize0 = Min(ubParams_.validN, static_cast<uint32_t>(fp32NumPerVfLen_));
    uint32_t nSize1 = ubParams_.validN - nSize0;
    __VEC_SCOPE__
    {
        for (uint16_t mIdx = 0; mIdx < mSize; mIdx++) {
            AscendC::Reg::MaskReg maskN0 = AscendC::Reg::UpdateMask<CType>(nSize0);
            AscendC::Reg::RegTensor<BiasType> biasReg{};
            AscendC::Reg::RegTensor<CType> mmAddReg{};
            AscendC::Reg::AddrReg mmAddAddrReg = AscendC::Reg::CreateAddrReg<CType>(mIdx, accRowStride);

            AscendC::Reg::DataCopy(biasReg, bias);
            AscendC::Reg::LoadAlign<CType, AscendC::Reg::LoadDist::DIST_NORM>(mmAddReg, mmAdd, mmAddAddrReg);
            AscendC::Reg::Add(mmAddReg, mmAddReg, biasReg, maskN0);
            AscendC::Reg::StoreAlign<CType, AscendC::Reg::StoreDist::DIST_NORM_B32>(mmAdd, mmAddReg, mmAddAddrReg,
                                                                                    maskN0);
            if constexpr (ndNum == 1) {
                continue;
            }
            // 第 2 个 ND 段的 bias 位于第 1 段之后 fp32NumPerVfLen_ 个元素处（CopyInBias 搬入时对齐）
            AscendC::Reg::MaskReg maskN1 = AscendC::Reg::UpdateMask<CType>(nSize1);
            AscendC::Reg::DataCopy(biasReg, bias + fp32NumPerVfLen_);
            AscendC::Reg::LoadAlign<CType, AscendC::Reg::LoadDist::DIST_NORM>(mmAddReg, mmAdd + accNdStride,
                                                                              mmAddAddrReg);
            AscendC::Reg::Add(mmAddReg, mmAddReg, biasReg, maskN1);
            AscendC::Reg::StoreAlign<CType, AscendC::Reg::StoreDist::DIST_NORM_B32>(mmAdd + accNdStride, mmAddReg,
                                                                                    mmAddAddrReg, maskN1);
        }
    }
}

QBMM_BLOCK_EPILOGUE_PERTILE_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpiloguePertile4UbBank<QBMM_BLOCK_EPILOGUE_PERTILE_FUNC_LOCAL_PARAMS>::AivPostProcess(
    const AscendC::LocalTensor<CType>& mmAddUb)
{
    if (ubParams_.validM == 0 || ubParams_.validN == 0) {
        if (isBias_) {
            // CopyInBias 已占用当前 bias 缓冲，提前退出时同样需要释放并翻转
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(biasPingPongID_);
            biasPingPongID_ = biasPingPongID_ ^ 1;
        }
        return;
    }
    if constexpr (AscendC::IsSameType<CType, BiasType>::value) {
        if (isBias_) {
            auto biasUb = biasPingPongID_ == 2 ? biasUbPing_ : biasUbPong_;
            auto mmAddUbAddr = reinterpret_cast<__ubuf__ CType*>(mmAddUb.GetPhyAddr());
            auto biasUbAddr = reinterpret_cast<__ubuf__ BiasType*>(biasUb.GetPhyAddr());
            if (ubParams_.ndNum == 1) {
                AddBias<1U>(mmAddUbAddr, biasUbAddr, ubParams_.validM);
            } else {
                AddBias<2U>(mmAddUbAddr, biasUbAddr, ubParams_.validM);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(biasPingPongID_);
            biasPingPongID_ = biasPingPongID_ ^ 1;
        }
    }
    if constexpr (AscendC::IsSameType<YType, CType>::value) {
        NotifyV2Mte3(0);
        CopyB32Out(mmAddUb);
    } else {
        if (ubParams_.ndNum == 1) {
            CastToB16AndCopyOut<1>(mmAddUb);
        } else {
            CastToB16AndCopyOut<2>(mmAddUb); // 2: 每行若超64元素需要两次vf cast
        }
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventIdMTE32V_);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventIdMTE32V_);
}

QBMM_BLOCK_EPILOGUE_PERTILE_CLASS_LOCAL_PARAMS
template <uint32_t ndNum>
__aicore__ inline void BlockEpiloguePertile4UbBank<QBMM_BLOCK_EPILOGUE_PERTILE_FUNC_LOCAL_PARAMS>::CastToB16AndCopyOut(
    const AscendC::LocalTensor<CType>& mmAddUb)
{
    AscendC::PipeBarrier<PIPE_V>();
    // 4-bank 布局：累加区同行第 2 个 ND 偏移 1KB（以元素计）
    uint32_t accNdStride = UbBank4Layout::NdStrideElems<CType>();
    uint32_t nSize0 = Min(ubParams_.validN, static_cast<uint32_t>(fp32NumPerVfLen_));
    uint32_t nSize1 = ubParams_.validN - nSize0;

    // 与 BlockEpiloguePertile 的 CastAndCopyOut 一致，使用高层 Cast + UnaryRepeatParams；
    // 源（mmAdd 累加区）相邻行间隔 2 个 bank 组（2KB，即64个32B块），目标（ubRes）每行 128 个 YType（8 块）
    AscendC::UnaryRepeatParams castRepeatParam;
    castRepeatParam.srcBlkStride = 1;
    castRepeatParam.dstBlkStride = 1;
    castRepeatParam.srcRepStride = UbBank4Layout::RowStrideElems<CType>() * sizeof(CType) / AscendC::ONE_BLK_SIZE;
    castRepeatParam.dstRepStride = 128 * sizeof(YType) / AscendC::ONE_BLK_SIZE; // 128: ubRes 每行 YType 元素数

    if (nSize0 > 0) {
        AscendC::Cast(ubResPing_, mmAddUb, AscendC::RoundMode::CAST_RINT, nSize0,
                      static_cast<uint16_t>(ubParams_.validM), castRepeatParam);
    }
    if constexpr (ndNum == 2) {
        // 第 2 个 ND：源偏移 1 个 bank 组（1KB），目标偏移每行第 64 个 YType（128B）处
        AscendC::Cast(ubResPing_[64], mmAddUb[accNdStride], AscendC::RoundMode::CAST_RINT, nSize1,
                      static_cast<uint16_t>(ubParams_.validM), castRepeatParam);
    }

    NotifyV2Mte3(0);
    CopyOut(ubResPing_, ubParams_.validM, ubParams_.validN, 128 - ubParams_.validN, // 128 表示base_n的最大值
            Get<MNK_N>(problemShape_) - ubParams_.validN, ubParams_.offsetY);
}

QBMM_BLOCK_EPILOGUE_PERTILE_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpiloguePertile4UbBank<QBMM_BLOCK_EPILOGUE_PERTILE_FUNC_LOCAL_PARAMS>::CopyB32Out(
    const AscendC::LocalTensor<CType>& mmAddUb)
{
    uint64_t curNAlign = Align(ubParams_.validN, fp32NumPerVfLen_);

    // 累加结果相邻两行的stride是2个bank组（2KB）
    uint64_t accResStride = UbBank4Layout::RowStrideElems<CType>();

    if (CeilDiv(curNAlign, fp32NumPerVfLen_) == 1) {
        CopyOut(mmAddUb, ubParams_.validM, ubParams_.validN, accResStride - ubParams_.validN,
                Get<MNK_N>(problemShape_) - ubParams_.validN, ubParams_.offsetY);
    } else {
        uint32_t nSize0 = fp32NumPerVfLen_;
        uint32_t nSize1 = ubParams_.validN - nSize0;

        CopyOut(mmAddUb, ubParams_.validM, nSize0, accResStride - nSize0, Get<MNK_N>(problemShape_) - nSize0,
                ubParams_.offsetY);

        CopyOut(mmAddUb[nSize0 * sizeof(YType)], ubParams_.validM, nSize1, accResStride - nSize1,
                Get<MNK_N>(problemShape_) - nSize1, ubParams_.offsetY + nSize0);
    }
}

QBMM_BLOCK_EPILOGUE_PERTILE_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpiloguePertile4UbBank<QBMM_BLOCK_EPILOGUE_PERTILE_FUNC_LOCAL_PARAMS>::CopyOut(
    const AscendC::LocalTensor<YType>& ubRes, uint16_t blkCount, uint32_t blkLen, uint32_t srcStride,
    uint32_t dstStride, uint64_t yOffset)
{
    AscendC::DataCopyExtParams copyParams{blkCount, static_cast<uint32_t>(blkLen * sizeof(YType)),
                                          static_cast<uint32_t>(srcStride * sizeof(YType) / AscendC::ONE_BLK_SIZE),
                                          static_cast<uint32_t>(dstStride * sizeof(YType)), 0};
    AscendC::DataCopyPad<YType>(cGlobal_[yOffset], ubRes, copyParams);
}

} // namespace Block
} // namespace Gemm
} // namespace Cmct
