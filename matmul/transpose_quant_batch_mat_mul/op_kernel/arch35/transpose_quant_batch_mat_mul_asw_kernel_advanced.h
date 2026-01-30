/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file transpose_quant_batch_mat_mul_asw_kernel_advanced.h
 * \brief
 */
#ifndef TRANSPOSE_QUANT_BATCH_MAT_MUL_ASW_KERNEL_ADVANCED_H
#define TRANSPOSE_QUANT_BATCH_MAT_MUL_ASW_KERNEL_ADVANCED_H

#include "../../inc/platform.h"
#include "mm_extension_interface/tqbmm_custom_mm_policy.h"
#include "transpose_quant_batch_mat_mul_asw_block_advanced.h"
#include "basic_api/kernel_basic_intf.h"

using namespace Cmct::Gemm;

#define LOCAL_TEMPLATE_CLASS_MIX_PARAMS                                                                                \
    template <class aType, class bType, class scaleType, class biasType, class ptScaleType, class cType, bool aTrans,  \
              bool bTrans, class l0cDtype, class blockType, const MatmulConfig& mmCfg>
#define LOCAL_TEMPLATE_FUNC_MIX_PARAMS                                                                                 \
    aType, bType, scaleType, biasType, ptScaleType, cType, aTrans, bTrans, l0cDtype, blockType, mmCfg

namespace TransposeQuantBatchMatMulAdvanced {
using AscendC::AIC;
using AscendC::AIV;
using AscendC::DataCopyExtParams;
using AscendC::DataCopyPadParams;
using AscendC::DataCopyParams;
using AscendC::GlobalTensor;
using AscendC::IsSameType;
using AscendC::LocalTensor;
using AscendC::QuePosition;
using AscendC::TPipe;
using AscendC::TPosition;
using AscendC::TQue;

constexpr AscendC::MicroAPI::CastTrait ctInt322Fp32 = {
    AscendC::MicroAPI::RegLayout::UNKNOWN, AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

constexpr AscendC::MicroAPI::CastTrait ctFp322Half = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};

constexpr AscendC::MicroAPI::CastTrait ctHalf2Fp32Zero = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};

constexpr AscendC::MicroAPI::CastTrait ctHalf2Fp32One = {
    AscendC::MicroAPI::RegLayout::ONE, AscendC::MicroAPI::SatMode::UNKNOWN, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};

LOCAL_TEMPLATE_CLASS_MIX_PARAMS
class TransposeQuantBatchMatMulAswKernel {
public:
    __aicore__ inline TransposeQuantBatchMatMulAswKernel() {}
    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR scaleGM, GM_ADDR ptScaleGM, GM_ADDR cGM,
                                GM_ADDR workSpace, const void* tilingData, TPipe* pipe);
    __aicore__ inline void UpdateGlobalAddr(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR scaleGM, GM_ADDR ptScaleGM, GM_ADDR cGM,
                                            GM_ADDR workSpace);
    __aicore__ inline void Process();

public:
    using aT = AscendC::MatmulType<TPosition::GM, CubeFormat::ND, aType, aTrans>;
    using bT = AscendC::MatmulType<TPosition::GM, CubeFormat::ND, bType, bTrans>;
    using biasT = AscendC::MatmulType<TPosition::GM, CubeFormat::ND, biasType>;
    using cT = AscendC::MatmulType<TPosition::VECIN, CubeFormat::ND_ALIGN, l0cDtype>;
    AscendC::MatmulImpl<aT, bT, cT, biasT, mmCfg, AscendC::MatmulCallBackFunc<nullptr, nullptr, nullptr>,
                        AscendC::TQBmmCustomMatmulPolicy>
        mm;

    constexpr static uint32_t BUFFER_NUM = 2;
    constexpr static uint8_t AIC_SYNC_AIV_MODE = 4;
    constexpr static uint16_t FLAG_ID_MAX = 16;
    constexpr static uint16_t AIV_SYNC_AIC_FLAG = 6;
    constexpr static uint16_t AIC_SYNC_AIV_FLAG = 8;
    constexpr static uint8_t CV_RATIO = 2;
    constexpr static uint16_t DATA_BLOCK = 32;
    constexpr static uint32_t FP32_OUTPUT_TIMES = 4;

protected:
    __aicore__ inline void MMCompute();
    __aicore__ inline void SetOrgShape();
    __aicore__ inline void DequantCompute();
    __aicore__ inline void VFDoDequantWithX1Pertoken(__ubuf__ cType* dequantOutInUbAddr,
                                                     __ubuf__ l0cDtype* l0cOutUbAddr, uint64_t offsetPtScale,
                                                     uint16_t mSize);
    __aicore__ inline void VFDoDequant(__ubuf__ cType* dst, __ubuf__ l0cDtype* l0cOut, __ubuf__ scaleType* scale,
                                       __ubuf__ ptScaleType* perTokenScale, uint16_t mSize, uint16_t nSize);
    __aicore__ inline void NotifyCube()
    {
        AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE, PIPE_V>(AIV_SYNC_AIC_FLAG);
    }
    __aicore__ inline void WaitForVector()
    {
        AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE, PIPE_FIX>(AIV_SYNC_AIC_FLAG);
        AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE, PIPE_FIX>(AIV_SYNC_AIC_FLAG + FLAG_ID_MAX);
    }
    __aicore__ inline void NotifyVector()
    {
        AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE, PIPE_FIX>(AIC_SYNC_AIV_FLAG);
        AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE, PIPE_FIX>(AIC_SYNC_AIV_FLAG + FLAG_ID_MAX);
    }
    __aicore__ inline void WaitForCube()
    {
        AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE, PIPE_V>(AIC_SYNC_AIV_FLAG);
    }
    __aicore__ inline void CopyDataFromGm2Ub();
    __aicore__ inline void CopyX1ScaleFromGm2Ub(LocalTensor<ptScaleType>& dst, uint64_t blockLen, uint64_t offset);
    __aicore__ inline void CopyX2ScaleFromGm2Ub(LocalTensor<scaleType>& dst);
    __aicore__ inline void CopyDequantResFromUb2Gm(uint64_t blockCount, uint64_t offset, LocalTensor<cType>& src);
    __aicore__ inline void FreeUbTensor();

protected:
    uint32_t blockIdx_;
    uint32_t subBlockIdx_;
    blockType block_;
    TPipe* pipe_;
    GlobalTensor<aType> aGlobal_;
    GlobalTensor<bType> bGlobal_;
    GlobalTensor<cType> cGlobal_;
    GlobalTensor<scaleType> scaleGlobal_;
    GlobalTensor<ptScaleType> pertokenScaleGlobal_;
    LocalTensor<l0cDtype> l0cOutUb_;
    LocalTensor<scaleType> scaleUb_;
    LocalTensor<ptScaleType> ptScaleUb_;

    // define the que
    TQue<QuePosition::VECIN, 1> vecQueMMRes_;
    TQue<QuePosition::VECIN, 1> vecQueScale_;
    TQue<QuePosition::VECIN, 1> vecQuePertokenScale_;
    TQue<QuePosition::VECOUT, 1> vecQueOut_;

    const BatchMatMulV3TilingData* tilingData_;
};

LOCAL_TEMPLATE_CLASS_MIX_PARAMS
__aicore__ inline void TransposeQuantBatchMatMulAswKernel<LOCAL_TEMPLATE_FUNC_MIX_PARAMS>::Init(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR scaleGM, GM_ADDR ptScaleGM, GM_ADDR cGM, GM_ADDR workSpace, const void* tilingData,
    TPipe* pipe)
{
    blockIdx_ = AscendC::GetBlockIdx();
    if ASCEND_IS_AIV {
        blockIdx_ = blockIdx_ / AscendC::GetTaskRation();
        subBlockIdx_ = AscendC::GetSubBlockIdx();
    }
    pipe_ = pipe;
    tilingData_ = static_cast<const BatchMatMulV3TilingData*>(tilingData);

    UpdateGlobalAddr(aGM, bGM, scaleGM, ptScaleGM, cGM, workSpace);
    if ASCEND_IS_AIC {
        mm.Init(&tilingData_->matMulTilingData.tCubeTiling, pipe);
        SetOrgShape();
    }
    uint32_t mForSingleVec = CeilDiv(static_cast<uint64_t>(tilingData_->matMulTilingData.tCubeTiling.baseM), CV_RATIO);
    pipe_->InitBuffer(vecQueMMRes_, 1,
                      mForSingleVec * tilingData_->matMulTilingData.tCubeTiling.baseN * sizeof(l0cDtype));
    l0cOutUb_ = vecQueMMRes_.AllocTensor<l0cDtype>();
    // 仅AIV相关的buffer
    if ASCEND_IS_AIV {
        pipe_->InitBuffer(vecQueScale_, 1, tilingData_->matMulTilingData.tCubeTiling.baseN * sizeof(scaleType));
        pipe_->InitBuffer(vecQuePertokenScale_, 1,
                          Align(mForSingleVec * sizeof(ptScaleType), static_cast<uint64_t>(DATA_BLOCK)));
        // fp16/bf16分两次输出，fp32分四次输出
        pipe_->InitBuffer(vecQueOut_, BUFFER_NUM,
                          CeilDiv(static_cast<uint64_t>(mForSingleVec), FP32_OUTPUT_TIMES) *
                              tilingData_->matMulTilingData.tCubeTiling.baseN * sizeof(cType));
    }
}

LOCAL_TEMPLATE_CLASS_MIX_PARAMS
__aicore__ inline void TransposeQuantBatchMatMulAswKernel<LOCAL_TEMPLATE_FUNC_MIX_PARAMS>::SetOrgShape()
{
    uint64_t mergeBatchK = tilingData_->cBatchDimAll * tilingData_->matMulTilingData.tCubeTiling.Ka;
    mm.SetOrgShape(tilingData_->matMulTilingData.tCubeTiling.M, tilingData_->matMulTilingData.tCubeTiling.N,
                   mergeBatchK, tilingData_->matMulTilingData.tCubeTiling.Kb, 0);
}

LOCAL_TEMPLATE_CLASS_MIX_PARAMS
__aicore__ inline void TransposeQuantBatchMatMulAswKernel<LOCAL_TEMPLATE_FUNC_MIX_PARAMS>::UpdateGlobalAddr(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR scaleGM, GM_ADDR ptScaleGM, GM_ADDR cGM, GM_ADDR workSpace)
{
    block_.Init(tilingData_, blockIdx_);
    if ASCEND_IS_AIC {
        aGlobal_.SetGlobalBuffer((__gm__ aType*)aGM);
        bGlobal_.SetGlobalBuffer((__gm__ bType*)bGM);
    }
    if ASCEND_IS_AIV {
        scaleGlobal_.SetGlobalBuffer((__gm__ scaleType*)scaleGM);
        pertokenScaleGlobal_.SetGlobalBuffer((__gm__ ptScaleType*)ptScaleGM);
        cGlobal_.SetGlobalBuffer((__gm__ cType*)cGM);
    }
}

LOCAL_TEMPLATE_CLASS_MIX_PARAMS
__aicore__ inline void TransposeQuantBatchMatMulAswKernel<LOCAL_TEMPLATE_FUNC_MIX_PARAMS>::Process()
{
    for (uint64_t bIndex = 0; bIndex < tilingData_->cBatchDimAll; ++bIndex) {
        block_.offset_.batchOffset = bIndex;
        bool isVecSetSyncCom = false;
        for (uint64_t j = 0; j < block_.params_.round; j++) {
            block_.UpdateBasicIndex(j);
            if (block_.params_.index < block_.params_.totalCnt) {
                block_.UpdateBlockParams(j);
                if (block_.params_.singleCoreM > 0 && block_.params_.singleCoreN > 0) {
                    block_.CalcGMOffset();
                    if ASCEND_IS_AIC {
                        mm.SetSingleShape(block_.params_.singleCoreM, block_.params_.singleCoreN,
                                          tilingData_->matMulTilingData.tCubeTiling.singleCoreK);
                        if (block_.offset_.batchOffset * block_.params_.round + j > 0) {
                            WaitForVector();
                        }
                        MMCompute();
                        NotifyVector();
                    }
                    isVecSetSyncCom = true;
                    if ASCEND_IS_AIV {
                        WaitForCube();
                        DequantCompute();
                        NotifyCube();
                    }
                }
            }
        }
        // 由于vec最后一次会通过NotifyCube多发一次硬同步，所以cube侧需要额外加一次硬同步
        if ASCEND_IS_AIC {
            if (block_.offset_.batchOffset == tilingData_->cBatchDimAll - 1 && isVecSetSyncCom) {
                WaitForVector();
            }
        }
    }
}

LOCAL_TEMPLATE_CLASS_MIX_PARAMS
__aicore__ inline void TransposeQuantBatchMatMulAswKernel<LOCAL_TEMPLATE_FUNC_MIX_PARAMS>::MMCompute()
{
    mm.SetTensorA(aGlobal_[block_.offset_.offsetA], aTrans);
    mm.SetTensorB(bGlobal_[block_.offset_.offsetB], bTrans);
    mm.Iterate();
    mm.GetTensorC(l0cOutUb_, 0, true);
}

LOCAL_TEMPLATE_CLASS_MIX_PARAMS
__aicore__ inline void TransposeQuantBatchMatMulAswKernel<LOCAL_TEMPLATE_FUNC_MIX_PARAMS>::DequantCompute()
{
    auto halfSingleM = CeilDiv(block_.params_.singleCoreM, static_cast<uint64_t>(2)); // 分配给2个AIV计算
    auto singleMInVec = subBlockIdx_ == 1 ? block_.params_.singleCoreM - halfSingleM : halfSingleM;
    if (singleMInVec == 0) {
        return;
    }
    uint64_t mOffset = static_cast<uint64_t>(subBlockIdx_ * halfSingleM);
    CopyDataFromGm2Ub();
    // UB空间受限，分四次输出
    uint16_t splitNumOfOut = singleMInVec >= 4 ? 4 : singleMInVec;
    auto mSizeForOnce = CeilDiv(singleMInVec, static_cast<uint64_t>(splitNumOfOut));
    for (uint16_t i = 0; i < splitNumOfOut; i++) {
        // do dequant in vector
        uint64_t offsetL0c =
            i * mSizeForOnce * Align(block_.params_.singleCoreN, static_cast<uint64_t>(DATA_BLOCK / sizeof(l0cDtype)));
        if (i * mSizeForOnce >= singleMInVec) {
            break;
        }
        auto mSize = singleMInVec - i * mSizeForOnce >= mSizeForOnce ? mSizeForOnce : singleMInVec - i * mSizeForOnce;
        LocalTensor<cType> dequantOutInUB = vecQueOut_.AllocTensor<cType>();

        __ubuf__ cType* dequantOutInUbAddr = (__ubuf__ cType*)dequantOutInUB.GetPhyAddr();
        __ubuf__ l0cDtype* l0cOutUbAddr = (__ubuf__ l0cDtype*)l0cOutUb_.GetPhyAddr();
        l0cOutUbAddr = l0cOutUbAddr + offsetL0c;

        uint64_t offsetPtScale = i * mSizeForOnce;
        VFDoDequantWithX1Pertoken(dequantOutInUbAddr, l0cOutUbAddr, offsetPtScale, mSize);
        vecQueOut_.EnQue<cType>(dequantOutInUB);
        // mmDequant result: UB -> GM
        dequantOutInUB = vecQueOut_.DeQue<cType>();
        CopyDequantResFromUb2Gm(mSize,
                                (mOffset + i * mSizeForOnce) * tilingData_->matMulTilingData.tCubeTiling.N *
                                    tilingData_->cBatchDimAll,
                                dequantOutInUB);
        vecQueOut_.FreeTensor(dequantOutInUB);
    }
    FreeUbTensor();
}

LOCAL_TEMPLATE_CLASS_MIX_PARAMS
__aicore__ inline void TransposeQuantBatchMatMulAswKernel<LOCAL_TEMPLATE_FUNC_MIX_PARAMS>::CopyDataFromGm2Ub()
{
    auto halfSingleM = CeilDiv(block_.params_.singleCoreM, static_cast<uint64_t>(2)); // 分配给2个AIV计算
    auto singleMInVec = subBlockIdx_ == 1 ? block_.params_.singleCoreM - halfSingleM : halfSingleM;
    // scale: GM -> UB
    scaleUb_ = vecQueScale_.AllocTensor<scaleType>();
    CopyX2ScaleFromGm2Ub(scaleUb_);
    vecQueScale_.EnQue<scaleType>(scaleUb_);
    scaleUb_ = vecQueScale_.DeQue<scaleType>();

    uint64_t mOffset = subBlockIdx_ * halfSingleM;
    // perTokenScale: GM -> UB
    ptScaleUb_ = vecQuePertokenScale_.AllocTensor<ptScaleType>();
    CopyX1ScaleFromGm2Ub(ptScaleUb_, singleMInVec * sizeof(ptScaleType), block_.offset_.offsetPerTokenScale + mOffset);
    vecQuePertokenScale_.EnQue<ptScaleType>(ptScaleUb_);
    ptScaleUb_ = vecQuePertokenScale_.DeQue<ptScaleType>();
}

LOCAL_TEMPLATE_CLASS_MIX_PARAMS
__aicore__ inline void TransposeQuantBatchMatMulAswKernel<LOCAL_TEMPLATE_FUNC_MIX_PARAMS>::CopyX1ScaleFromGm2Ub(
    LocalTensor<ptScaleType>& dst, uint64_t blockLen, uint64_t offset)
{
    DataCopyParams ptScale2UbParams{1, 0, 0, 0};
    DataCopyPadParams padParams;
    ptScale2UbParams.blockLen = blockLen;
    AscendC::DataCopyPad(dst, pertokenScaleGlobal_[offset], ptScale2UbParams, padParams);
}

LOCAL_TEMPLATE_CLASS_MIX_PARAMS
__aicore__ inline void
TransposeQuantBatchMatMulAswKernel<LOCAL_TEMPLATE_FUNC_MIX_PARAMS>::CopyX2ScaleFromGm2Ub(LocalTensor<scaleType>& dst)
{
    DataCopyParams scale2UbParams{1, 0, 0, 0};
    DataCopyPadParams padParams;
    scale2UbParams.blockLen = block_.params_.singleCoreN * sizeof(scaleType);
    AscendC::DataCopyPad(dst, scaleGlobal_[block_.offset_.offsetScale], scale2UbParams, padParams);
}

LOCAL_TEMPLATE_CLASS_MIX_PARAMS
__aicore__ inline void TransposeQuantBatchMatMulAswKernel<LOCAL_TEMPLATE_FUNC_MIX_PARAMS>::CopyDequantResFromUb2Gm(
    uint64_t blockCount, uint64_t offset, LocalTensor<cType>& src)
{
    DataCopyExtParams ub2GmParams{1, 0, 0, 0, 0};
    ub2GmParams.blockLen = block_.params_.singleCoreN * sizeof(cType);
    ub2GmParams.blockCount = blockCount;
    ub2GmParams.dstStride =
        (tilingData_->matMulTilingData.tCubeTiling.N * tilingData_->cBatchDimAll - block_.params_.singleCoreN) *
        sizeof(cType);
    AscendC::DataCopyPad(cGlobal_[block_.offset_.offsetC + offset], src, ub2GmParams);
}

LOCAL_TEMPLATE_CLASS_MIX_PARAMS
__aicore__ inline void TransposeQuantBatchMatMulAswKernel<LOCAL_TEMPLATE_FUNC_MIX_PARAMS>::FreeUbTensor()
{
    vecQueScale_.FreeTensor(scaleUb_);
    vecQuePertokenScale_.FreeTensor(ptScaleUb_);
}

LOCAL_TEMPLATE_CLASS_MIX_PARAMS
__aicore__ inline void TransposeQuantBatchMatMulAswKernel<LOCAL_TEMPLATE_FUNC_MIX_PARAMS>::VFDoDequantWithX1Pertoken(
    __ubuf__ cType* dequantOutInUbAddr, __ubuf__ l0cDtype* l0cOutUbAddr, uint64_t offsetPtScale, uint16_t mSize)
{
    __ubuf__ ptScaleType* ptScaleUbAddr = (__ubuf__ ptScaleType*)ptScaleUb_.GetPhyAddr();
    ptScaleUbAddr = ptScaleUbAddr + offsetPtScale;
    VFDoDequant(dequantOutInUbAddr, l0cOutUbAddr, (__ubuf__ scaleType*)scaleUb_.GetPhyAddr(), ptScaleUbAddr, mSize,
                block_.params_.singleCoreN);
}

LOCAL_TEMPLATE_CLASS_MIX_PARAMS
__aicore__ inline void TransposeQuantBatchMatMulAswKernel<LOCAL_TEMPLATE_FUNC_MIX_PARAMS>::VFDoDequant(
    __ubuf__ cType* dst, __ubuf__ l0cDtype* l0cOut, __ubuf__ scaleType* scale, __ubuf__ ptScaleType* perTokenScale,
    uint16_t mSize, uint16_t nSize)
{
    uint32_t eleNumPerVf = platform::GetVRegSize() / sizeof(l0cDtype);
    uint32_t nSrcUbAligned = Align(nSize, static_cast<uint16_t>(DATA_BLOCK / sizeof(l0cDtype)));
    uint32_t nDstUbAligned = Align(nSize, static_cast<uint16_t>(DATA_BLOCK / sizeof(cType)));
    uint16_t nLoopCnt = (nSize + eleNumPerVf - 1) / eleNumPerVf;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::MaskReg maskN4B16 =
            AscendC::MicroAPI::CreateMask<bfloat16_t, AscendC::MicroAPI::MaskPattern::ALL>();
        for (uint16_t mIdx = 0; mIdx < mSize; mIdx++) {
            uint32_t elementNum = nSize;
            for (uint16_t vfBlockIdx = 0; vfBlockIdx < nLoopCnt; vfBlockIdx++) {
                AscendC::MicroAPI::RegTensor<l0cDtype> l0cOutReg;
                AscendC::MicroAPI::RegTensor<scaleType> scaleReg;
                AscendC::MicroAPI::RegTensor<ptScaleType> perTokenScaleReg;
                AscendC::MicroAPI::RegTensor<float> castSrcOutReg, castScaleReg, mulScaleOutReg, mulPtScaleOutReg;
                AscendC::MicroAPI::RegTensor<cType> castResultOutReg;
                AscendC::MicroAPI::MaskReg maskN = AscendC::MicroAPI::UpdateMask<l0cDtype>(elementNum);
                // copy input from ub to register, addr of ub should align to 32B
                uint32_t l0cOutOffset = mIdx * nSrcUbAligned + vfBlockIdx * eleNumPerVf;
                AscendC::MicroAPI::DataCopy(l0cOutReg, l0cOut + l0cOutOffset);
                // cast l0cOut from int32 to float
                castSrcOutReg = l0cOutReg;
                // l0c_out * scale
                AscendC::MicroAPI::DataCopy(scaleReg, scale + vfBlockIdx * eleNumPerVf);
                castScaleReg = scaleReg;
                AscendC::MicroAPI::Mul(mulScaleOutReg, castSrcOutReg, castScaleReg, maskN);
                // out * perTokenScale
                AscendC::MicroAPI::DataCopy<ptScaleType, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                    perTokenScaleReg, perTokenScale + mIdx);
                AscendC::MicroAPI::Mul(mulPtScaleOutReg, mulScaleOutReg, perTokenScaleReg, maskN);
                // cast dequant result from float to fp16/bf16
                AscendC::MicroAPI::Cast<cType, float, ctFp322Half>(castResultOutReg, mulPtScaleOutReg, maskN);
                // copy out from register to ub
                uint32_t dstUbOffset = mIdx * nDstUbAligned + vfBlockIdx * eleNumPerVf;
                AscendC::MicroAPI::DataCopy<cType, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(
                    dst + dstUbOffset, castResultOutReg, maskN);
            }
        }
    }
}

} // namespace TransposeQuantBatchMatMulAdvanced

#endif // QBMM_MIX_ONLINE_DYNAMIC_H