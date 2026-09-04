/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* Internal implementation section of deep_norm_grad.h. Include only from deep_norm_grad.h. */

template <typename T>
__aicore__ inline void DeepNormGrad<T>::ComputeFirstPass(LocalTensor<T>& dy, LocalTensor<T>& x, LocalTensor<T>& gx,
                                                         LocalTensor<T>& gamma, LocalTensor<float>& mean,
                                                         LocalTensor<float>& rstd, uint32_t count)
{
    __local_mem__ T* dyPtr = (__local_mem__ T*)dy.GetPhyAddr();
    __local_mem__ T* xPtr = (__local_mem__ T*)x.GetPhyAddr();
    __local_mem__ T* gxPtr = (__local_mem__ T*)gx.GetPhyAddr();
    __local_mem__ T* gammaPtr = (__local_mem__ T*)gamma.GetPhyAddr();
    __local_mem__ float* meanPtr = (__local_mem__ float*)mean.GetPhyAddr();
    __local_mem__ float* rstdPtr = (__local_mem__ float*)rstd.GetPhyAddr();
    __local_mem__ float* tmpPtr = (__local_mem__ float*)calc0Buf_.Get<float>().GetPhyAddr();
    __local_mem__ float* tmpNormPtr = (__local_mem__ float*)calc1Buf_.Get<float>().GetPhyAddr();
    uint16_t loops = static_cast<uint16_t>((count + VL_FP32 - 1) / VL_FP32);
    uint32_t remaining = count;
    float alpha = alpha_;
    __VEC_SCOPE__
    {
        RegTensor<float> dyReg;
        RegTensor<float> xReg;
        RegTensor<float> gxReg;
        RegTensor<float> gammaReg;
        RegTensor<float> meanReg;
        RegTensor<float> rstdReg;
        RegTensor<float> tmpReg;
        RegTensor<float> normReg;
        RegTensor<float> productReg;
        MaskReg mask;
        DataCopy<float, LoadDist::DIST_BRC_B32>(meanReg, meanPtr);
        DataCopy<float, LoadDist::DIST_BRC_B32>(rstdReg, rstdPtr);
        for (uint16_t i = 0; i < loops; ++i) {
            uint32_t valid = Min(remaining, VL_FP32);
            remaining -= valid;
            mask = UpdateMask<float>(valid);
            uint32_t offset = static_cast<uint32_t>(i) * VL_FP32;
            NormCommon::NormCommonRegbase::LoadRegForDtype<T>(dyPtr, dyReg, mask, offset);
            NormCommon::NormCommonRegbase::LoadRegForDtype<T>(gammaPtr, gammaReg, mask, offset);
            NormCommon::NormCommonRegbase::LoadRegForDtype<T>(xPtr, xReg, mask, offset);
            NormCommon::NormCommonRegbase::LoadRegForDtype<T>(gxPtr, gxReg, mask, offset);
            Mul(tmpReg, dyReg, gammaReg, mask);
            Muls(xReg, xReg, alpha, mask);
            Add(normReg, xReg, gxReg, mask);
            Sub(normReg, normReg, meanReg, mask);
            Mul(productReg, rstdReg, rstdReg, mask);
            Mul(productReg, productReg, rstdReg, mask);
            Mul(productReg, productReg, normReg, mask);
            Mul(productReg, productReg, tmpReg, mask);
            Mul(normReg, tmpReg, rstdReg, mask);
            DataCopy<float, StoreDist::DIST_NORM_B32>(tmpPtr + offset, productReg, mask);
            DataCopy<float, StoreDist::DIST_NORM_B32>(tmpNormPtr + offset, normReg, mask);
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }
}

template <typename T>
__aicore__ inline void DeepNormGrad<T>::ComputeSecondPass(LocalTensor<T>& dy, LocalTensor<T>& x, LocalTensor<T>& gx,
                                                          LocalTensor<T>& gamma, LocalTensor<float>& mean,
                                                          LocalTensor<float>& rstd, LocalTensor<float>& avgTmp,
                                                          LocalTensor<float>& avgTmpNorm, uint32_t count)
{
    LocalTensor<T> dx = dxQueue_.template AllocTensor<T>();
    LocalTensor<T> dgx = dgxQueue_.template AllocTensor<T>();
    __local_mem__ T* dyPtr = (__local_mem__ T*)dy.GetPhyAddr();
    __local_mem__ T* xPtr = (__local_mem__ T*)x.GetPhyAddr();
    __local_mem__ T* gxPtr = (__local_mem__ T*)gx.GetPhyAddr();
    __local_mem__ T* gammaPtr = (__local_mem__ T*)gamma.GetPhyAddr();
    __local_mem__ T* dxPtr = (__local_mem__ T*)dx.GetPhyAddr();
    __local_mem__ T* dgxPtr = (__local_mem__ T*)dgx.GetPhyAddr();
    __local_mem__ float* meanPtr = (__local_mem__ float*)mean.GetPhyAddr();
    __local_mem__ float* rstdPtr = (__local_mem__ float*)rstd.GetPhyAddr();
    __local_mem__ float* avgTmpPtr = (__local_mem__ float*)avgTmp.GetPhyAddr();
    __local_mem__ float* avgTmpNormPtr = (__local_mem__ float*)avgTmpNorm.GetPhyAddr();
    uint16_t loops = static_cast<uint16_t>((count + VL_FP32 - 1) / VL_FP32);
    uint32_t remaining = count;
    float alpha = alpha_;
    __VEC_SCOPE__
    {
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        RegTensor<float> dyReg;
        RegTensor<float> xReg;
        RegTensor<float> gxReg;
        RegTensor<float> gammaReg;
        RegTensor<float> meanReg;
        RegTensor<float> rstdReg;
        RegTensor<float> avgTmpReg;
        RegTensor<float> avgTmpNormReg;
        RegTensor<float> tmpReg;
        RegTensor<float> normReg;
        RegTensor<float> dgxReg;
        RegTensor<float> dxReg;
        MaskReg mask;
        DataCopy<float, LoadDist::DIST_BRC_B32>(meanReg, meanPtr);
        DataCopy<float, LoadDist::DIST_BRC_B32>(rstdReg, rstdPtr);
        DataCopy<float, LoadDist::DIST_BRC_B32>(avgTmpReg, avgTmpPtr);
        DataCopy<float, LoadDist::DIST_BRC_B32>(avgTmpNormReg, avgTmpNormPtr);
        for (uint16_t i = 0; i < loops; ++i) {
            uint32_t valid = Min(remaining, VL_FP32);
            remaining -= valid;
            mask = UpdateMask<float>(valid);
            uint32_t offset = static_cast<uint32_t>(i) * VL_FP32;
            NormCommon::NormCommonRegbase::LoadRegForDtype<T>(dyPtr, dyReg, mask, offset);
            NormCommon::NormCommonRegbase::LoadRegForDtype<T>(gammaPtr, gammaReg, mask, offset);
            NormCommon::NormCommonRegbase::LoadRegForDtype<T>(xPtr, xReg, mask, offset);
            NormCommon::NormCommonRegbase::LoadRegForDtype<T>(gxPtr, gxReg, mask, offset);
            Mul(tmpReg, dyReg, gammaReg, mask);
            Muls(xReg, xReg, alpha, mask);
            Add(normReg, xReg, gxReg, mask);
            Sub(normReg, normReg, meanReg, mask);
            Mul(dgxReg, tmpReg, rstdReg, mask);
            Mul(normReg, normReg, avgTmpReg, mask);
            Add(dgxReg, dgxReg, normReg, mask);
            Add(dgxReg, dgxReg, avgTmpNormReg, mask);
            Muls(dxReg, dgxReg, alpha, mask);
            StoreOutputForDtype(dgxPtr, dgxReg, mask, offset);
            StoreOutputForDtype(dxPtr, dxReg, mask, offset);
        }
    }
    dxQueue_.EnQue(dx);
    dgxQueue_.EnQue(dgx);
}

template <typename T>
__aicore__ inline void DeepNormGrad<T>::ComputeGammaBeta(LocalTensor<T>& dy, LocalTensor<T>& x, LocalTensor<T>& gx,
                                                         LocalTensor<float>& mean, LocalTensor<float>& rstd,
                                                         LocalTensor<float>& dgamma, LocalTensor<float>& dbeta,
                                                         uint32_t count)
{
    __local_mem__ T* dyPtr = (__local_mem__ T*)dy.GetPhyAddr();
    __local_mem__ T* xPtr = (__local_mem__ T*)x.GetPhyAddr();
    __local_mem__ T* gxPtr = (__local_mem__ T*)gx.GetPhyAddr();
    __local_mem__ float* meanPtr = (__local_mem__ float*)mean.GetPhyAddr();
    __local_mem__ float* rstdPtr = (__local_mem__ float*)rstd.GetPhyAddr();
    __local_mem__ float* dgammaPtr = (__local_mem__ float*)dgamma.GetPhyAddr();
    __local_mem__ float* dbetaPtr = (__local_mem__ float*)dbeta.GetPhyAddr();
    uint16_t loops = static_cast<uint16_t>((count + VL_FP32 - 1) / VL_FP32);
    uint32_t remaining = count;
    float alpha = alpha_;
    __VEC_SCOPE__
    {
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        RegTensor<float> dyReg;
        RegTensor<float> xReg;
        RegTensor<float> gxReg;
        RegTensor<float> meanReg;
        RegTensor<float> rstdReg;
        RegTensor<float> normReg;
        RegTensor<float> dgammaReg;
        RegTensor<float> dbetaReg;
        RegTensor<float> productReg;
        MaskReg mask;
        DataCopy<float, LoadDist::DIST_BRC_B32>(meanReg, meanPtr);
        DataCopy<float, LoadDist::DIST_BRC_B32>(rstdReg, rstdPtr);
        for (uint16_t i = 0; i < loops; ++i) {
            uint32_t valid = Min(remaining, VL_FP32);
            remaining -= valid;
            mask = UpdateMask<float>(valid);
            uint32_t offset = static_cast<uint32_t>(i) * VL_FP32;
            NormCommon::NormCommonRegbase::LoadRegForDtype<T>(dyPtr, dyReg, mask, offset);
            NormCommon::NormCommonRegbase::LoadRegForDtype<T>(xPtr, xReg, mask, offset);
            NormCommon::NormCommonRegbase::LoadRegForDtype<T>(gxPtr, gxReg, mask, offset);
            DataCopy(dgammaReg, dgammaPtr + offset);
            DataCopy(dbetaReg, dbetaPtr + offset);
            Muls(xReg, xReg, alpha, mask);
            Add(normReg, xReg, gxReg, mask);
            Sub(normReg, normReg, meanReg, mask);
            Mul(normReg, normReg, rstdReg, mask);
            Mul(productReg, dyReg, normReg, mask);
            Add(dgammaReg, dgammaReg, productReg, mask);
            Add(dbetaReg, dbetaReg, dyReg, mask);
            DataCopy<float, StoreDist::DIST_NORM_B32>(dgammaPtr + offset, dgammaReg, mask);
            DataCopy<float, StoreDist::DIST_NORM_B32>(dbetaPtr + offset, dbetaReg, mask);
        }
    }
}

template <typename T>
__aicore__ inline void DeepNormGrad<T>::ComputeSmallDSecondPass(
    LocalTensor<T>& dy, LocalTensor<T>& x, LocalTensor<T>& gx, LocalTensor<T>& gamma, LocalTensor<float>& mean,
    LocalTensor<float>& rstd, LocalTensor<float>& avgTmp, LocalTensor<float>& avgTmpNorm, LocalTensor<T>& dx,
    LocalTensor<T>& dgx, LocalTensor<float>& dgamma, LocalTensor<float>& dbeta, LocalTensor<float>& dgammaComp,
    LocalTensor<float>& dbetaComp, uint32_t count)
{
    __local_mem__ T* dyPtr = (__local_mem__ T*)dy.GetPhyAddr();
    __local_mem__ T* xPtr = (__local_mem__ T*)x.GetPhyAddr();
    __local_mem__ T* gxPtr = (__local_mem__ T*)gx.GetPhyAddr();
    __local_mem__ T* gammaPtr = (__local_mem__ T*)gamma.GetPhyAddr();
    __local_mem__ T* dxPtr = (__local_mem__ T*)dx.GetPhyAddr();
    __local_mem__ T* dgxPtr = (__local_mem__ T*)dgx.GetPhyAddr();
    __local_mem__ float* meanPtr = (__local_mem__ float*)mean.GetPhyAddr();
    __local_mem__ float* rstdPtr = (__local_mem__ float*)rstd.GetPhyAddr();
    __local_mem__ float* avgTmpPtr = (__local_mem__ float*)avgTmp.GetPhyAddr();
    __local_mem__ float* avgTmpNormPtr = (__local_mem__ float*)avgTmpNorm.GetPhyAddr();
    __local_mem__ float* dgammaPtr = (__local_mem__ float*)dgamma.GetPhyAddr();
    __local_mem__ float* dbetaPtr = (__local_mem__ float*)dbeta.GetPhyAddr();
    __local_mem__ float* dgammaCompPtr = (__local_mem__ float*)dgammaComp.GetPhyAddr();
    __local_mem__ float* dbetaCompPtr = (__local_mem__ float*)dbetaComp.GetPhyAddr();
    uint16_t loops = static_cast<uint16_t>((count + VL_FP32 - 1) / VL_FP32);
    uint32_t remaining = count;
    float alpha = alpha_;
    __VEC_SCOPE__
    {
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        RegTensor<float> dyReg;
        RegTensor<float> xReg;
        RegTensor<float> gxReg;
        RegTensor<float> gammaReg;
        RegTensor<float> meanReg;
        RegTensor<float> rstdReg;
        RegTensor<float> avgTmpReg;
        RegTensor<float> avgTmpNormReg;
        RegTensor<float> normReg;
        RegTensor<float> dyGammaReg;
        RegTensor<float> dgxReg;
        RegTensor<float> dxReg;
        RegTensor<float> valueReg;
        RegTensor<float> sumReg;
        RegTensor<float> compReg;
        RegTensor<float> adjustedReg;
        RegTensor<float> newSumReg;
        RegTensor<float> deltaReg;
        MaskReg mask;
        DataCopy<float, LoadDist::DIST_BRC_B32>(meanReg, meanPtr);
        DataCopy<float, LoadDist::DIST_BRC_B32>(rstdReg, rstdPtr);
        DataCopy<float, LoadDist::DIST_BRC_B32>(avgTmpReg, avgTmpPtr);
        DataCopy<float, LoadDist::DIST_BRC_B32>(avgTmpNormReg, avgTmpNormPtr);
        for (uint16_t i = 0; i < loops; ++i) {
            uint32_t valid = Min(remaining, VL_FP32);
            remaining -= valid;
            mask = UpdateMask<float>(valid);
            uint32_t offset = static_cast<uint32_t>(i) * VL_FP32;
            NormCommon::NormCommonRegbase::LoadRegForDtype<T>(dyPtr, dyReg, mask, offset);
            NormCommon::NormCommonRegbase::LoadRegForDtype<T>(xPtr, xReg, mask, offset);
            NormCommon::NormCommonRegbase::LoadRegForDtype<T>(gxPtr, gxReg, mask, offset);
            NormCommon::NormCommonRegbase::LoadRegForDtype<T>(gammaPtr, gammaReg, mask, offset);
            DataCopy(sumReg, dgammaPtr + offset);
            DataCopy(compReg, dgammaCompPtr + offset);

            Muls(xReg, xReg, alpha, mask);
            Add(normReg, xReg, gxReg, mask);
            Sub(normReg, normReg, meanReg, mask);
            Mul(valueReg, dyReg, normReg, mask);
            Mul(valueReg, valueReg, rstdReg, mask);
            Sub(adjustedReg, valueReg, compReg, mask);
            Add(newSumReg, sumReg, adjustedReg, mask);
            Sub(deltaReg, newSumReg, sumReg, mask);
            Sub(compReg, deltaReg, adjustedReg, mask);
            DataCopy<float, StoreDist::DIST_NORM_B32>(dgammaPtr + offset, newSumReg, mask);
            DataCopy<float, StoreDist::DIST_NORM_B32>(dgammaCompPtr + offset, compReg, mask);

            DataCopy(sumReg, dbetaPtr + offset);
            DataCopy(compReg, dbetaCompPtr + offset);
            Sub(adjustedReg, dyReg, compReg, mask);
            Add(newSumReg, sumReg, adjustedReg, mask);
            Sub(deltaReg, newSumReg, sumReg, mask);
            Sub(compReg, deltaReg, adjustedReg, mask);
            DataCopy<float, StoreDist::DIST_NORM_B32>(dbetaPtr + offset, newSumReg, mask);
            DataCopy<float, StoreDist::DIST_NORM_B32>(dbetaCompPtr + offset, compReg, mask);

            Mul(dyGammaReg, dyReg, gammaReg, mask);
            Mul(dgxReg, dyGammaReg, rstdReg, mask);
            Mul(normReg, normReg, avgTmpReg, mask);
            Add(dgxReg, dgxReg, normReg, mask);
            Add(dgxReg, dgxReg, avgTmpNormReg, mask);
            Muls(dxReg, dgxReg, alpha, mask);
            StoreOutputForDtype(dgxPtr, dgxReg, mask, offset);
            StoreOutputForDtype(dxPtr, dxReg, mask, offset);
        }
    }
}

template <typename T>
template <uint32_t COLS>
__aicore__ inline void DeepNormGrad<T>::ComputeTinyDBatch(LocalTensor<T>& dy, LocalTensor<T>& x, LocalTensor<T>& gx,
                                                          LocalTensor<T>& gamma, LocalTensor<float>& mean,
                                                          LocalTensor<float>& rstd, LocalTensor<T>& dx,
                                                          LocalTensor<T>& dgx, LocalTensor<float>& dgamma,
                                                          LocalTensor<float>& dbeta, LocalTensor<float>& dgammaComp,
                                                          LocalTensor<float>& dbetaComp, uint32_t rows)
{
    static_assert(COLS == 1 || COLS == 2, "tiny-D batch only supports one or two columns");
    __local_mem__ T* dyPtr = (__local_mem__ T*)dy.GetPhyAddr();
    __local_mem__ T* xPtr = (__local_mem__ T*)x.GetPhyAddr();
    __local_mem__ T* gxPtr = (__local_mem__ T*)gx.GetPhyAddr();
    __local_mem__ T* gammaPtr = (__local_mem__ T*)gamma.GetPhyAddr();
    __local_mem__ T* dxPtr = (__local_mem__ T*)dx.GetPhyAddr();
    __local_mem__ T* dgxPtr = (__local_mem__ T*)dgx.GetPhyAddr();
    __local_mem__ float* meanPtr = (__local_mem__ float*)mean.GetPhyAddr();
    __local_mem__ float* rstdPtr = (__local_mem__ float*)rstd.GetPhyAddr();
    __local_mem__ float* dgammaPtr = (__local_mem__ float*)dgamma.GetPhyAddr();
    __local_mem__ float* dbetaPtr = (__local_mem__ float*)dbeta.GetPhyAddr();
    __local_mem__ float* dgammaCompPtr = (__local_mem__ float*)dgammaComp.GetPhyAddr();
    __local_mem__ float* dbetaCompPtr = (__local_mem__ float*)dbetaComp.GetPhyAddr();
    uint32_t rowStride = tiling_->smallRowStride;
    float alpha = alpha_;
    float negInvCols = -invCols_;
    __VEC_SCOPE__
    {
        RegTensor<float> dyReg;
        RegTensor<float> xReg;
        RegTensor<float> gxReg;
        RegTensor<float> gammaReg;
        RegTensor<float> meanReg;
        RegTensor<float> rstdReg;
        RegTensor<float> normReg;
        RegTensor<float> dyGammaReg;
        RegTensor<float> tmpNormReg;
        RegTensor<float> productReg;
        RegTensor<float> avgTmpReg;
        RegTensor<float> avgTmpDupReg;
        RegTensor<float> avgTmpNormReg;
        RegTensor<float> avgTmpNormDupReg;
        RegTensor<float> dgxReg;
        RegTensor<float> dxReg;
        RegTensor<float> dgammaSumReg;
        RegTensor<float> dbetaSumReg;
        RegTensor<float> dgammaCompReg;
        RegTensor<float> dbetaCompReg;
        RegTensor<float> adjustedReg;
        RegTensor<float> newSumReg;
        RegTensor<float> deltaReg;
        RegTensor<float> valueReg;
        uint32_t validCols = COLS;
        MaskReg mask = UpdateMask<float>(validCols);
        MaskReg scalarMask = CreateMask<float, MaskPattern::VL1>();

        NormCommon::NormCommonRegbase::LoadRegForDtype<T>(gammaPtr, gammaReg, mask, 0);
        DataCopy(dgammaSumReg, dgammaPtr);
        DataCopy(dbetaSumReg, dbetaPtr);
        DataCopy(dgammaCompReg, dgammaCompPtr);
        DataCopy(dbetaCompReg, dbetaCompPtr);
        for (uint16_t localRow = 0; localRow < rows; ++localRow) {
            uint32_t offset = static_cast<uint32_t>(localRow) * rowStride;
            NormCommon::NormCommonRegbase::LoadRegForDtype<T>(dyPtr, dyReg, mask, offset);
            NormCommon::NormCommonRegbase::LoadRegForDtype<T>(xPtr, xReg, mask, offset);
            NormCommon::NormCommonRegbase::LoadRegForDtype<T>(gxPtr, gxReg, mask, offset);
            DataCopy<float, LoadDist::DIST_BRC_B32>(meanReg, meanPtr + localRow);
            DataCopy<float, LoadDist::DIST_BRC_B32>(rstdReg, rstdPtr + localRow);

            Muls(xReg, xReg, alpha, mask);
            Add(normReg, xReg, gxReg, mask);
            Sub(normReg, normReg, meanReg, mask);

            Mul(valueReg, dyReg, normReg, mask);
            Mul(valueReg, valueReg, rstdReg, mask);
            Sub(adjustedReg, valueReg, dgammaCompReg, mask);
            Add(newSumReg, dgammaSumReg, adjustedReg, mask);
            Sub(deltaReg, newSumReg, dgammaSumReg, mask);
            Sub(dgammaCompReg, deltaReg, adjustedReg, mask);
            AscendC::Reg::Copy<float, AscendC::Reg::MaskMergeMode::MERGING>(dgammaSumReg, newSumReg, mask);

            Sub(adjustedReg, dyReg, dbetaCompReg, mask);
            Add(newSumReg, dbetaSumReg, adjustedReg, mask);
            Sub(deltaReg, newSumReg, dbetaSumReg, mask);
            Sub(dbetaCompReg, deltaReg, adjustedReg, mask);
            AscendC::Reg::Copy<float, AscendC::Reg::MaskMergeMode::MERGING>(dbetaSumReg, newSumReg, mask);

            Mul(dyGammaReg, dyReg, gammaReg, mask);
            Mul(tmpNormReg, dyGammaReg, rstdReg, mask);
            Mul(productReg, rstdReg, rstdReg, mask);
            Mul(productReg, productReg, rstdReg, mask);
            Mul(productReg, productReg, normReg, mask);
            Mul(productReg, productReg, dyGammaReg, mask);
            if constexpr (COLS == 1) {
                Muls(avgTmpReg, productReg, negInvCols, mask);
                Muls(avgTmpNormReg, tmpNormReg, negInvCols, mask);
                Mul(dgxReg, normReg, avgTmpReg, mask);
                Add(dgxReg, tmpNormReg, dgxReg, mask);
                Add(dgxReg, dgxReg, avgTmpNormReg, mask);
            } else {
                ReduceSum(avgTmpReg, productReg, mask);
                Muls(avgTmpReg, avgTmpReg, negInvCols, scalarMask);
                Duplicate(avgTmpDupReg, avgTmpReg, mask);
                ReduceSum(avgTmpNormReg, tmpNormReg, mask);
                Muls(avgTmpNormReg, avgTmpNormReg, negInvCols, scalarMask);
                Duplicate(avgTmpNormDupReg, avgTmpNormReg, mask);
                Mul(dgxReg, normReg, avgTmpDupReg, mask);
                Add(dgxReg, tmpNormReg, dgxReg, mask);
                Add(dgxReg, dgxReg, avgTmpNormDupReg, mask);
            }
            Muls(dxReg, dgxReg, alpha, mask);
            StoreOutputForDtype(dgxPtr, dgxReg, mask, offset);
            StoreOutputForDtype(dxPtr, dxReg, mask, offset);
        }
        DataCopy<float, StoreDist::DIST_NORM_B32>(dgammaPtr, dgammaSumReg, mask);
        DataCopy<float, StoreDist::DIST_NORM_B32>(dbetaPtr, dbetaSumReg, mask);
        DataCopy<float, StoreDist::DIST_NORM_B32>(dgammaCompPtr, dgammaCompReg, mask);
        DataCopy<float, StoreDist::DIST_NORM_B32>(dbetaCompPtr, dbetaCompReg, mask);
    }
}
