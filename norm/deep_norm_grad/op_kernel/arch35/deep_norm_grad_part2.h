/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* Internal continuation of deep_norm_grad.h. */
RegTensor<float> valueReg;
RegTensor<float> adjustedReg;
RegTensor<float> newSumReg;
RegTensor<float> deltaReg;
MaskReg mask;
for (uint16_t i = 0; i < loops; ++i) {
    uint32_t valid = Min(remaining, VL_FP32);
    remaining -= valid;
    mask = UpdateMask<float>(valid);
    uint32_t offset = static_cast<uint32_t>(i) * VL_FP32;
    DataCopy(sumReg, sumPtr + offset);
    DataCopy(compensationReg, compensationPtr + offset);
    DataCopy(valueReg, valuePtr + offset);
    Sub(adjustedReg, valueReg, compensationReg, mask);
    Add(newSumReg, sumReg, adjustedReg, mask);
    Sub(deltaReg, newSumReg, sumReg, mask);
    Sub(compensationReg, deltaReg, adjustedReg, mask);
    DataCopy<float, StoreDist::DIST_NORM_B32>(sumPtr + offset, newSumReg, mask);
    DataCopy<float, StoreDist::DIST_NORM_B32>(compensationPtr + offset, compensationReg, mask);
}
}
}

template <typename T>
__aicore__ inline void DeepNormGrad<T>::ProcessBackwardRow(uint64_t row)
{
    LocalTensor<float> mean = CopyInScalar(meanQueue_, meanGm_, row);
    LocalTensor<float> rstd = CopyInScalar(rstdQueue_, rstdGm_, row);
    LocalTensor<float> sumTmp = sumTmpBuf_.Get<float>();
    LocalTensor<float> sumTmpNorm = sumTmpNormBuf_.Get<float>();
    LocalTensor<float> tileSums = tileSumBuf_.Get<float>();
    LocalTensor<float> tileTmp = tileSums;
    LocalTensor<float> tileTmpNorm = tileSums[SCALAR_BLOCK_ELEMS];
    LocalTensor<float> calc0 = calc0Buf_.Get<float>();
    LocalTensor<float> calc1 = calc1Buf_.Get<float>();
    InitScalar(sumTmp);
    InitScalar(sumTmpNorm);

    uint64_t rowOffset = row * tiling_->numCols;
    for (uint64_t col = 0; col < tiling_->numCols; col += tileLength_) {
        uint32_t count = static_cast<uint32_t>(Min(static_cast<uint64_t>(tileLength_), tiling_->numCols - col));
        LocalTensor<T> dy = CopyInTensor(dyQueue_, dyGm_, rowOffset + col, count);
        LocalTensor<T> x = CopyInTensor(xQueue_, xGm_, rowOffset + col, count);
        LocalTensor<T> gx = CopyInTensor(gxQueue_, gxGm_, rowOffset + col, count);
        LocalTensor<T> gamma = CopyInTensor(gammaQueue_, gammaGm_, col, count);
        ComputeFirstPass(dy, x, gx, gamma, mean, rstd, count);
        NormCommon::NormCommonRegbase::CalculateReduceSum(calc0, tileTmp, reduceTmpBuf_, count, GetPowerSplit(count));
        NormCommon::NormCommonRegbase::CalculateReduceSum(calc1, tileTmpNorm, reduceTmpBuf_, count,
                                                          GetPowerSplit(count));
        AccumulateScalar(sumTmp, tileTmp);
        AccumulateScalar(sumTmpNorm, tileTmpNorm);
        dyQueue_.FreeTensor(dy);
        xQueue_.FreeTensor(x);
        gxQueue_.FreeTensor(gx);
        gammaQueue_.FreeTensor(gamma);
    }
    ScaleScalar(sumTmp, -invCols_);
    ScaleScalar(sumTmpNorm, -invCols_);

    for (uint64_t col = 0; col < tiling_->numCols; col += tileLength_) {
        uint32_t count = static_cast<uint32_t>(Min(static_cast<uint64_t>(tileLength_), tiling_->numCols - col));
        LocalTensor<T> dy = CopyInTensor(dyQueue_, dyGm_, rowOffset + col, count);
        LocalTensor<T> x = CopyInTensor(xQueue_, xGm_, rowOffset + col, count);
        LocalTensor<T> gx = CopyInTensor(gxQueue_, gxGm_, rowOffset + col, count);
        LocalTensor<T> gamma = CopyInTensor(gammaQueue_, gammaGm_, col, count);
        ComputeSecondPass(dy, x, gx, gamma, mean, rstd, sumTmp, sumTmpNorm, count);
        dyQueue_.FreeTensor(dy);
        xQueue_.FreeTensor(x);
        gxQueue_.FreeTensor(gx);
        gammaQueue_.FreeTensor(gamma);
        CopyOutTensor(dxQueue_, dxGm_, rowOffset + col, count);
        CopyOutTensor(dgxQueue_, dgxGm_, rowOffset + col, count);
    }
    meanQueue_.FreeTensor(mean);
    rstdQueue_.FreeTensor(rstd);
}

template <typename T>
__aicore__ inline void DeepNormGrad<T>::ProcessBackward()
{
    uint32_t core = GetBlockIdx();
    if (core >= tiling_->backwardBlockDim) {
        return;
    }
    uint64_t rowBegin = static_cast<uint64_t>(core) * tiling_->rowsPerCore;
    if (rowBegin >= tiling_->numRows) {
        return;
    }
    uint64_t rowEnd = rowBegin + Min(tiling_->rowsPerCore, tiling_->numRows - rowBegin);
    for (uint64_t row = rowBegin; row < rowEnd; ++row) {
        ProcessBackwardRow(row);
    }
}

template <typename T>
__aicore__ inline void DeepNormGrad<T>::ProcessGammaBeta()
{
    uint32_t core = GetBlockIdx();
    if (core >= tiling_->gammaBetaBlockDim) {
        return;
    }
    uint64_t colBegin = static_cast<uint64_t>(core) * tiling_->colsPerCore;
    if (colBegin >= tiling_->numCols) {
        return;
    }
    uint64_t colEnd = colBegin + Min(tiling_->colsPerCore, tiling_->numCols - colBegin);
    for (uint64_t col = colBegin; col < colEnd; col += tileLength_) {
        uint32_t count = static_cast<uint32_t>(Min(static_cast<uint64_t>(tileLength_), colEnd - col));
        LocalTensor<float> dbeta = dbetaQueue_.template AllocTensor<float>();
        LocalTensor<float> dgamma = dgammaQueue_.template AllocTensor<float>();
        Duplicate(dbeta, 0.0f, tileLengthAlign_);
        Duplicate(dgamma, 0.0f, tileLengthAlign_);
        for (uint64_t row = 0; row < tiling_->numRows; ++row) {
            uint64_t offset = row * tiling_->numCols + col;
            LocalTensor<float> mean = CopyInScalar(meanQueue_, meanGm_, row);
            LocalTensor<float> rstd = CopyInScalar(rstdQueue_, rstdGm_, row);
            LocalTensor<T> dy = CopyInTensor(dyQueue_, dyGm_, offset, count);
            LocalTensor<T> x = CopyInTensor(xQueue_, xGm_, offset, count);
            LocalTensor<T> gx = CopyInTensor(gxQueue_, gxGm_, offset, count);
            ComputeGammaBeta(dy, x, gx, mean, rstd, dgamma, dbeta, count);
            meanQueue_.FreeTensor(mean);
            rstdQueue_.FreeTensor(rstd);
            dyQueue_.FreeTensor(dy);
            xQueue_.FreeTensor(x);
            gxQueue_.FreeTensor(gx);
        }
        dbetaQueue_.EnQue(dbeta);
        dgammaQueue_.EnQue(dgamma);
        CopyOutFloat(dbetaQueue_, dbetaGm_, col, count);
        CopyOutFloat(dgammaQueue_, dgammaGm_, col, count);
    }
}

template <typename T>
__aicore__ inline void DeepNormGrad<T>::ProcessSmallD()
{
    uint32_t core = GetBlockIdx();
    uint64_t rowBegin = static_cast<uint64_t>(core) * tiling_->rowsPerCore;
    uint64_t rowEnd = rowBegin + Min(tiling_->rowsPerCore, tiling_->numRows - rowBegin);
    uint32_t count = static_cast<uint32_t>(tiling_->numCols);

    LocalTensor<T> gamma = CopyInTensor(gammaQueue_, gammaGm_, 0, count);
    LocalTensor<float> dgamma = dgammaQueue_.template AllocTensor<float>();
    LocalTensor<float> dbeta = dbetaQueue_.template AllocTensor<float>();
    LocalTensor<float> dgammaComp = dgammaCompBuf_.Get<float>();
    LocalTensor<float> dbetaComp = dbetaCompBuf_.Get<float>();
    Duplicate(dgamma, 0.0f, tiling_->smallColsAlign);
    Duplicate(dbeta, 0.0f, tiling_->smallColsAlign);
    Duplicate(dgammaComp, 0.0f, tiling_->smallColsAlign);
    Duplicate(dbetaComp, 0.0f, tiling_->smallColsAlign);

    for (uint64_t row = rowBegin; row < rowEnd; row += tiling_->smallRowsPerTile) {
        uint32_t rows = static_cast<uint32_t>(
            Min(static_cast<uint64_t>(tiling_->smallRowsPerTile), static_cast<uint64_t>(rowEnd - row)));
        uint64_t tensorOffset = row * tiling_->numCols;
        LocalTensor<T> dy = CopyInTensorBatch(dyQueue_, dyGm_, tensorOffset, rows);
        LocalTensor<T> x = CopyInTensorBatch(xQueue_, xGm_, tensorOffset, rows);
        LocalTensor<T> gx = CopyInTensorBatch(gxQueue_, gxGm_, tensorOffset, rows);
        LocalTensor<float> mean = CopyInScalars(meanQueue_, meanGm_, row, rows);
        LocalTensor<float> rstd = CopyInScalars(rstdQueue_, rstdGm_, row, rows);
        LocalTensor<T> dx = dxQueue_.template AllocTensor<T>();
        LocalTensor<T> dgx = dgxQueue_.template AllocTensor<T>();

        if (count == 1) {
            ComputeTinyDBatch<1>(dy, x, gx, gamma, mean, rstd, dx, dgx, dgamma, dbeta, dgammaComp, dbetaComp, rows);
        } else if (count == 2) {
            ComputeTinyDBatch<2>(dy, x, gx, gamma, mean, rstd, dx, dgx, dgamma, dbeta, dgammaComp, dbetaComp, rows);
        } else {
            for (uint32_t localRow = 0; localRow < rows; ++localRow) {
                uint32_t localOffset = localRow * tiling_->smallRowStride;
                LocalTensor<T> dyRow = dy[localOffset];
                LocalTensor<T> xRow = x[localOffset];
                LocalTensor<T> gxRow = gx[localOffset];
                LocalTensor<T> dxRow = dx[localOffset];
                LocalTensor<T> dgxRow = dgx[localOffset];
                LocalTensor<float> meanRow = mean[localRow];
                LocalTensor<float> rstdRow = rstd[localRow];
                LocalTensor<float> sumTmp = sumTmpBuf_.Get<float>();
                LocalTensor<float> sumTmpNorm = sumTmpNormBuf_.Get<float>();
                LocalTensor<float> tileSums = tileSumBuf_.Get<float>();
                LocalTensor<float> tileTmp = tileSums;
                LocalTensor<float> tileTmpNorm = tileSums[SCALAR_BLOCK_ELEMS];
                LocalTensor<float> calc0 = calc0Buf_.Get<float>();
                LocalTensor<float> calc1 = calc1Buf_.Get<float>();
                InitScalar(sumTmp);
                InitScalar(sumTmpNorm);
                ComputeFirstPass(dyRow, xRow, gxRow, gamma, meanRow, rstdRow, count);
                NormCommon::NormCommonRegbase::CalculateReduceSum(calc0, tileTmp, reduceTmpBuf_, count,
                                                                  GetPowerSplit(count));
                NormCommon::NormCommonRegbase::CalculateReduceSum(calc1, tileTmpNorm, reduceTmpBuf_, count,
                                                                  GetPowerSplit(count));
                AccumulateScalar(sumTmp, tileTmp);
                AccumulateScalar(sumTmpNorm, tileTmpNorm);
                ScaleScalar(sumTmp, -invCols_);
                ScaleScalar(sumTmpNorm, -invCols_);
                ComputeSmallDSecondPass(dyRow, xRow, gxRow, gamma, meanRow, rstdRow, sumTmp, sumTmpNorm, dxRow, dgxRow,
                                        dgamma, dbeta, dgammaComp, dbetaComp, count);
            }
        }

        dxQueue_.EnQue(dx);
        dgxQueue_.EnQue(dgx);
        dyQueue_.FreeTensor(dy);
        xQueue_.FreeTensor(x);
        gxQueue_.FreeTensor(gx);
        meanQueue_.FreeTensor(mean);
        rstdQueue_.FreeTensor(rstd);
        CopyOutTensorBatch(dxQueue_, dxGm_, tensorOffset, rows);
        CopyOutTensorBatch(dgxQueue_, dgxGm_, tensorOffset, rows);
    }
    gammaQueue_.FreeTensor(gamma);

    uint64_t workspaceBase = static_cast<uint64_t>(core) * 2 * tiling_->smallColsAlign;
    dgammaQueue_.EnQue(dgamma);
    dbetaQueue_.EnQue(dbeta);
    CopyOutFloat(dgammaQueue_, workspaceGm_, workspaceBase, count);
    CopyOutFloat(dbetaQueue_, workspaceGm_, workspaceBase + tiling_->smallColsAlign, count);
    ReduceSmallDPartials();
}

template <typename T>
__aicore__ inline void DeepNormGrad<T>::ReduceSmallDPartials()
{
    SyncAll();
    if (GetBlockIdx() != 0) {
        return;
    }

    uint32_t count = static_cast<uint32_t>(tiling_->numCols);
    LocalTensor<float> dgamma = dgammaQueue_.template AllocTensor<float>();
    LocalTensor<float> dbeta = dbetaQueue_.template AllocTensor<float>();
    LocalTensor<float> dgammaComp = calc0Buf_.Get<float>();
    LocalTensor<float> dbetaComp = calc1Buf_.Get<float>();
    Duplicate(dgamma, 0.0f, tiling_->smallColsAlign);
    Duplicate(dbeta, 0.0f, tiling_->smallColsAlign);
    Duplicate(dgammaComp, 0.0f, tiling_->smallColsAlign);
    Duplicate(dbetaComp, 0.0f, tiling_->smallColsAlign);

    for (uint32_t core = 0; core < tiling_->gammaBetaBlockDim; ++core) {
        uint64_t workspaceBase = static_cast<uint64_t>(core) * 2 * tiling_->smallColsAlign;
        LocalTensor<float> dgammaPartial = CopyInFloatTensor(dyQueue_, workspaceGm_, workspaceBase, count);
        LocalTensor<float> dbetaPartial = CopyInFloatTensor(xQueue_, workspaceGm_,
                                                            workspaceBase + tiling_->smallColsAlign, count);
        AccumulateKahan(dgamma, dgammaComp, dgammaPartial, count);
        AccumulateKahan(dbeta, dbetaComp, dbetaPartial, count);
        dyQueue_.FreeTensor(dgammaPartial);
        xQueue_.FreeTensor(dbetaPartial);
    }

    dgammaQueue_.EnQue(dgamma);
    dbetaQueue_.EnQue(dbeta);
    CopyOutFloat(dgammaQueue_, dgammaGm_, 0, count);
    CopyOutFloat(dbetaQueue_, dbetaGm_, 0, count);
}

template <typename T>
__aicore__ inline void DeepNormGrad<T>::Process()
{
    if (tiling_->gammaBetaRowSplit != 0) {
        ProcessSmallD();
    } else {
        ProcessBackward();
        ProcessGammaBeta();
    }
}

} // namespace DeepNormGradArch35

#endif // DEEP_NORM_GRAD_ARCH35_H
