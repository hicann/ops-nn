/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file batch_matmul_v3_iterbatch_basicapi_tiling.cc
 * \brief
 */

#include "batch_matmul_v3_iterbatch_basicapi_tiling.h"
#include "batch_matmul_v3_tiling_strategy.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_tiling_registry.h"
#include "batch_matmul_v3_tiling_key.h"

namespace optiling {
namespace batch_matmul_v3_advanced {
using namespace strategy;
using StrideIndexPairs = std::vector<std::pair<int64_t, std::pair<int64_t, int64_t>>>;
MM_REGISTER_TILING_TEMPLATE(BatchMatMulV3, BatchMatMulV3IterBatchBasicApiTiling, DAV_3510, ITER_BATCH_BASICAPI);

bool BatchMatMulV3IterBatchBasicApiTiling::IsContiguousStride(StrideIndexPairs& strideIndexPairs) const
{
    int64_t expectStride = 1;
    for (auto it = strideIndexPairs.rbegin(); it != strideIndexPairs.rend(); it++) {
        if (it->first != expectStride) {
            return false;
        }
        expectStride *= it->second.second;
    }
    return true;
}

uint64_t BatchMatMulV3IterBatchBasicApiTiling::GetTilingKey() const
{
    return BatchMatMulV3TilingKey()
        .SetTrans(args_.isATrans, args_.isBTrans)
        .SetBatchModel(MatMulV3BatchModel::SINGLE_BIAS_MODEL)
        .SetL0C2Out(l0C2Out_)
        .SetApiLevel(MatMulV3ApiLevel::BASIC_LEVEL)
        .GetTilingKey();
}

bool BatchMatMulV3IterBatchBasicApiTiling::IsMat2TransposeNonContiguous(
    const gert::Shape& viewShape) const
{
    // 获得stride 然后根据stride判断
    auto mat2ViewStride = context_->GetInputStride(1);
    int64_t dimNum = mat2ViewStride->GetDimNum();
    StrideIndexPairs strideIndexPairs;
    strideIndexPairs.reserve(dimNum);
    auto lastStride = INT64_MAX;
    bool isTranspose = false;
    for (int64_t i = 0; i < dimNum; i++) {
        int64_t curStride = mat2ViewStride->GetStride(i);
        if (curStride == 0 || viewShape[i] == 1) {
            return false;
        }
        if (lastStride < curStride) {
            isTranspose = true;
        }
        lastStride = curStride;
        strideIndexPairs.emplace_back(std::make_pair(curStride, std::make_pair(i, viewShape[i])));
    }
    if (!isTranspose) {
        return false;
    }
    // strides顺序排序
    std::sort(strideIndexPairs.rbegin(), strideIndexPairs.rend());
    if (!IsContiguousStride(strideIndexPairs)) {
        return false;
    }
    std::vector<int> indexs;
    for (auto it = strideIndexPairs.begin(); it != strideIndexPairs.end(); it++) {
        indexs.push_back(it->second.first);
    }
    // 3D场景只有下标符合{2,0,1} or {1,0,2}才是满足支持transpose场景
    std::set<std::vector<int>> transposeNeed = {{2, 0, 1}};
    std::set<std::vector<int>> transposeNoNeed = {{1, 0, 2}};
    auto isNeedSwap = find(transposeNeed.begin(), transposeNeed.end(), indexs);
    auto isNoNeedSwap = find(transposeNoNeed.begin(), transposeNoNeed.end(), indexs);
    if (isNeedSwap == transposeNeed.end() && isNoNeedSwap == transposeNoNeed.end()) {
        return false;
    }
    return true;
}

bool BatchMatMulV3IterBatchBasicApiTiling::IsCapable()
{
    if (batchInfo_->batchBias > 1UL) {
        return false;
    }
    bool isNotEqualBatch = batchInfo_->batchA0 != batchInfo_->batchB0 || batchInfo_->batchA1 != batchInfo_->batchB1 ||
                           batchInfo_->batchA2 != batchInfo_->batchB2 || batchInfo_->batchA3 != batchInfo_->batchB3;
    if (isNotEqualBatch)  {
        return false;
    }
    if (batchInfo_->batchC <= compileInfo_.aicNum) {
        return false;
    }
    c0Size_ = BLOCK_BYTE_SIZE / args_.aDtypeSize;
    // when fp16 or (fp32 and m,k), m align to 16; when fp32 and k,m, m align to 8 * 2 for frac combine in loadtol0a
    alignMValue_ = ops::CeilAlign(args_.mValue, BASIC_BLOCK_SIZE_16);
    alignKValue_ = ops::CeilAlign(args_.kValue, BASIC_BLOCK_SIZE_16);
    // when fp16 or (fp32 and n,k), m align to 16; when fp32 and k,n, n align to 8 * 2 for frac combine in loadtol0b
    alignNValue_ = ops::CeilAlign(args_.nValue, BASIC_BLOCK_SIZE_16);
    uint64_t alignBiasSize = 0UL;
    if (args_.hasBias) {
        alignBiasSize = alignNValue_ * GetSizeByDataType(args_.biasType);
    }
    if (((alignMValue_ * alignKValue_ + alignKValue_ * alignNValue_) * args_.aDtypeSize + alignBiasSize) * DB_SIZE >
        compileInfo_.l1Size) {
        return false;
    }

    iterBatchL0A_ = ops::FloorDiv(compileInfo_.l0ASize / DB_SIZE, alignMValue_ * alignKValue_ * args_.aDtypeSize);
    iterBatchL0B_ = ops::FloorDiv(compileInfo_.l0BSize / DB_SIZE, alignKValue_ * alignNValue_ * args_.aDtypeSize);
    iterBatchL0C_ = ops::FloorDiv(compileInfo_.l0CSize / DB_SIZE, alignMValue_ * alignNValue_ * DATA_SIZE_FP32);
    iterBatchL1_ = ops::FloorDiv(compileInfo_.l1Size / DB_SIZE, (alignMValue_ * alignKValue_ +
                                 alignKValue_ * alignNValue_) * args_.aDtypeSize + alignBiasSize);
    l0CanLoadBatch_ = (std::min({iterBatchL0A_, iterBatchL0B_, iterBatchL0C_, iterBatchL1_}) >= 1UL);
    if (args_.hasBias) {
        uint64_t iterBatchBiasBt = ops::FloorDiv(compileInfo_.btSize / DB_SIZE, alignNValue_ * DATA_SIZE_FP32);
        l0CanLoadBatch_ = l0CanLoadBatch_ && (iterBatchBiasBt >= 1UL);
    }
    constexpr static double defaultBalanceOfBatch = 0.8;
    if (!l0CanLoadBatch_) {
        // if l0 can not load multi part of batch, cal to avoid formulate unbalance issue.
        double avgIterBatch = static_cast<double>(batchInfo_->batchC) / static_cast<double>(compileInfo_.aicNum);
        double actualMaxIterBatch = static_cast<double>(ops::CeilDiv(ops::CeilDiv(batchInfo_->batchC, iterBatchL1_),
                                    compileInfo_.aicNum) * iterBatchL1_); // calculate one of the core load max batch
        double balanceRateOfBatch = avgIterBatch / actualMaxIterBatch; // calculate fb rate of batch
        if (balanceRateOfBatch < defaultBalanceOfBatch) { // balance of batch lower than 0.8
            OP_LOGI(args_.opName, "FormulteBalanceRate lower than 0.8, unable to enter in bmm iterbatch module");
            return false;
        }
    }
    OP_LOGI(args_.opName, "Enter BatchMatmul basicapi iterbatch module.");
    return true;
}

uint64_t BatchMatMulV3IterBatchBasicApiTiling::GetNumBlocks() const
{
    return compileInfo_.aicNum;
}

ge::graphStatus BatchMatMulV3IterBatchBasicApiTiling::DoOpTiling()
{
    constexpr uint64_t mmadCount = 8UL; // cube count which will cause issuequene
    constexpr uint64_t fullCopySize = 64 * 1024UL; // datasize moving once which can use full of bandwith
    if (mmadCount * (alignMValue_ * alignKValue_ + alignKValue_ * alignNValue_) * args_.aDtypeSize > fullCopySize) {
        runInfo_.iterBatchL1 = std::min({iterBatchL1_, mmadCount, ops::CeilDiv(batchInfo_->batchC,
                                         compileInfo_.aicNum)});
        runInfo_.iterBatchL0 = std::max(std::min({iterBatchL0A_, iterBatchL0B_, iterBatchL0C_, runInfo_.iterBatchL1,
                                        mmadCount}), 1UL);
    } else {
        runInfo_.iterBatchL1 = std::min({iterBatchL1_, ops::CeilDiv(batchInfo_->batchC, compileInfo_.aicNum)});
        runInfo_.iterBatchL0 = std::max(std::min({iterBatchL0A_, iterBatchL0B_, iterBatchL0C_, runInfo_.iterBatchL1}),
                                        1UL);
    }

    // transpose
    auto mat2ViewShape = context_->GetInputShape(1)->GetOriginShape();
    // 特殊处理3D非连续场景
    if (context_->InputIsView(1) && IsMat2TransposeNonContiguous(mat2ViewShape)) {
        const size_t mat2DimNum = mat2ViewShape.GetDimNum();
        size_t innerBatch = 0;
        innerBatch = mat2ViewShape[mat2DimNum - NUM_THREE];
        size_t upperBound = std::min(innerBatch, runInfo_.iterBatchL1);
        size_t iterBatchL1 = 1;
        for (size_t x = upperBound; x >= 1; --x) {
            if (innerBatch % x == 0) {
                iterBatchL1 = x;
                break;
            }
        }
        runInfo_.iterBatchL1 = iterBatchL1;
        runInfo_.innerBatch = innerBatch;
        runInfo_.iterBatchL0 =
            std::max(std::min({iterBatchL0A_, iterBatchL0C_, runInfo_.iterBatchL0, runInfo_.iterBatchL1}), 1UL);
    }

    if (l0CanLoadBatch_) { // l0 can load multi batch, baseM,N,K equals to aligned ori m,n,k
        runInfo_.baseM = alignMValue_;
        runInfo_.baseN = alignNValue_;
        runInfo_.baseK = alignKValue_;
    } else { // l0 cannot load multi batch, adjust baseM, baseN, baseK for l0
        // adjust basic block by l0a/l0b shape, choice small side of small matrix
        if ((alignMValue_ < alignNValue_) && (alignMValue_ > alignKValue_)) { // fix K from mat A, adjust m,n
            runInfo_.baseK = alignKValue_;
            runInfo_.baseM = std::min(ops::FloorDiv(compileInfo_.l0ASize / DB_SIZE / args_.aDtypeSize, runInfo_.baseK),
                                      alignMValue_);
            runInfo_.baseN = std::min(ops::FloorDiv(compileInfo_.l0BSize / DB_SIZE / args_.bDtypeSize, runInfo_.baseK),
                                      alignNValue_);
        } else if ((alignMValue_ < alignNValue_) && (alignMValue_ <= alignKValue_)) { // fix M from mat A, adjust K,N
            runInfo_.baseM = alignMValue_;
            runInfo_.baseK = std::min(ops::FloorDiv(compileInfo_.l0ASize / DB_SIZE / args_.aDtypeSize, runInfo_.baseM),
                                      alignKValue_);
            runInfo_.baseN = std::min(ops::FloorDiv(compileInfo_.l0BSize / DB_SIZE / args_.bDtypeSize, runInfo_.baseK),
                                      alignNValue_);
        } else if ((alignMValue_ >= alignNValue_) && (alignNValue_ > alignKValue_)) { // fix K from mat B, adjust M,N
            runInfo_.baseK = alignKValue_;
            runInfo_.baseN = std::min(ops::FloorDiv(compileInfo_.l0BSize / DB_SIZE / args_.bDtypeSize, runInfo_.baseK),
                                      alignNValue_);
            runInfo_.baseM = std::min(ops::FloorDiv(compileInfo_.l0ASize / DB_SIZE / args_.aDtypeSize, runInfo_.baseK),
                                      alignMValue_);
        } else if ((alignMValue_ >= alignNValue_) && (alignNValue_ <= alignKValue_)) { // fix N from mat B, adjust M,K
            runInfo_.baseN = alignNValue_;
            runInfo_.baseK = std::min(ops::FloorDiv(compileInfo_.l0BSize / DB_SIZE / args_.bDtypeSize, runInfo_.baseN),
                                      alignKValue_);
            runInfo_.baseM = std::min(ops::FloorDiv(compileInfo_.l0ASize / DB_SIZE / args_.aDtypeSize, runInfo_.baseK),
                                      alignMValue_);
        }
        //baseN满足C2大小
        if (args_.hasBias) {
            runInfo_.baseN = std::min(runInfo_.baseN, compileInfo_.btSize / DB_SIZE / DATA_SIZE_FP32);
        }
        // adjust basic block by l0c shape
        if (runInfo_.baseM >= runInfo_.baseN) {
            runInfo_.baseM = std::min(runInfo_.baseM, ops::FloorDiv(compileInfo_.l0CSize / DB_SIZE / NUM_FOUR,
                                      runInfo_.baseN));
        } else {
            runInfo_.baseN = std::min(runInfo_.baseN, ops::FloorDiv(compileInfo_.l0CSize / DB_SIZE / NUM_FOUR,
                                      runInfo_.baseM));
        }
        runInfo_.baseM = ops::FloorAlign(std::max(runInfo_.baseM, BASIC_BLOCK_SIZE_16), BASIC_BLOCK_SIZE_16);
        runInfo_.baseN = ops::FloorAlign(std::max(runInfo_.baseN, BASIC_BLOCK_SIZE_16), BASIC_BLOCK_SIZE_16);
        runInfo_.baseK = ops::FloorAlign(std::max(runInfo_.baseK, BASIC_BLOCK_SIZE_16), BASIC_BLOCK_SIZE_16);
    }

    // enable Fixpipe optimization
    if (args_.nValue * args_.bDtypeSize > BASIC_BLOCK_SIZE_256 &&
        args_.nValue % (BASIC_BLOCK_SIZE_256 / args_.bDtypeSize) != 0) {
        l0C2Out_ = MatMulV3L0C2Out::ND_FIXPIPE_1_2;
    }
    OP_LOGI(
        args_.opName,
        "In IterBatchBasicApi module, temp iterBatchL0A is %lu, temp iterBatchL0B is %lu, \
            temp iterBatchL0C is %lu, temp iterBatchL1 is %lu, after calculation actual runInfo_.iterBatchL0 is %lu, \
            runInfo_.iterBatchL1 is %lu, runInfo_.baseM is %lu, runInfo_.baseN is %lu",
        iterBatchL0A_, iterBatchL0B_, iterBatchL0C_, iterBatchL1_, runInfo_.iterBatchL0, runInfo_.iterBatchL1,
        runInfo_.baseM, runInfo_.baseN);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchMatMulV3IterBatchBasicApiTiling::GetTilingData(TilingResult& tiling) const
{
    return GetTilingDataImpl<BatchMatMulV3IterBatchBasicTilingData>(tiling);
}

} // namespace batch_matmul_v3_advanced
} // namespace optiling