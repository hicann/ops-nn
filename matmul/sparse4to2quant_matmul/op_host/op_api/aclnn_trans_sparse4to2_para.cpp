/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_sparse4to2quant_matmul_weight_nz.h"

#include <cstdlib>
#include <map>
#include <vector>
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "util/math_util.h"

using Ops::Base::CeilDiv;
using std::map;
using std::unordered_map;
using std::vector;

namespace {
static const size_t WEIGHT_SHAPE_DIM = 2;
static const int64_t SPARSE_MULTI = 2;
static const size_t SPARSE_INDEX_MULTI = 4;
static const size_t SPARSE_ATOMIC_SIZE = 8;
static const size_t INDEX_C0_SIZE = 8;
static const size_t NZ_DIM = 4;
static const int64_t NZ_C0_SIZE = 16;
static const int64_t NZ_K0_INT8_SIZE = 32;
struct CompressedParam {
    uint64_t firstIndex = 0;
    uint64_t secondIndex = 0;
    uint8_t weightIndexValue = 0;
};

// key of map is mask of 4 input value, value is {the fiest non-zero index, the second non-zero index, int8 index value}
static const unordered_map<uint8_t, CompressedParam> SPARSE_MAP{
    {0b0000, {0UL, 0UL, 0b0}},
    {0b0001, {0UL, 3UL, (0b10 << 2) | 0b00}},
    {0b0010, {2UL, 3UL, (0b00 << 2) | 0b10}},
    {0b0011, {2UL, 3UL, (0b10 << 2) | 0b10}},
    {0b0100, {1UL, 2UL, (0b00 << 2) | 0b01}},
    {0b0101, {1UL, 3UL, (0b10 << 2) | 0b01}},
    {0b0110, {1UL, 2UL, (0b01 << 2) | 0b01}},
    {0b1000, {0UL, 1UL, (0b00 << 2) | 0b00}},
    {0b1001, {0UL, 3UL, (0b10 << 2) | 0b00}},
    {0b1010, {0UL, 2UL, (0b01 << 2) | 0b00}},
    {0b1100, {0UL, 1UL, (0b00 << 2) | 0b00}}
};

static aclnnStatus CalcSparse4to2ParaSize(
    const aclIntArray* shape, uint64_t& sparseWeightSize, uint64_t& indexSize, vector<int64_t>& sparseWeightShape,
    vector<int64_t>& indexShape)
{
    const aclIntArray& weightShape = *shape;
    OP_CHECK(
        weightShape.Size() == WEIGHT_SHAPE_DIM, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dimension of shape must be 2"),
        return ACLNN_ERR_PARAM_INVALID);
    const int64_t shapeN = weightShape[0];
    const int64_t shapeK = weightShape[1];
    sparseWeightShape.push_back(CeilDiv(CeilDiv(shapeK, NZ_K0_INT8_SIZE), SPARSE_MULTI)); // ceil(K1 / 2)
    sparseWeightShape.push_back(CeilDiv(shapeN, NZ_C0_SIZE));                             // n1
    sparseWeightShape.push_back(NZ_C0_SIZE);                                              // n0
    sparseWeightShape.push_back(NZ_K0_INT8_SIZE);                                         // k0
    indexShape.push_back(sparseWeightShape[0]);
    indexShape.push_back(sparseWeightShape[1]);
    indexShape.push_back(NZ_C0_SIZE);
    indexShape.push_back(NZ_K0_INT8_SIZE / SPARSE_INDEX_MULTI); // k0 // 4
    sparseWeightSize = 1;
    for (auto value : sparseWeightShape) {
        sparseWeightSize *= value;
    }
    indexSize = sparseWeightSize / SPARSE_INDEX_MULTI;
    return ACLNN_SUCCESS;
}

void ReleaseMalloc(int8_t** sparseWeight, uint8_t** index, int64_t** sparseWeightShape, int64_t** indexShape)
{
    if (*sparseWeight != nullptr) {
        free(*sparseWeight);
        *sparseWeight = nullptr;
    }
    if (*index != nullptr) {
        free(*index);
        *index = nullptr;
    }
    if (*sparseWeightShape != nullptr) {
        free(*sparseWeightShape);
        *sparseWeightShape = nullptr;
    }
    if (*indexShape != nullptr) {
        free(*indexShape);
        *indexShape = nullptr;
    }
}

void ProcessWeightPattern(uint8_t& weightPattern, int8_t weightValue) {
    weightPattern <<= 1; // 用二进制的0和1分别来表示weight值为零/非零
    if (weightValue != 0) {
        weightPattern |= 1;
    }
}

bool ProcessWeightBlock(
    std::vector<int8_t>& sparseWeightBlock, std::vector<int8_t>& compressedWeightBlock, uint8_t& weightIndexBlock)
{
    // 单次处理8个weight数据，生成4个sparseWeight数据以及1个index
    uint8_t weightPattern0 = 0; // 前4个数pattern
    uint8_t weightPattern1 = 0; // 后4个数pattern
    for (size_t j = 0; j < SPARSE_INDEX_MULTI; j++) {
        // 低四位
        ProcessWeightPattern(weightPattern0, sparseWeightBlock[j]);
        // 高四位
        ProcessWeightPattern(weightPattern1, sparseWeightBlock[SPARSE_INDEX_MULTI + j]);
    }

    auto compressedParam0 = SPARSE_MAP.find(weightPattern0);
    auto compressedParam1 = SPARSE_MAP.find(weightPattern1);
    if (compressedParam0 == SPARSE_MAP.end() || compressedParam1 == SPARSE_MAP.end()) {
        std::cout << "compressedParam0 = " << weightPattern0 << std::endl;
        std::cout << "compressedParam1 = " << weightPattern1 << std::endl;
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Data of weight not satisfy 50%% sparsity rate.");
        return false;
    }
    compressedWeightBlock.push_back(sparseWeightBlock[compressedParam0->second.firstIndex]);
    compressedWeightBlock.push_back(sparseWeightBlock[compressedParam0->second.secondIndex]);
    compressedWeightBlock.push_back(sparseWeightBlock[compressedParam1->second.firstIndex + SPARSE_INDEX_MULTI]);
    compressedWeightBlock.push_back(sparseWeightBlock[compressedParam1->second.secondIndex + SPARSE_INDEX_MULTI]);
    weightIndexBlock =
        (compressedParam1->second.weightIndexValue << SPARSE_INDEX_MULTI) | compressedParam0->second.weightIndexValue;
    return true;
}

bool SetOneBlockDense(
    const int8_t* weight, int8_t* sparseWeight, uint8_t* index, int32_t validWeightCount)
{
    if (validWeightCount < 0) {
        return true;
    }
    uint8_t weightIndexBlock = 0;
    std::vector<int8_t> sparseWeightBlock;
    sparseWeightBlock.reserve(SPARSE_ATOMIC_SIZE); // 8个数为4选2的一组
    std::vector<int8_t> compressedWeightBlock;
    compressedWeightBlock.reserve(SPARSE_INDEX_MULTI); // 对应的压缩后的4个数
    for (uint64_t i = 0; i < static_cast<uint64_t>(validWeightCount); i++) {
        sparseWeightBlock.push_back(weight[i]);
    }
    for (uint64_t i = 0; i < SPARSE_ATOMIC_SIZE - validWeightCount; i++) {
        sparseWeightBlock.push_back(0);
    }
    if (!ProcessWeightBlock(sparseWeightBlock, compressedWeightBlock, weightIndexBlock)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Data of weight not satisfy 50%% spatsity rate.");
        return false;
    }
    for (uint64_t i = 0; i < SPARSE_INDEX_MULTI; i++) {
        sparseWeight[i] = compressedWeightBlock[i];
    }
    *index = weightIndexBlock;
    return true;
}

bool CalParseWeightAndIndex(
    const int8_t* weight, const aclIntArray* shape, int8_t* sparseWeight, uint8_t* index, int64_t* sparseWeightShape)
{
    int64_t nDimNzShape = sparseWeightShape[1] * sparseWeightShape[2];
    int64_t nDimNdShape = (*shape)[0];
    int64_t kDimNdShape = (*shape)[1];

    for (int64_t k1Loop = 0; k1Loop < sparseWeightShape[0]; k1Loop++) { // K1循环
        for (int64_t nLoop = 0; nLoop < nDimNzShape; nLoop++) {         // n1*n0循环
            if (nLoop >= nDimNdShape) {                                 // N维度的padding
                continue;
            }
            // weight中64个数对应sparseWeight一个C0, 8个数对应1个index
            for (int64_t k0Loop = 0; k0Loop < static_cast<int64_t>(INDEX_C0_SIZE); k0Loop++) {
                int64_t kNdOffset = k1Loop * NZ_K0_INT8_SIZE * SPARSE_MULTI + k0Loop * SPARSE_ATOMIC_SIZE;
                int32_t validWeightCount = kNdOffset + SPARSE_ATOMIC_SIZE >= static_cast<uint64_t>(kDimNdShape) ?
                                            (kDimNdShape - kNdOffset) : SPARSE_ATOMIC_SIZE;
                int64_t ndOffset = nLoop * kDimNdShape + kNdOffset;
                int64_t sparseWeightOffset = (k1Loop * nDimNzShape + nLoop) * NZ_K0_INT8_SIZE +
                                             k0Loop * SPARSE_INDEX_MULTI;
                int64_t sparseIndexOffset = (k1Loop * nDimNzShape + nLoop) * INDEX_C0_SIZE + k0Loop;

                CHECK_RET(
                    SetOneBlockDense(&weight[ndOffset], &sparseWeight[sparseWeightOffset], &index[sparseIndexOffset],
                                     validWeightCount),
                    false);
            }
        }
    }

    return true;
}

void SetOutputDims(const vector<int64_t>& shape, int64_t* dims)
{
    // 恒成立
    if (dims != nullptr) {
        for (uint64_t i = 0; i < shape.size(); i++) {
            dims[i] = shape[i];
        }
    }
}
} // namespace

aclnnStatus aclnnTransSparse4to2Para(
    const int8_t* weight, aclIntArray* shape, int8_t** sparseWeight, int64_t** sparseWeightDims,
    uint64_t* sparseWeightDimsNum, uint8_t** index, int64_t** indexDims, uint64_t* indexDimsNum)
{
    CHECK_RET(shape != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(weight != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(sparseWeight != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(sparseWeightDims != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(sparseWeightDimsNum != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(index != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(indexDims != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(indexDimsNum != nullptr, ACLNN_ERR_INNER_NULLPTR);
    uint64_t sparseWeightSize = 0;
    uint64_t indexSize = 0;
    vector<int64_t> sparseWShape;
    vector<int64_t> idxShape;
    // 计算输出的size以及shape
    auto ret = CalcSparse4to2ParaSize(shape, sparseWeightSize, indexSize, sparseWShape, idxShape);
    OP_CHECK(
        ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Calculate size of sparseWeight and index failed"),
        return ret);
    // align
    *sparseWeight = (int8_t*)calloc(sparseWeightSize, sizeof(int8_t));
    OP_CHECK(
        *sparseWeight != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The *sparseWeight is nullptr."),
        return ACLNN_ERR_INNER_NULLPTR);
    *index = static_cast<uint8_t*>(calloc(indexSize, sizeof(uint8_t)));
    if (*index == nullptr) {
        ReleaseMalloc(sparseWeight, index, sparseWeightDims, indexDims);
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The *index is nullptr.");
        return ACLNN_ERR_INNER_NULLPTR;
    }
    *sparseWeightDims = static_cast<int64_t*>(calloc(sparseWShape.size(), sizeof(int64_t)));
    if (*sparseWeightDims == nullptr) {
        ReleaseMalloc(sparseWeight, index, sparseWeightDims, indexDims);
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The *sparseWeightDims is nullptr.");
        return ACLNN_ERR_INNER_NULLPTR;
    }
    *indexDims = static_cast<int64_t*>(calloc(idxShape.size(), sizeof(int64_t)));
    if (*indexDims == nullptr) {
        ReleaseMalloc(sparseWeight, index, sparseWeightDims, indexDims);
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The *indexDims is nullptr.");
        return ACLNN_ERR_INNER_NULLPTR;
    }
    SetOutputDims(sparseWShape, *sparseWeightDims);
    SetOutputDims(idxShape, *indexDims);
    if (!CalParseWeightAndIndex(weight, shape, *sparseWeight, *index, *sparseWeightDims)) {
        ReleaseMalloc(sparseWeight, index, sparseWeightDims, indexDims);
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Calculate sparseWeight and index failed");
        return ACLNN_ERR_PARAM_INVALID;
    }

    *sparseWeightDimsNum = sparseWShape.size();
    *indexDimsNum = idxShape.size();
    return ACLNN_SUCCESS;
}