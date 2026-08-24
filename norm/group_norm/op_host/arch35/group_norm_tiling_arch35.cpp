/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file group_norm_tiling_arch35.cpp
 * \brief
 */
#include "group_norm_tiling_arch35.h"
#include "op_api/runtime2_util_nn.h"
#include "error_util.h"
#include <nlohmann/json.hpp>
#include "op_host/tiling_templates_registry.h"

using namespace ge;
using namespace std;

namespace optiling {
static const int32_t DIM_0 = 0;
static const int32_t DIM_1 = 1;
static const int32_t MIN_LEN = 2;
static const int32_t INDEX_NUM_GROUPS = 0;
static const int32_t INDEX_EPSILON = 2;
static const int32_t INDEX_X = 0;
static const int32_t BYTES_FOR_ALIGN = 1024;
static const int32_t FLOAT32_BYTES = 4;
static const int64_t INPUT_IDX_X = 0;
static const int64_t INPUT_IDX_GAMMA = 1;
static const int64_t INPUT_IDX_BETA = 2;
static const int64_t PROCESSSIZE = 8192;
static const int64_t RESERVED_WORKSPACE_SIZE_950 = 16L * 1024L * 1024L;
static const int64_t FOUR_BUFFER = 4;
static const int64_t BUFFER_NUM = 2;
static const int64_t DOUBLE_BUFFER = 2;
static const int64_t DICHOTOMY_ADD_COEFF = 2;
static const int64_t ULONG_BIT_LEN = 64;
static const int64_t MAX_CHANNEL_SIZE = 4096;
static const int64_t MAX_NUM_PER_CORE = 2048;
static const float DEFAULT_EPS = 1e-4F;

inline std::unique_ptr<nlohmann::json> GetCompileInfoJson(gert::TilingParseContext* context)
{
    auto json_str = context->GetCompiledJson();
    OP_CHECK_IF(json_str == nullptr, OP_LOGE(context->GetNodeName(), "json_str is nullptr!"), return nullptr);
    std::unique_ptr<nlohmann::json> parsed_object_cinfo = std::make_unique<nlohmann::json>(
        nlohmann::json::parse(json_str));
    return parsed_object_cinfo;
}

struct WelfordTilingInitResult {
    int64_t loopNum{0};
    int64_t loopTail{0};
    int64_t processSize{0};
    int64_t innerLoopNum{0};
    int64_t innerLoopTail{0};
    int64_t hwNum{0};
    int64_t hwNumAlign{0};
    bool checkResult{false};
};

inline static ge::graphStatus GroupNormSetTilingData(gert::TilingContext* context, GroupNormTilingData& tilingData)
{
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

inline static int64_t CeilDiv(int64_t value, int64_t factor)
{
    if (factor == 0) {
        return value;
    }
    return (value + factor - 1) / factor;
}

inline static int64_t DownAlign(int64_t a, int64_t b)
{
    if (b == 0) {
        return a;
    }
    return (a / b) * b;
}

inline static int64_t RoundUp(int64_t a, int64_t b) { return CeilDiv(a, b) * b; }

static ge::graphStatus CheckInputXShape(const gert::TilingContext* context, const gert::Shape& xShape)
{
    size_t xDims = xShape.GetDimNum();
    OP_CHECK_IF(xDims < MIN_LEN,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "x", std::to_string(xDims).c_str(),
                                             "greater than or equal to 2"),
                return ge::GRAPH_FAILED);
    for (size_t i = 0; i < xDims; i++) {
        int64_t curDim = xShape.GetDim(i);
        OP_CHECK_IF((curDim < 0),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        context->GetNodeName(), "x", Ops::Base::ToString(xShape).c_str(),
                        ("The dim[" + std::to_string(i) + "] of x must be non-negative, got " + std::to_string(curDim))
                            .c_str()),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckInputParams(const gert::TilingContext* context)
{
    // 校验输入x的shape和数据类型。
    auto inputX = context->GetInputTensor(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX);
    auto xDesc = context->GetInputDesc(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    auto xDtype = xDesc->GetDataType();
    OP_CHECK_IF((xDtype != ge::DT_FLOAT16 && xDtype != ge::DT_FLOAT),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "x",
                                                      ge::TypeUtils::DataTypeToSerialString(xDtype).c_str(),
                                                      "GroupNorm only supports float16 and float32"),
                return ge::GRAPH_FAILED);
    int64_t xDtypeSize = static_cast<int64_t>(ge::GetSizeByDataType(xDtype));
    OP_CHECK_IF((xDtypeSize <= 0),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "x",
                                                      ge::TypeUtils::DataTypeToSerialString(xDtype).c_str(),
                                                      "The dtype size of x must be greater than 0"),
                return ge::GRAPH_FAILED);
    auto xShape = inputX->GetStorageShape();
    if (CheckInputXShape(context, xShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    int64_t channel = xShape.GetDim(DIM_1);
    OP_CHECK_IF(channel <= 0,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "x", Ops::Base::ToString(xShape).c_str(),
                                                      "The channel dimension of x must be greater than 0"),
                return ge::GRAPH_FAILED);

    // 校验gamma和beta的shape及数据类型。
    auto gammaShapePtr = context->GetInputShape(INPUT_IDX_GAMMA);
    OP_CHECK_NULL_WITH_CONTEXT(context, gammaShapePtr);
    auto gammaShape = gammaShapePtr->GetStorageShape();
    OP_CHECK_IF(gammaShape.GetDimNum() != 1,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "gamma",
                                                         std::to_string(gammaShape.GetDimNum()).c_str(),
                                                         "The shape dim of gamma must be 1"),
                return ge::GRAPH_FAILED);
    int64_t gammaSizes = gammaShape.GetDim(DIM_0);
    OP_CHECK_IF(gammaSizes != channel,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    context->GetNodeName(), "gamma", Ops::Base::ToString(gammaShape).c_str(),
                    ("The shape of gamma must be the same as channel(dim[1] of input x) size, "
                     "got gamma size = " +
                     std::to_string(gammaSizes) + ", channel = " + std::to_string(channel))
                        .c_str()),
                return ge::GRAPH_FAILED);
    auto betaShapePtr = context->GetInputShape(INPUT_IDX_BETA);
    OP_CHECK_NULL_WITH_CONTEXT(context, betaShapePtr);
    auto betaShape = betaShapePtr->GetStorageShape();
    OP_CHECK_IF(betaShape.GetDimNum() != 1,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "beta",
                                                         std::to_string(betaShape.GetDimNum()).c_str(),
                                                         "The shape dim of beta must be 1"),
                return ge::GRAPH_FAILED);
    int64_t betaSizes = betaShape.GetDim(DIM_0);
    OP_CHECK_IF(betaSizes != channel,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    context->GetNodeName(), "beta", Ops::Base::ToString(betaShape).c_str(),
                    ("The shape dim of beta should be 1, and the shape of beta must be the same as "
                     "channel(dim[1] of input x) size, got beta size = " +
                     std::to_string(betaSizes) + ", channel = " + std::to_string(channel))
                        .c_str()),
                return ge::GRAPH_FAILED);
    auto gammaDtypePtr = context->GetInputDesc(INPUT_IDX_GAMMA);
    OP_CHECK_NULL_WITH_CONTEXT(context, gammaDtypePtr);
    auto gammaDtype = gammaDtypePtr->GetDataType();
    int64_t gammaDtypeSize = static_cast<int64_t>(ge::GetSizeByDataType(gammaDtype));
    auto betaDtypePtr = context->GetInputDesc(INPUT_IDX_BETA);
    OP_CHECK_NULL_WITH_CONTEXT(context, betaDtypePtr);
    auto betaDtype = betaDtypePtr->GetDataType();
    int64_t betaDtypeSize = static_cast<int64_t>(ge::GetSizeByDataType(betaDtype));
    OP_CHECK_IF((gammaDtypeSize <= 0 || gammaDtypeSize != betaDtypeSize),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "gamma",
                                                      (ge::TypeUtils::DataTypeToSerialString(gammaDtype)).c_str(),
                                                      "The datatype size of gamma must be greater than or equal to 0, "
                                                      "and the dtype of gamma must be same as beta."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((xDtype != gammaDtype || xDtype != betaDtype),
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context->GetNodeName(), "x, gamma, beta",
                                                       (ge::TypeUtils::DataTypeToSerialString(xDtype) + ", " +
                                                        ge::TypeUtils::DataTypeToSerialString(gammaDtype) + ", " +
                                                        ge::TypeUtils::DataTypeToSerialString(betaDtype))
                                                           .c_str(),
                                                       "The dtype of gamma and beta must be the same as x"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAttrParams(const gert::TilingContext* context)
{
    auto inputX = context->GetInputTensor(INPUT_IDX_X);
    auto xShape = inputX->GetStorageShape();
    int64_t channel = xShape.GetDim(DIM_1);
    // 校验分组数及通道整除关系。
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* numGroups = attrs->GetAttrPointer<int64_t>(INDEX_NUM_GROUPS);
    OP_CHECK_NULL_WITH_CONTEXT(context, numGroups);
    const float* epsilon = attrs->GetAttrPointer<float>(INDEX_EPSILON);
    OP_CHECK_IF(epsilon != nullptr && !(*epsilon > 0.0F),
                OP_LOGE(context->GetNodeName(), "eps must be greater than 0, got %f", *epsilon),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (*numGroups <= 0),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "num_groups", std::to_string(*numGroups).c_str(),
                                              "num_groups must be bigger than 0."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (channel % *numGroups != 0),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "num_groups", std::to_string(*numGroups).c_str(),
                                              ("channel(dim[1] of input x) must be integer multiples of num_groups, "
                                               "got channel = " +
                                               std::to_string(channel) + ", num_groups = " + std::to_string(*numGroups))
                                                  .c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static int64_t GetOptionalInputTensorSize(const gert::TilingContext* context, int64_t index, int64_t specifiedValue = 0)
{
    auto tensorDesc = context->GetInputDesc(index);
    if (tensorDesc == nullptr) {
        return 0;
    }
    auto compileInfo = context->GetCompileInfo<GroupNormCompileInfo>();
    int64_t blockSize = compileInfo->blockSize;
    auto dtypeSize = ge::GetSizeByDataType(tensorDesc->GetDataType());
    if (specifiedValue != 0) {
        return RoundUp(specifiedValue * static_cast<int64_t>(dtypeSize), blockSize);
    }

    auto storageShape = context->GetInputShape(index);
    OP_CHECK_NULL_WITH_CONTEXT(context, storageShape);
    auto shape = storageShape->GetStorageShape();
    int64_t num = 1;
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        num = num * shape.GetDim(i);
    }
    auto numUbSize = RoundUp(num * static_cast<int64_t>(dtypeSize), blockSize);
    return numUbSize;
}

static void GetDichotomyAddParams(const gert::TilingContext* context, int64_t r, int64_t& power, int64_t& dichotomyK,
                                  int64_t& extraSize, int64_t& lastNum)
{
    power = 0;
    dichotomyK = 0;
    extraSize = 0;
    lastNum = 0;
    if (r <= 0) {
        return;
    }
    auto compileInfo = context->GetCompileInfo<GroupNormCompileInfo>();
    int64_t vl = compileInfo->vectorLength / FLOAT32_BYTES;
    int64_t blockSize = compileInfo->blockSize;
    int64_t basePower = (1L << (ULONG_BIT_LEN - 1 - __builtin_clzl(static_cast<uint64_t>(r))));
    power = basePower == r ? basePower / DICHOTOMY_ADD_COEFF : basePower;
    int64_t extraOriSize = power / vl;
    extraSize = RoundUp(extraOriSize * FLOAT32_BYTES, blockSize);
    if (extraOriSize < vl) {
        lastNum = extraOriSize;
        return;
    }
    int64_t totalNum = extraOriSize / vl;
    int64_t base = 1;
    lastNum = vl;
    while (base < totalNum) {
        dichotomyK++;
        base *= DICHOTOMY_ADD_COEFF;
    }
}

static ge::graphStatus SetAttrParams(const gert::TilingContext* context, GroupNormTilingData& tilingData)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* numGroups = attrs->GetAttrPointer<int64_t>(INDEX_NUM_GROUPS);
    OP_CHECK_NULL_WITH_CONTEXT(context, numGroups);
    const float* epsilonPtr = attrs->GetAttrPointer<float>(INDEX_EPSILON);
    float eps = epsilonPtr == nullptr ? DEFAULT_EPS : *epsilonPtr;
    tilingData.set_numGroups(*numGroups);
    tilingData.set_epsilon(eps);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetTilingParams(const gert::TilingContext* context, GroupNormTilingData& tilingData)
{
    auto inputX = context->GetInputTensor(INPUT_IDX_X);
    auto xShape = inputX->GetStorageShape();
    int64_t hwNum = 1;
    size_t xDims = xShape.GetDimNum();
    for (size_t i = 2; i < xDims; i++) {
        hwNum = hwNum * xShape.GetDim(i);
    }
    tilingData.set_shapeC(xShape.GetDim(DIM_1));
    tilingData.set_shapeD(xShape.GetDim(DIM_1) / tilingData.get_numGroups());
    tilingData.set_hwNum(hwNum);
    tilingData.set_elemNum(tilingData.get_shapeD() * hwNum);
    tilingData.set_processSize(PROCESSSIZE);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetBlockTiling(const gert::TilingContext* context, GroupNormTilingData& tilingData)
{
    auto compileInfo = context->GetCompileInfo<GroupNormCompileInfo>();
    auto inputX = context->GetInputTensor(INPUT_IDX_X);
    auto xShape = inputX->GetStorageShape();
    int64_t shapeN = xShape.GetDim(DIM_0);
    tilingData.set_numPerCore(CeilDiv(shapeN * tilingData.get_numGroups(), compileInfo->coreNum));
    tilingData.set_realCoreNum(CeilDiv(shapeN * tilingData.get_numGroups(), tilingData.get_numPerCore()));
    tilingData.set_numLastCore(shapeN * tilingData.get_numGroups() -
                               tilingData.get_numPerCore() * (tilingData.get_realCoreNum() - 1));
    return ge::GRAPH_SUCCESS;
}

static void SetUbTiling(GroupNormTilingData& tilingData)
{
    tilingData.set_loopNum(CeilDiv(tilingData.get_elemNum(), tilingData.get_processSize()));
    tilingData.set_loopTail(tilingData.get_elemNum() - tilingData.get_processSize() * (tilingData.get_loopNum() - 1));
    tilingData.set_innerLoopNum(CeilDiv(tilingData.get_hwNum(), tilingData.get_processSize()));
    tilingData.set_innerLoopTail(tilingData.get_hwNum() -
                                 tilingData.get_processSize() * (tilingData.get_innerLoopNum() - 1));
}

// 按归约轴和通道轴的UB占用选择TwoPass或Welford模板。
static void SetTilingKey4Ascend950(const gert::TilingContext* context, int64_t& maxReduceCount, int64_t& ubRemain,
                                   bool& isReduceFullLoad, GroupNormTilingData& tilingData)
{
    auto compileInfo = context->GetCompileInfo<GroupNormCompileInfo>();
    int64_t ubSize = compileInfo->ubSize;
    int64_t blockSize = compileInfo->blockSize;
    int64_t reduceCount = tilingData.get_shapeD() * tilingData.get_hwNum();
    int64_t gammaUbSize = GetOptionalInputTensorSize(context, INPUT_IDX_GAMMA);
    int64_t betaUbSize = GetOptionalInputTensorSize(context, INPUT_IDX_BETA);
    int64_t realNumPerCore = std::min(
        MAX_NUM_PER_CORE, static_cast<int64_t>(std::max(tilingData.get_numPerCore(), tilingData.get_numLastCore())));
    int64_t meanUbSize = RoundUp(realNumPerCore * FLOAT32_BYTES, blockSize);
    int64_t rstdUbSize = RoundUp(realNumPerCore * FLOAT32_BYTES, blockSize);

    int64_t otherUbSize = gammaUbSize + betaUbSize + meanUbSize + rstdUbSize;
    int64_t xDtypeSize = static_cast<int64_t>(ge::GetSizeByDataType(context->GetInputDesc(INPUT_IDX_X)->GetDataType()));
    int64_t meanUbExtraSize = 0;
    int64_t varianceUbExtraSize = RoundUp(realNumPerCore * xDtypeSize, blockSize);
    if (xDtypeSize != FLOAT32_BYTES) {
        meanUbExtraSize = RoundUp(realNumPerCore * xDtypeSize, blockSize);
    }
    otherUbSize = otherUbSize + meanUbExtraSize + varianceUbExtraSize;
    int64_t dichotomyAddPower = 0;
    int64_t dichotomyAddK = 0;
    int64_t dichotomyAddExtraSize = 0;
    int64_t dichotomyAddLastNum = 0;
    GetDichotomyAddParams(context, reduceCount, dichotomyAddPower, dichotomyAddK, dichotomyAddExtraSize,
                          dichotomyAddLastNum);
    otherUbSize += dichotomyAddExtraSize;

    ubRemain = ubSize <= otherUbSize ? 0 : ubSize - otherUbSize;
    OP_CHECK_IF((xDtypeSize == 0), OP_LOGE(context->GetNodeName(), "xDtypeSize is zero."), return);
    maxReduceCount = (ubRemain / (DOUBLE_BUFFER * BUFFER_NUM)) / xDtypeSize;

    if (maxReduceCount > reduceCount) {
        isReduceFullLoad = true;
        tilingData.set_tilingKey(static_cast<int64_t>(GroupNormTilingKey::TILINGKEY_TWOPASS_PERF));
        return;
    }
    bool isLargeChannel = static_cast<int64_t>(tilingData.get_shapeC()) > MAX_CHANNEL_SIZE;
    int64_t newUbRemain = ubRemain;
    // 大通道场景按单组大小重算gamma和beta占用。
    if (isLargeChannel) {
        int64_t gammaSplitUbSize = GetOptionalInputTensorSize(context, INPUT_IDX_GAMMA, tilingData.get_shapeD());
        int64_t betaSplitUbSize = GetOptionalInputTensorSize(context, INPUT_IDX_BETA, tilingData.get_shapeD());
        otherUbSize = otherUbSize - gammaUbSize - betaUbSize + gammaSplitUbSize + betaSplitUbSize;
        newUbRemain = ubSize <= otherUbSize ? 0 : ubSize - otherUbSize;
        int64_t newMaxReduceCount = (newUbRemain / (DOUBLE_BUFFER * BUFFER_NUM)) / xDtypeSize;
        if (newMaxReduceCount > reduceCount) {
            isReduceFullLoad = true;
            maxReduceCount = newMaxReduceCount;
            ubRemain = newUbRemain;
            tilingData.set_tilingKey(static_cast<int64_t>(GroupNormTilingKey::TILINGKEY_TWOPASS_GENERALIZED));
            return;
        }
    }
    // 归约轴无法全载时释放二分累加空间并切换Welford模板。
    isReduceFullLoad = false;
    int64_t meanAndRstdSize = meanUbSize + rstdUbSize + meanUbExtraSize + varianceUbExtraSize;
    if (isLargeChannel) {
        ubRemain = ubSize - meanAndRstdSize;
        tilingData.set_tilingKey(static_cast<int64_t>(GroupNormTilingKey::TILINGKEY_WELFORD_GENERALIZED));
    } else {
        ubRemain = ubSize - meanAndRstdSize - gammaUbSize - betaUbSize;
        tilingData.set_tilingKey(static_cast<int64_t>(GroupNormTilingKey::TILINGKEY_WELFORD_PERF));
    }
    maxReduceCount = (ubRemain / (DOUBLE_BUFFER * BUFFER_NUM)) / xDtypeSize;
}

static void SetDichotomyAddParams(const gert::TilingContext* context, GroupNormTilingData& tilingData)
{
    int64_t reduceCount = tilingData.get_shapeD() * tilingData.get_hwNum();
    int64_t dichotomyAddPower = 0;
    int64_t dichotomyAddK = 0;
    int64_t dichotomyAddExtraSize = 0;
    int64_t dichotomyAddLastNum = 0;
    GetDichotomyAddParams(context, reduceCount, dichotomyAddPower, dichotomyAddK, dichotomyAddExtraSize,
                          dichotomyAddLastNum);
    tilingData.set_dichotomyAddPower(dichotomyAddPower);
    tilingData.set_dichotomyAddK(dichotomyAddK);
    tilingData.set_dichotomyAddLastNum(dichotomyAddLastNum);
}

static void SetWelfordParallelN(const gert::TilingContext* context, int64_t xDtypeSize, int64_t ubRemain,
                                GroupNormTilingData& tilingData)
{
    tilingData.set_parallelN(0);
    auto compileInfo = context->GetCompileInfo<GroupNormCompileInfo>();
    int64_t blockSize = compileInfo->blockSize;
    OP_CHECK_IF((xDtypeSize <= 0), OP_LOGE(context->GetNodeName(), "xDtypeSize must be positive."), return);
    int64_t coeff = FLOAT32_BYTES / xDtypeSize;
    int64_t totalNum = BUFFER_NUM * (coeff + 1);
    int64_t welfordBase = blockSize / xDtypeSize;
    OP_CHECK_IF((totalNum <= 0 || welfordBase <= 0),
                OP_LOGE(context->GetNodeName(), "Invalid Welford tiling parameters."), return);
    int64_t maxParallelN = DownAlign((ubRemain / xDtypeSize) / totalNum, welfordBase);
    OP_CHECK_IF((maxParallelN <= 0),
                OP_LOGE(context->GetNodeName(), "UB is insufficient for Welford parallel processing."), return);

    int64_t dichotomyAddPower = 0;
    int64_t dichotomyAddK = 0;
    int64_t dichotomyAddExtraSize = 0;
    int64_t dichotomyAddLastNum = 0;
    GetDichotomyAddParams(context, maxParallelN, dichotomyAddPower, dichotomyAddK, dichotomyAddExtraSize,
                          dichotomyAddLastNum);
    int64_t ubCurUse = maxParallelN * BUFFER_NUM * xDtypeSize + dichotomyAddExtraSize +
                       maxParallelN * BUFFER_NUM * FLOAT32_BYTES;
    while (ubCurUse > ubRemain && maxParallelN > welfordBase) {
        maxParallelN -= welfordBase;
        GetDichotomyAddParams(context, maxParallelN, dichotomyAddPower, dichotomyAddK, dichotomyAddExtraSize,
                              dichotomyAddLastNum);
        ubCurUse = maxParallelN * BUFFER_NUM * xDtypeSize + dichotomyAddExtraSize +
                   maxParallelN * BUFFER_NUM * FLOAT32_BYTES;
    }
    OP_CHECK_IF((ubCurUse > ubRemain),
                OP_LOGE(context->GetNodeName(), "UB is insufficient for a Welford processing block."), return);

    if (maxParallelN > tilingData.get_elemNum()) {
        maxParallelN = tilingData.get_elemNum();
        OP_CHECK_IF((maxParallelN <= 0),
                    OP_LOGE(context->GetNodeName(), "Element count must be positive for Welford tiling."), return);
        GetDichotomyAddParams(context, maxParallelN, dichotomyAddPower, dichotomyAddK, dichotomyAddExtraSize,
                              dichotomyAddLastNum);
    }
    tilingData.set_dichotomyAddPower(dichotomyAddPower);
    tilingData.set_dichotomyAddK(dichotomyAddK);
    tilingData.set_dichotomyAddLastNum(dichotomyAddLastNum);
    tilingData.set_parallelN(maxParallelN);
}

static void SetUbTiling4TwoPass(const gert::TilingContext* context, GroupNormTilingData& tilingData,
                                int64_t maxReduceCount, int64_t xDtypeSize)
{
    auto compileInfo = context->GetCompileInfo<GroupNormCompileInfo>();
    int64_t blockSize = compileInfo->blockSize;
    int64_t elemNum = tilingData.get_elemNum();
    OP_CHECK_IF((xDtypeSize == 0), OP_LOGE(context->GetNodeName(), "xDtypeSize is zero."), return);
    int64_t elemNumAlign = RoundUp(elemNum, blockSize / xDtypeSize);
    SetDichotomyAddParams(context, tilingData);
    OP_CHECK_IF((elemNumAlign == 0), OP_LOGE(context->GetNodeName(), "ElemNumAlign is zero."), return);
    int64_t count = maxReduceCount / elemNumAlign;
    int64_t processSize = count * elemNumAlign;
    tilingData.set_processSize(processSize);
}

static WelfordTilingInitResult InitWelfordTilingCommon(const gert::TilingContext* context,
                                                       GroupNormTilingData& tilingData, int64_t blockSize,
                                                       int64_t xDtypeSize)
{
    WelfordTilingInitResult result{};
    result.hwNum = tilingData.get_hwNum();
    OP_CHECK_IF((xDtypeSize == 0), OP_LOGE(context->GetNodeName(), "xDtypeSize is zero."), return result);
    result.hwNumAlign = RoundUp(result.hwNum, blockSize / xDtypeSize);
    OP_CHECK_IF((result.hwNumAlign == 0), OP_LOGE(context->GetNodeName(), "HwNumAlign is zero."), return result);
    result.checkResult = true;
    return result;
}

static void SetUbTiling4WelfordPerf(const gert::TilingContext* context, GroupNormTilingData& tilingData,
                                    int64_t maxReduceCount, int64_t ubRemain, int64_t xDtypeSize)
{
    auto compileInfo = context->GetCompileInfo<GroupNormCompileInfo>();
    int64_t blockSize = compileInfo->blockSize;
    SetWelfordParallelN(context, xDtypeSize, ubRemain, tilingData);
    WelfordTilingInitResult result = InitWelfordTilingCommon(context, tilingData, blockSize, xDtypeSize);
    OP_CHECK_IF((result.checkResult == false), OP_LOGE(context->GetNodeName(), "InitWelfordTilingCommon Failed."),
                return);
    int64_t count = maxReduceCount / result.hwNumAlign;
    if (count >= 1) {
        result.loopNum = CeilDiv(tilingData.get_shapeD(), count);
        result.loopTail = (tilingData.get_shapeD() - (result.loopNum - 1) * count) * result.hwNumAlign;
        result.processSize = count * result.hwNumAlign;
        result.innerLoopNum = 1;
    } else {
        auto maxReduceCountDownAlign = DownAlign(maxReduceCount, blockSize / xDtypeSize);
        result.innerLoopNum = CeilDiv(result.hwNum, maxReduceCountDownAlign);
        result.innerLoopTail = result.hwNum - maxReduceCountDownAlign * (result.innerLoopNum - 1);
        result.processSize = maxReduceCountDownAlign;
        result.loopNum = tilingData.get_shapeD();
        result.loopTail = 1;
    }
    tilingData.set_loopNum(result.loopNum);
    tilingData.set_loopTail(result.loopTail);
    tilingData.set_processSize(result.processSize);
    tilingData.set_innerLoopNum(result.innerLoopNum);
    tilingData.set_innerLoopTail(result.innerLoopTail);
}
static void SetUbTiling4WelfordGeneralized(const gert::TilingContext* context, GroupNormTilingData& tilingData,
                                           int64_t ubRemain, int64_t xDtypeSize)
{
    auto compileInfo = context->GetCompileInfo<GroupNormCompileInfo>();
    int64_t blockSize = compileInfo->blockSize;
    WelfordTilingInitResult result = InitWelfordTilingCommon(context, tilingData, blockSize, xDtypeSize);
    OP_CHECK_IF((result.checkResult == false), OP_LOGE(context->GetNodeName(), "InitWelfordTilingCommon Failed."),
                return);
    int64_t maxReduceCount = (ubRemain / (DOUBLE_BUFFER * BUFFER_NUM)) / xDtypeSize;
    int64_t count = maxReduceCount / result.hwNumAlign;
    int64_t gammaRealSize = GetOptionalInputTensorSize(context, INPUT_IDX_GAMMA, count);
    int64_t betaRealSize = GetOptionalInputTensorSize(context, INPUT_IDX_BETA, count);
    int64_t curUbSize = gammaRealSize + betaRealSize +
                        count * result.hwNumAlign * xDtypeSize * BUFFER_NUM * DOUBLE_BUFFER;
    while (curUbSize > ubRemain && count >= 1) {
        count--;
        gammaRealSize = GetOptionalInputTensorSize(context, INPUT_IDX_GAMMA, count);
        betaRealSize = GetOptionalInputTensorSize(context, INPUT_IDX_BETA, count);
        curUbSize = gammaRealSize + betaRealSize + count * result.hwNumAlign * xDtypeSize * BUFFER_NUM * DOUBLE_BUFFER;
    }
    if (count >= 1) {
        result.loopNum = CeilDiv(tilingData.get_shapeD(), count);
        result.loopTail = (tilingData.get_shapeD() - (result.loopNum - 1) * count) * result.hwNumAlign;
        result.processSize = count * result.hwNumAlign;
        result.innerLoopNum = 1;
        ubRemain = ubRemain - gammaRealSize - betaRealSize;
    } else {
        gammaRealSize = blockSize;
        betaRealSize = blockSize;
        ubRemain = ubRemain - gammaRealSize - betaRealSize;
        maxReduceCount = (ubRemain / (DOUBLE_BUFFER * BUFFER_NUM)) / xDtypeSize;
        int64_t maxReduceCountDownAlign = DownAlign(maxReduceCount, blockSize / xDtypeSize);
        result.innerLoopNum = CeilDiv(result.hwNum, maxReduceCountDownAlign);
        result.innerLoopTail = result.hwNum - maxReduceCountDownAlign * (result.innerLoopNum - 1);
        result.processSize = maxReduceCountDownAlign;
        result.loopNum = tilingData.get_shapeD();
        result.loopTail = 1;
    }
    SetWelfordParallelN(context, xDtypeSize, ubRemain, tilingData);
    tilingData.set_loopNum(result.loopNum);
    tilingData.set_loopTail(result.loopTail);
    tilingData.set_processSize(result.processSize);
    tilingData.set_innerLoopNum(result.innerLoopNum);
    tilingData.set_innerLoopTail(result.innerLoopTail);
}

static void SetUbTiling4Welford(const gert::TilingContext* context, GroupNormTilingData& tilingData,
                                int64_t maxReduceCount, int64_t ubRemain, int64_t xDtypeSize)
{
    if (tilingData.get_tilingKey() == static_cast<int64_t>(GroupNormTilingKey::TILINGKEY_WELFORD_PERF)) {
        return SetUbTiling4WelfordPerf(context, tilingData, maxReduceCount, ubRemain, xDtypeSize);
    }
    return SetUbTiling4WelfordGeneralized(context, tilingData, ubRemain, xDtypeSize);
}

static void SetUbTiling4Ascend950(const gert::TilingContext* context, int64_t maxReduceCount, int64_t ubRemain,
                                  bool isReduceFullLoad, GroupNormTilingData& tilingData)
{
    auto compileInfo = context->GetCompileInfo<GroupNormCompileInfo>();
    int64_t ubSize = compileInfo->ubSize;
    int64_t xDtypeSize = static_cast<int64_t>(ge::GetSizeByDataType(context->GetInputDesc(INPUT_IDX_X)->GetDataType()));
    tilingData.set_ubSize(ubSize);
    if (!isReduceFullLoad) {
        SetUbTiling4Welford(context, tilingData, maxReduceCount, ubRemain, xDtypeSize);
    } else {
        SetUbTiling4TwoPass(context, tilingData, maxReduceCount, xDtypeSize);
    }
}

static ge::graphStatus SetTilingForAscend950(const gert::TilingContext* context, GroupNormTilingData& tilingData)
{
    int64_t maxReduceCount = 0;
    int64_t ubRemain = 0;
    bool reduceFullLoad = false;
    SetTilingKey4Ascend950(context, maxReduceCount, ubRemain, reduceFullLoad, tilingData);
    SetUbTiling4Ascend950(context, maxReduceCount, ubRemain, reduceFullLoad, tilingData);
    OP_CHECK_IF((!reduceFullLoad && tilingData.get_parallelN() <= 0),
                OP_LOGE(context->GetNodeName(), "Failed to calculate Welford parallel tiling."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((tilingData.get_processSize() <= 0),
                OP_LOGE(context->GetNodeName(), "Failed to calculate a positive process size."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetEmptyBatchTilingData(gert::TilingContext* context, GroupNormTilingData& tilingData)
{
    tilingData.set_realCoreNum(0);
    tilingData.set_tilingKey(static_cast<int64_t>(GroupNormTilingKey::TILINGKEY_WELFORD_PERF));
    OP_CHECK_IF(GroupNormSetTilingData(context, tilingData) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeType(), "Failed to save empty-batch tiling data."), return ge::GRAPH_FAILED);
    // N为0时三个输出均为空，设置零核避免下发Kernel。
    context->SetBlockDim(tilingData.get_realCoreNum());
    context->SetTilingKey(tilingData.get_tilingKey());
    size_t* workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    workspaces[0] = 0;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SetGroupNormTilingData(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "Start running Tiling4GroupNorm.");
    OP_CHECK_IF((CheckInputParams(context) != ge::GRAPH_SUCCESS),
                OP_LOGE(context->GetNodeName(), "InputParams is invalid."), return ge::GRAPH_FAILED);
    OP_CHECK_IF((CheckAttrParams(context) != ge::GRAPH_SUCCESS),
                OP_LOGE(context->GetNodeName(), "AttrParams is invalid."), return ge::GRAPH_FAILED);

    GroupNormTilingData tilingData;
    OP_CHECK_IF((SetAttrParams(context, tilingData) != ge::GRAPH_SUCCESS),
                OP_LOGE(context->GetNodeName(), "Set attrParams failed."), return ge::GRAPH_FAILED);
    auto xTensor = context->GetInputTensor(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xTensor);
    const gert::Shape& xShape = xTensor->GetStorageShape();
    if (xShape.GetShapeSize() == 0) {
        OP_CHECK_IF(
            xShape.GetDim(DIM_0) != 0,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "x", Ops::Base::ToString(xShape).c_str(),
                                                  "An empty x is supported only when the batch dimension is 0"),
            return ge::GRAPH_FAILED);
        return SetEmptyBatchTilingData(context, tilingData);
    }
    OP_CHECK_IF((SetTilingParams(context, tilingData) != ge::GRAPH_SUCCESS),
                OP_LOGE(context->GetNodeName(), "Set tilingParams failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF((SetBlockTiling(context, tilingData) != ge::GRAPH_SUCCESS),
                OP_LOGE(context->GetNodeName(), "Set blockTiling failed."), return ge::GRAPH_FAILED);
    SetUbTiling(tilingData);
    OP_CHECK_IF(SetTilingForAscend950(context, tilingData) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "SetTilingForAscend950 failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(GroupNormSetTilingData(context, tilingData) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeType(), "GroupNormSetTilingData set tiling data fail."),
                return ge::GRAPH_FAILED);
    context->SetBlockDim(tilingData.get_realCoreNum());
    context->SetTilingKey(tilingData.get_tilingKey());
    size_t* workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    workspaces[0] = RESERVED_WORKSPACE_SIZE_950;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4GroupNorm(gert::TilingContext* context)
{
    auto compile_info = context->GetCompileInfo<GroupNormCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);

    // 获取输入shape并校验最小维数。
    auto input_first = context->GetInputShape(0);
    OP_CHECK_IF(input_first == nullptr, OP_LOGE(context->GetNodeName(), "get input_first failed."),
                return ge::GRAPH_FAILED);
    const gert::Shape& input_shape = input_first->GetStorageShape();

    const int32_t input_dim_size = input_shape.GetDimNum();
    OP_CHECK_IF(input_dim_size < MIN_LEN,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "x", std::to_string(input_dim_size).c_str(),
                                             "greater than or equal to 2"),
                return ge::GRAPH_FAILED);

    OP_LOGD(context->GetNodeName(), "GroupNorm tik_compile_info is null, runs ascendc tiling func");
    ge::graphStatus set_tiling_data_statues = SetGroupNormTilingData(context);
    return set_tiling_data_statues;
}

static ge::graphStatus TilingPrepare4GroupNorm(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "begin to get compile info for GroupNorm.");
    auto compile_info = GetCompileInfoPtr<GroupNormCompileInfo>(context);
    OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);
    std::unique_ptr<nlohmann::json> parsed_object_cinfo = GetCompileInfoJson(context);
    OP_CHECK_NULL_WITH_CONTEXT(context, parsed_object_cinfo);
    const nlohmann::json& vars = (*parsed_object_cinfo)["vars"];
    const nlohmann::json& all_vars = (*parsed_object_cinfo)["_vars"];
    if (vars.empty() && all_vars.empty()) {
        OP_LOGD(context->GetNodeName(), "GroupNorm no need to parse compile info.");
        auto platform_info = context->GetPlatformInfo();
        OP_CHECK_NULL_WITH_CONTEXT(context, platform_info);
        auto ascendc_platform = platform_ascendc::PlatformAscendC(platform_info);
        compile_info->coreNum = ascendc_platform.GetCoreNumAiv();
        OP_CHECK_IF((compile_info->coreNum <= 0),
                    OP_LOGE(context->GetNodeName(), "Get coreNum failed, coreNum: %d", compile_info->coreNum),
                    return ge::GRAPH_FAILED);
        uint64_t ubSize;
        ascendc_platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        compile_info->ubSize = static_cast<int64_t>(ubSize);
        OP_CHECK_IF((compile_info->ubSize <= 0),
                    OP_LOGE(context->GetNodeName(), "Get ubSize failed, ubSize: %ld", compile_info->ubSize),
                    return ge::GRAPH_FAILED);
        compile_info->blockSize = Ops::Base::GetUbBlockSize(context);
        OP_CHECK_IF((compile_info->blockSize <= 0),
                    OP_LOGE(context->GetNodeName(), "Get blockSize failed, blockSize: %ld", compile_info->blockSize),
                    return ge::GRAPH_FAILED);
        compile_info->vectorLength = Ops::Base::GetVRegSize(context);
        OP_CHECK_IF(
            (compile_info->vectorLength <= 0),
            OP_LOGE(context->GetNodeName(), "Get vectorLength failed, vectorLength: %ld", compile_info->vectorLength),
            return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }
    return ge::GRAPH_FAILED;
}

// 注册GroupNorm的tiling与compile info解析接口。
IMPL_OP_OPTILING(GroupNorm).Tiling(Tiling4GroupNorm).TilingParse<GroupNormCompileInfo>(TilingPrepare4GroupNorm);
} // namespace optiling
