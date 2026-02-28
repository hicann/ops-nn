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
 * \file group_norm_v2_tiling_arch35.cpp
 * \brief
 */
#include "group_norm_v2_tiling_arch35.h"
#include "op_api/runtime2_util.h"
#include "error_util.h"
#include <nlohmann/json.hpp>

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
static const uint64_t INPUT_IDX_X = 0;
static const uint64_t INPUT_IDX_GAMMA = 1;
static const uint64_t INPUT_IDX_BETA = 2;
static const uint64_t PROCESSSIZE = 8192;
static const uint64_t BLOCK_SIZE = 32U;
static const uint64_t VECTOR_LENGTH = 256U;
static const uint64_t RESERVED_WORKSPACE_SIZE_950 = 16UL * 1024UL * 1024UL;
static const uint64_t FOUR_BUFFER = 4;
static const uint64_t BUFFER_NUM = 2;
static const uint64_t DOUBLE_BUFFER = 2;
static const uint64_t DICHOTOMY_ADD_COEFF = 2;
static const uint64_t ULONG_BIT_LEN = 64;
static const uint64_t MAX_CHANNEL_SIZE = 4096;
static const uint64_t MAX_NUM_PER_CORE = 2048;
static const float DEFAULT_EPS = 1e-5;

inline std::unique_ptr<nlohmann::json> GetCompileInfoJson(gert::TilingParseContext* context) {
  auto json_str = context->GetCompiledJson();
  OP_CHECK_IF(json_str == nullptr, OP_LOGE(context->GetNodeName(), "json_str is nullptr!"), return nullptr);
  std::unique_ptr<nlohmann::json> parsed_object_cinfo =
      std::make_unique<nlohmann::json>(nlohmann::json::parse(json_str));
  return parsed_object_cinfo;
}

struct WelfordTilingInitResult {
    uint64_t loopNum{0};
    uint64_t loopTail{0};
    uint64_t processSize{0};
    uint64_t innerLoopNum{0};
    uint64_t innerLoopTail{0};
    uint64_t hwNum{0};
    uint64_t hwNumAlign{0};
    bool checkResult{false};
};

inline static ge::graphStatus GroupNormV2SetTilingData(gert::TilingContext *context, GroupNormV2TilingData &tilingData)
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

inline static int64_t RoundUp(int64_t a, int64_t b)
{
    return CeilDiv(a, b) * b;
}


static bool isMixType(const gert::TilingContext *context)
{
    auto xDtype = context->GetInputDesc(INPUT_IDX_X)->GetDataType();
    uint64_t xDtypeSize = ge::GetSizeByDataType(xDtype);
    auto gammaDesc = context->GetInputDesc(INPUT_IDX_GAMMA);
    uint64_t gammaDtypeSize = ge::GetSizeByDataType(gammaDesc->GetDataType());
    if (gammaDtypeSize == xDtypeSize) {
        return false;
    }
    return true;
}

static ge::graphStatus CheckInputXShape(const gert::TilingContext *context, const gert::Shape &xShape)
{
    uint64_t xDims = xShape.GetDimNum();
    for (uint64_t i = 0; i < xDims; i++) {
        int64_t curDim = xShape.GetDim(i);
        OP_CHECK_IF((curDim <= 0),
            OP_LOGE(context->GetNodeName(),
            "The input %lu dimension must be greater than 0, currently is %ld.", i, curDim),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

// GNV2支持两种混合精度(bfloat16, float32, float32) (float16, float32, float32)
static ge::graphStatus CheckMixType(const ge::DataType &xDtype, const ge::DataType &gammaDtype)
{
    if (xDtype == gammaDtype) {
        return ge::GRAPH_SUCCESS;
    }
    if (xDtype == ge::DT_FLOAT16 && gammaDtype == ge::DT_FLOAT) {
        return ge::GRAPH_SUCCESS;
    }
    if (xDtype == ge::DT_BF16 && gammaDtype == ge::DT_FLOAT) {
        return ge::GRAPH_SUCCESS;
    }
    return ge::GRAPH_FAILED;
}

static ge::graphStatus CheckInputParams(const gert::TilingContext *context)
{
    // check x
    auto inputX = context->GetInputTensor(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX);
    auto xDtype = context->GetInputDesc(INPUT_IDX_X)->GetDataType();
    uint64_t xDtypeSize = ge::GetSizeByDataType(xDtype);
    OP_CHECK_IF((xDtypeSize <= 0),
        OP_LOGE(context->GetNodeName(), "xDtypeSize is invalid %lu, please check.", xDtypeSize),
        return ge::GRAPH_FAILED);
    auto xShape = inputX->GetStorageShape();
    uint64_t channel = xShape.GetDim(DIM_1);
    if (CheckInputXShape(context, xShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // check gamma and beta
    auto gammaShapePtr = context->GetInputShape(INPUT_IDX_GAMMA);
    OP_CHECK_NULL_WITH_CONTEXT(context, gammaShapePtr);
    auto gammaShape = gammaShapePtr->GetStorageShape();
    uint64_t gammaSizes = gammaShape.GetDim(DIM_0);
    OP_CHECK_IF((gammaShape.GetDimNum() != 1 || gammaSizes != channel),
        OP_LOGE(context->GetNodeName(),
        "Gamma dimension should be one, and the shape of gamma must be the same as  channel size of input, currently "
        "is %lu.",
        gammaSizes),
        return ge::GRAPH_FAILED);
    auto betaShapePtr = context->GetInputShape(INPUT_IDX_BETA);
    OP_CHECK_NULL_WITH_CONTEXT(context, betaShapePtr);
    auto betaShape = betaShapePtr->GetStorageShape();
    uint64_t betaSizes = betaShape.GetDim(DIM_0);
    OP_CHECK_IF((betaShape.GetDimNum() != 1 || betaSizes != channel),
        OP_LOGE(context->GetNodeName(),
        "Beta dimension should be one, and the shape of beta must be the same as channel size of input, currently is "
        "%lu.",
        betaSizes),
        return ge::GRAPH_FAILED);
    auto gammaDtypePtr = context->GetInputDesc(INPUT_IDX_GAMMA);
    OP_CHECK_NULL_WITH_CONTEXT(context, gammaDtypePtr);
    auto gammaDtype = gammaDtypePtr->GetDataType();
    uint64_t gammaDtypeSize = ge::GetSizeByDataType(gammaDtype);
    auto betaDtypePtr = context->GetInputDesc(INPUT_IDX_BETA);
    OP_CHECK_NULL_WITH_CONTEXT(context, betaDtypePtr);
    auto betaDtype = betaDtypePtr->GetDataType();
    uint64_t betaDtypeSize = ge::GetSizeByDataType(betaDtype);
    OP_CHECK_IF((gammaDtypeSize < 0 || gammaDtypeSize != betaDtypeSize),
        OP_LOGE(context->GetNodeName(),
        "The dtype of gamma and beta must be consistent, currently gamma is %d, beta is %d.", gammaDtype, betaDtype),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF((CheckMixType(xDtype, gammaDtype) == ge::GRAPH_FAILED),
        OP_LOGE(context->GetNodeName(),
        "The dtype combination of gamma, beta and inputs is invalid."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAttrParams(const gert::TilingContext *context)
{
    auto inputX = context->GetInputTensor(INPUT_IDX_X);
    auto xShape = inputX->GetStorageShape();
    uint64_t channel = xShape.GetDim(DIM_1);
    // check num_groups
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t *numGroups = attrs->GetAttrPointer<int64_t>(INDEX_NUM_GROUPS);
    OP_CHECK_IF((*numGroups <= 0),
        OP_LOGE(context->GetNodeName(), "numGroups must be bigger than 0."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF((channel % *numGroups != 0),
        OP_LOGE(context->GetNodeName(), "channel must be integer multiples of numGroups."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static uint64_t GetOptionalInputTensorSize(const gert::TilingContext *context, uint64_t index,
    uint64_t specifiedValue = 0)
{
    auto tensorDesc = context->GetInputDesc(index);
    if (tensorDesc == nullptr) {
        return 0;
    }
    auto compileInfo = reinterpret_cast<const GroupNormV2CompileInfo *>(context->GetCompileInfo());
    uint32_t blockSize = BLOCK_SIZE;
    auto dtypeSize = ge::GetSizeByDataType(tensorDesc->GetDataType());
    if (specifiedValue != 0) {
        return RoundUp(specifiedValue * dtypeSize, blockSize);
    }

    auto storageShape = context->GetInputShape(index);
    OP_CHECK_NULL_WITH_CONTEXT(context, storageShape);
    auto shape = storageShape->GetStorageShape();
    uint64_t num = 1;
    for (uint64_t i = 0; i < shape.GetDimNum(); i++) {
        num = num * shape.GetDim(i);
    }
    auto numUbSize = RoundUp(num * dtypeSize, blockSize);
    return numUbSize;
}

static void GetDichotomyAddParams(const gert::TilingContext *context, uint64_t r, uint64_t &power, uint64_t &dichotomyK,
    uint64_t &extraSize, uint64_t &lastNum)
{
    auto compileInfo = reinterpret_cast<const GroupNormV2CompileInfo *>(context->GetCompileInfo());
    uint32_t vl = VECTOR_LENGTH / FLOAT32_BYTES;
    uint32_t blockSize = BLOCK_SIZE;
    uint64_t basePower = (1L << (ULONG_BIT_LEN - 1 - __builtin_clzl(r)));
    power = basePower == r ? basePower / DICHOTOMY_ADD_COEFF : basePower;
    uint64_t extraOriSize = power / vl;
    extraSize = RoundUp(extraOriSize * FLOAT32_BYTES, blockSize);
    dichotomyK = 0;
    if (extraOriSize < vl) {
        lastNum = extraOriSize;
        return;
    }
    uint64_t totalNum = extraOriSize / vl;
    uint64_t base = 1;
    lastNum = vl;
    while (base < totalNum) {
        dichotomyK++;
        base *= DICHOTOMY_ADD_COEFF;
    }
}

static ge::graphStatus SetAttrParams(const gert::TilingContext *context, GroupNormV2TilingData &tilingData)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t *numGroups = attrs->GetAttrPointer<int64_t>(INDEX_NUM_GROUPS);
    const float *epsilonPtr = attrs->GetAttrPointer<float>(INDEX_EPSILON);
    float eps = epsilonPtr == nullptr ? DEFAULT_EPS : *epsilonPtr;
    tilingData.set_numGroups(*numGroups);
    tilingData.set_epsilon(eps);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetTilingParams(const gert::TilingContext *context, GroupNormV2TilingData &tilingData)
{
    auto inputX = context->GetInputTensor(INPUT_IDX_X);
    auto xShape = inputX->GetStorageShape();
    uint64_t hwNum = 1;
    uint64_t xDims = xShape.GetDimNum();
    for (uint64_t i = 2; i < xDims; i++) {
        hwNum = hwNum * xShape.GetDim(i);
    }
    tilingData.set_shapeC(xShape.GetDim(DIM_1));
    tilingData.set_shapeD(xShape.GetDim(DIM_1) / tilingData.get_numGroups());
    tilingData.set_hwNum(hwNum);
    tilingData.set_elemNum(tilingData.get_shapeD() * hwNum);
    tilingData.set_processSize(PROCESSSIZE);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetBlockTiling(const gert::TilingContext *context, GroupNormV2TilingData &tilingData)
{
    auto compileInfo = reinterpret_cast<const GroupNormV2CompileInfo *>(context->GetCompileInfo());
    auto inputX = context->GetInputTensor(INPUT_IDX_X);
    auto xShape = inputX->GetStorageShape();
    uint64_t shapeN = xShape.GetDim(DIM_0);
    tilingData.set_numPerCore(CeilDiv(shapeN * tilingData.get_numGroups(), compileInfo->coreNum));
    tilingData.set_realCoreNum(CeilDiv(shapeN * tilingData.get_numGroups(), tilingData.get_numPerCore()));
    tilingData.set_numLastCore(shapeN * tilingData.get_numGroups() -
        tilingData.get_numPerCore() * (tilingData.get_realCoreNum() - 1));
    uint64_t xShapeSize = xShape.GetShapeSize();
    if (xShapeSize == 0) {
        tilingData.set_realCoreNum(-1);
    }
    return ge::GRAPH_SUCCESS;
}

static void SetUbTiling(GroupNormV2TilingData &tilingData)
{
    tilingData.set_loopNum(CeilDiv(tilingData.get_elemNum(), tilingData.get_processSize()));
    tilingData.set_loopTail(tilingData.get_elemNum() - tilingData.get_processSize() * (tilingData.get_loopNum() - 1));
    tilingData.set_innerLoopNum(CeilDiv(tilingData.get_hwNum(), tilingData.get_processSize()));
    tilingData.set_innerLoopTail(tilingData.get_hwNum() -
        tilingData.get_processSize() * (tilingData.get_innerLoopNum() - 1));
}

/*
  950 tilingKey划分逻辑如下:
  1. R轴等于1,走TILINGKEY_REDUCE_ONE特化模板
  2. UB内默认存放2048根A轴，gamma和beta在UB内全载，在开启DB条件下，计算出能够全载的最大R轴
    2.1 R轴小于可全载的最大R轴,则走TILINGKEY_TWOPASS_PERF性能模板
    2.2 尝试切分gamma和beta，每次拷入gamma和beta时，拷贝一个完整的D大小的数据
  计算出新的可全载的最大R轴，如果大于R轴实际大小，则走TILINGKEY_TWOPASS_GENERALIZED泛化模板
    2.3
  当R轴不能全载时，如果gamma和beta小于4096，则走TILINGKEY_WELFORD_PERF模板，否则走TILINGKEY_WELFORD_GENERALIZED模板
  二分累加所需要的UB大小，计算规则如下:
  1. 优先按照R轴可以全载计算所需要的额外空间
  2. 如果R轴无法全载，则需要根据可用的UB空间，结合二分累加的额外空间，重新计算出最大的并行度
  在非全载模板下，二分累加的UB额外空间不会影响normalize+swish阶段一次可载入的R轴大小
  当Gamma或者Beta非空，并且和输入数据类型不一致时，认为是mix type场景
*/
static void SetTilingKey4Ascend950(const gert::TilingContext *context, uint64_t &maxReduceCount, uint64_t &ubRemain,
    bool &isReduceFullLoad, GroupNormV2TilingData &tilingData)
{
    auto compileInfo = reinterpret_cast<const GroupNormV2CompileInfo *>(context->GetCompileInfo());
    uint64_t ubSize = compileInfo->ubSize;
    uint32_t blockSize = BLOCK_SIZE;
    uint64_t reduceCount = tilingData.get_shapeD() * tilingData.get_hwNum();
    uint64_t gammaUbSize = GetOptionalInputTensorSize(context, INPUT_IDX_GAMMA);
    uint64_t betaUbSize = GetOptionalInputTensorSize(context, INPUT_IDX_BETA);
    uint64_t realNumPerCore = std::min(MAX_NUM_PER_CORE,
        static_cast<uint64_t>(std::max(tilingData.get_numPerCore(), tilingData.get_numLastCore())));
    uint64_t meanUbSize = RoundUp(realNumPerCore * FLOAT32_BYTES, blockSize);
    uint64_t rstdUbSize = RoundUp(realNumPerCore * FLOAT32_BYTES, blockSize);

    uint64_t otherUbSize = gammaUbSize + betaUbSize + meanUbSize + rstdUbSize;
    uint64_t xDtypeSize = ge::GetSizeByDataType(context->GetInputDesc(INPUT_IDX_X)->GetDataType());
    uint64_t meanUbExtraSize = 0;
    uint64_t rstdUbExtraSize = 0;
    if (xDtypeSize != FLOAT32_BYTES) {
        meanUbExtraSize = RoundUp(realNumPerCore * xDtypeSize, blockSize);
        rstdUbExtraSize = RoundUp(realNumPerCore * xDtypeSize, blockSize);
        otherUbSize = otherUbSize + meanUbExtraSize + rstdUbExtraSize;
    }
    bool mixType = isMixType(context);

    uint64_t dichotomyAddPower = 0;
    uint64_t dichotomyAddK = 0;
    uint64_t dichotomyAddExtraSize = 0;
    uint64_t dichotomyAddLastNum = 0;
    GetDichotomyAddParams(context, reduceCount, dichotomyAddPower, dichotomyAddK, dichotomyAddExtraSize,
        dichotomyAddLastNum);
    otherUbSize += dichotomyAddExtraSize;

    ubRemain = ubSize <= otherUbSize ? 0 : ubSize - otherUbSize;
    OP_CHECK_IF((xDtypeSize == 0),
        OP_LOGE(context->GetNodeName(), "XDtypeSize is zero."), return);
    maxReduceCount = (ubRemain / (DOUBLE_BUFFER * BUFFER_NUM)) / xDtypeSize;

    if (maxReduceCount > reduceCount) {
        isReduceFullLoad = true;
        int64_t tilingKey = mixType ? static_cast<int64_t>(GroupNormV2TilingKey::TILINGKEY_TWOPASS_PERF_MIX_TYPE) :
                                      static_cast<int64_t>(GroupNormV2TilingKey::TILINGKEY_TWOPASS_PERF);
        tilingData.set_tilingKey(tilingKey);
        return;
    }
    bool isLargeChannel = static_cast<uint64_t>(tilingData.get_shapeC()) > MAX_CHANNEL_SIZE;
    uint64_t newUbRemain = ubRemain;
    // 对于Channel轴过大的场景，尝试将gamma大小限制为ShapeD，beta大小限制为shapeD，重新计算最大可全载的R轴
    // 如果此时仍然无法全载，则走channel过大的非全载模板
    if (isLargeChannel) {
        uint64_t gammaSplitUbSize = GetOptionalInputTensorSize(context, INPUT_IDX_GAMMA, tilingData.get_shapeD());
        uint64_t betaSplitUbSize = GetOptionalInputTensorSize(context, INPUT_IDX_BETA, tilingData.get_shapeD());
        otherUbSize = otherUbSize - gammaUbSize - betaUbSize + gammaSplitUbSize + betaSplitUbSize;
        newUbRemain = ubSize <= otherUbSize ? 0 : ubSize - otherUbSize;
        uint64_t newMaxReduceCount = (newUbRemain / (DOUBLE_BUFFER * BUFFER_NUM)) / xDtypeSize;
        if (newMaxReduceCount > reduceCount) {
            isReduceFullLoad = true;
            maxReduceCount = newMaxReduceCount;
            ubRemain = newUbRemain;
            int64_t tilingKey = mixType ?
                static_cast<int64_t>(GroupNormV2TilingKey::TILINGKEY_TWOPASS_GENERALIZED_MIX_TYPE) :
                static_cast<int64_t>(GroupNormV2TilingKey::TILINGKEY_TWOPASS_GENERALIZED);
            tilingData.set_tilingKey(tilingKey);
            return;
        }
    }
    // R轴过大，无法走全载模版，此时需要将二分累加所需要的ub空间释放，重新更新ubRemain和newMaxReduceCount
    isReduceFullLoad = false;
    int64_t meanAndRstdSize = meanUbSize + rstdUbSize + meanUbExtraSize + rstdUbExtraSize;
    if (isLargeChannel) {
        ubRemain = ubSize - meanAndRstdSize;
        int64_t tilingKey = mixType ?
            static_cast<int64_t>(GroupNormV2TilingKey::TILINGKEY_WELFORD_GENERALIZED_MIX_TYPE) :
            static_cast<int64_t>(GroupNormV2TilingKey::TILINGKEY_WELFORD_GENERALIZED);
        tilingData.set_tilingKey(tilingKey);
    } else {
        ubRemain = ubSize - meanAndRstdSize - gammaUbSize - betaUbSize;
        int64_t tilingKey = mixType ? static_cast<int64_t>(GroupNormV2TilingKey::TILINGKEY_WELFORD_PERF_MIX_TYPE) :
                                      static_cast<int64_t>(GroupNormV2TilingKey::TILINGKEY_WELFORD_PERF);
        tilingData.set_tilingKey(tilingKey);
    }
    maxReduceCount = (ubRemain / (DOUBLE_BUFFER * BUFFER_NUM)) / xDtypeSize;
}

static void SetDichotomyAddParams(const gert::TilingContext *context, GroupNormV2TilingData &tilingData)
{
    uint64_t reduceCount = tilingData.get_shapeD() * tilingData.get_hwNum();
    uint64_t dichotomyAddPower = 0;
    uint64_t dichotomyAddK = 0;
    uint64_t dichotomyAddExtraSize = 0;
    uint64_t dichotomyAddLastNum = 0;
    GetDichotomyAddParams(context, reduceCount, dichotomyAddPower, dichotomyAddK, dichotomyAddExtraSize,
        dichotomyAddLastNum);
    tilingData.set_dichotomyAddPower(dichotomyAddPower);
    tilingData.set_dichotomyAddK(dichotomyAddK);
    tilingData.set_dichotomyAddLastNum(dichotomyAddLastNum);
}

static void SetWelfordParallelN(const gert::TilingContext *context, uint64_t xDtypeSize, uint64_t ubRemain,
    GroupNormV2TilingData &tilingData)
{
    auto compileInfo = reinterpret_cast<const GroupNormV2CompileInfo *>(context->GetCompileInfo());
    uint32_t blockSize = BLOCK_SIZE;
    OP_CHECK_IF((xDtypeSize == 0),
        OP_LOGE(context->GetNodeName(), "XDtypeSize is zero."), return);
    uint32_t coeff = FLOAT32_BYTES / xDtypeSize;
    uint32_t totalNum = BUFFER_NUM * (coeff + 1);
    uint32_t welfordBase = blockSize / xDtypeSize;
    OP_CHECK_IF((totalNum == 0),
        OP_LOGE(context->GetNodeName(), "TotalNum is zero."), return);
    uint32_t maxParallelN = DownAlign((ubRemain / xDtypeSize) / totalNum, welfordBase);

    uint64_t dichotomyAddPower = 0;
    uint64_t dichotomyAddK = 0;
    uint64_t dichotomyAddExtraSize = 0;
    uint64_t dichotomyAddLastNum = 0;
    GetDichotomyAddParams(context, maxParallelN, dichotomyAddPower, dichotomyAddK, dichotomyAddExtraSize,
        dichotomyAddLastNum);
    uint32_t ubCurUse =
        maxParallelN * BUFFER_NUM * xDtypeSize + dichotomyAddExtraSize + maxParallelN * BUFFER_NUM * FLOAT32_BYTES;
    while (ubCurUse > ubRemain) {
        maxParallelN -= welfordBase;
        GetDichotomyAddParams(context, maxParallelN, dichotomyAddPower, dichotomyAddK, dichotomyAddExtraSize,
            dichotomyAddLastNum);
        ubCurUse =
            maxParallelN * BUFFER_NUM * xDtypeSize + dichotomyAddExtraSize + maxParallelN * BUFFER_NUM * FLOAT32_BYTES;
    }

    if (maxParallelN > tilingData.get_elemNum()) {
        maxParallelN = tilingData.get_elemNum();
        GetDichotomyAddParams(context, maxParallelN, dichotomyAddPower, dichotomyAddK, dichotomyAddExtraSize,
            dichotomyAddLastNum);
    }
    tilingData.set_dichotomyAddPower(dichotomyAddPower);
    tilingData.set_dichotomyAddK(dichotomyAddK);
    tilingData.set_dichotomyAddLastNum(dichotomyAddLastNum);
    tilingData.set_parallelN(maxParallelN);
}

static void SetUbTiling4TwoPass(const gert::TilingContext *context, GroupNormV2TilingData &tilingData,
    uint64_t maxReduceCount, uint32_t xDtypeSize)
{
    auto compileInfo = reinterpret_cast<const GroupNormV2CompileInfo *>(context->GetCompileInfo());
    uint32_t blockSize = BLOCK_SIZE;
    uint64_t elemNum = tilingData.get_elemNum();
    OP_CHECK_IF((xDtypeSize == 0),
        OP_LOGE(context->GetNodeName(), "XDtypeSize is zero."), return);
    uint64_t elemNumAlign = RoundUp(elemNum, blockSize / xDtypeSize);
    SetDichotomyAddParams(context, tilingData);
    OP_CHECK_IF((elemNumAlign == 0),
        OP_LOGE(context->GetNodeName(), "ElemNumAlign is zero."), return);
    uint64_t count = maxReduceCount / elemNumAlign;
    uint64_t processSize = count * elemNumAlign;
    tilingData.set_processSize(processSize);
}

static WelfordTilingInitResult InitWelfordTilingCommon(const gert::TilingContext *context, 
    GroupNormV2TilingData &tilingData, uint32_t blockSize, uint32_t xDtypeSize) {
    WelfordTilingInitResult result{};
    result.hwNum = tilingData.get_hwNum();
    OP_CHECK_IF((xDtypeSize == 0),
        OP_LOGE(context->GetNodeName(), "XDtypeSize is zero."), return result);
    result.hwNumAlign = RoundUp(result.hwNum, blockSize / xDtypeSize);
    OP_CHECK_IF((result.hwNumAlign == 0),
        OP_LOGE(context->GetNodeName(), "HwNumAlign is zero."), return result);
    result.checkResult = true;
    return result;
}

static void SetUbTiling4WelfordPerf(const gert::TilingContext *context, GroupNormV2TilingData &tilingData,
    uint64_t maxReduceCount, uint32_t ubRemain, uint32_t xDtypeSize)
{
    auto compileInfo = reinterpret_cast<const GroupNormV2CompileInfo *>(context->GetCompileInfo());
    uint32_t blockSize = BLOCK_SIZE;
    SetWelfordParallelN(context, xDtypeSize, ubRemain, tilingData);
    WelfordTilingInitResult result = InitWelfordTilingCommon(context, tilingData, blockSize, xDtypeSize);
    OP_CHECK_IF((result.checkResult == false),
        OP_LOGE(context->GetNodeName(), "InitWelfordTilingCommon Failed."), return);
    uint64_t count = maxReduceCount / result.hwNumAlign;
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
static void SetUbTiling4WelfordGeneralized(const gert::TilingContext *context, GroupNormV2TilingData &tilingData,
    uint32_t ubRemain, uint32_t xDtypeSize)
{
    auto compileInfo = reinterpret_cast<const GroupNormV2CompileInfo *>(context->GetCompileInfo());
    uint32_t blockSize = BLOCK_SIZE;
    WelfordTilingInitResult result = InitWelfordTilingCommon(context, tilingData, blockSize, xDtypeSize);
    OP_CHECK_IF((result.checkResult == false),
        OP_LOGE(context->GetNodeName(), "InitWelfordTilingCommon Failed."), return);
    uint64_t maxReduceCount = (ubRemain / (DOUBLE_BUFFER * BUFFER_NUM)) / xDtypeSize;
    uint64_t count = maxReduceCount / result.hwNumAlign;
    uint64_t gammaRealSize = GetOptionalInputTensorSize(context, INPUT_IDX_GAMMA, count);
    uint64_t betaRealSize = GetOptionalInputTensorSize(context, INPUT_IDX_BETA, count);
    uint64_t curUbSize = gammaRealSize + betaRealSize + count * result.hwNumAlign * xDtypeSize * BUFFER_NUM * DOUBLE_BUFFER;
    while (curUbSize > ubRemain && count >= 1) {
        count--;
        uint64_t gammaRealSize = GetOptionalInputTensorSize(context, INPUT_IDX_GAMMA, count);
        uint64_t betaRealSize = GetOptionalInputTensorSize(context, INPUT_IDX_BETA, count);
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
        uint64_t maxReduceCountDownAlign = DownAlign(maxReduceCount, blockSize / xDtypeSize);
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

static void SetUbTiling4Welford(const gert::TilingContext *context, GroupNormV2TilingData &tilingData,
    uint64_t maxReduceCount, uint64_t ubRemain, uint32_t xDtypeSize)
{
    if (tilingData.get_tilingKey() == static_cast<int64_t>(GroupNormV2TilingKey::TILINGKEY_WELFORD_PERF) ||
        tilingData.get_tilingKey() == static_cast<int64_t>(GroupNormV2TilingKey::TILINGKEY_WELFORD_PERF_MIX_TYPE)) {
        return SetUbTiling4WelfordPerf(context, tilingData, maxReduceCount, ubRemain, xDtypeSize);
    }
    return SetUbTiling4WelfordGeneralized(context, tilingData, ubRemain, xDtypeSize);
}

static void SetUbTiling4Ascend950(const gert::TilingContext *context, uint64_t maxReduceCount, uint64_t ubRemain,
    bool isReduceFullLoad, GroupNormV2TilingData &tilingData)
{
    auto compileInfo = reinterpret_cast<const GroupNormV2CompileInfo *>(context->GetCompileInfo());
    int32_t ubSize = compileInfo->ubSize;
    uint64_t xDtypeSize = ge::GetSizeByDataType(context->GetInputDesc(INPUT_IDX_X)->GetDataType());
    tilingData.set_ubSize(ubSize);
    if (!isReduceFullLoad) {
        SetUbTiling4Welford(context, tilingData, maxReduceCount, ubRemain, xDtypeSize);
    } else {
        SetUbTiling4TwoPass(context, tilingData, maxReduceCount, xDtypeSize);
    }
}

static void SetTilingForAscend950(const gert::TilingContext *context, GroupNormV2TilingData &tilingData)
{
    uint64_t maxReduceCount = 0;
    uint64_t ubRemain = 0;
    bool reduceFullLoad = false;
    SetTilingKey4Ascend950(context, maxReduceCount, ubRemain, reduceFullLoad, tilingData);
    SetUbTiling4Ascend950(context, maxReduceCount, ubRemain, reduceFullLoad, tilingData);
}

ge::graphStatus SetTilingData(gert::TilingContext *context)
{
    OP_LOGD(context->GetNodeName(), "Start running Tiling4GroupNormV2.");
    OP_CHECK_IF((CheckInputParams(context) != ge::GRAPH_SUCCESS),
        OP_LOGE(context->GetNodeName(), "InputParams is invalid."), return ge::GRAPH_FAILED);
    OP_CHECK_IF((CheckAttrParams(context) != ge::GRAPH_SUCCESS),
        OP_LOGE(context->GetNodeName(), "AttrParams is invalid."), return ge::GRAPH_FAILED);

    GroupNormV2TilingData tilingData;
    OP_CHECK_IF((SetAttrParams(context, tilingData) != ge::GRAPH_SUCCESS),
        OP_LOGE(context->GetNodeName(), "Set attrParams failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF((SetTilingParams(context, tilingData) != ge::GRAPH_SUCCESS),
        OP_LOGE(context->GetNodeName(), "Set tilingParams failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF((SetBlockTiling(context, tilingData) != ge::GRAPH_SUCCESS),
        OP_LOGE(context->GetNodeName(), "Set blockTiling failed."), return ge::GRAPH_FAILED);
    SetUbTiling(tilingData);
    SetTilingForAscend950(context, tilingData);
    OP_CHECK_IF(GroupNormV2SetTilingData(context, tilingData) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeType(), "GroupNormV2SetTilingData set tiling data fail."),
        return ge::GRAPH_FAILED);
    context->SetBlockDim(tilingData.get_realCoreNum());
    context->SetTilingKey(tilingData.get_tilingKey());
    size_t *workspaces = context->GetWorkspaceSizes(1);
    workspaces[0] = RESERVED_WORKSPACE_SIZE_950;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4GroupNormV2(gert::TilingContext *context)
{
    const GroupNormV2CompileInfo *compile_info =
        reinterpret_cast<const GroupNormV2CompileInfo *>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);

    // get input shape info
    auto input_first = context->GetInputShape(0);
    OP_CHECK_IF(input_first == nullptr,
        OP_LOGE(context->GetNodeName(), "get input_first failed."), return ge::GRAPH_FAILED);
    const gert::Shape &input_shape = input_first->GetStorageShape();

    const int32_t input_dim_size = input_shape.GetDimNum();
    OP_CHECK_IF(input_dim_size < MIN_LEN,
        OP_LOGE(context->GetNodeName(), "input_dim can't be smaller than 2"),
        return ge::GRAPH_FAILED);

    OP_LOGD(context->GetNodeName(), "GroupNormV2 tik_compile_info is null, runs ascendc tiling func");
    ge::graphStatus set_tiling_data_statues = SetTilingData(context);
    return set_tiling_data_statues;
}

static ge::graphStatus TilingPrepare4GroupNormV2(gert::TilingParseContext *context)
{
    OP_LOGD(context->GetNodeName(), "begin to get compile info for GroupNormV2.");
    auto compile_info = GetCompileInfoPtr<GroupNormV2CompileInfo>(context);
    OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);
    std::unique_ptr<nlohmann::json> parsed_object_cinfo = GetCompileInfoJson(context);
    OP_CHECK_NULL_WITH_CONTEXT(context, parsed_object_cinfo);
    const nlohmann::json &vars = (*parsed_object_cinfo)["vars"];
    const nlohmann::json &all_vars = (*parsed_object_cinfo)["_vars"];
    if (vars.empty() && all_vars.empty()) {
        OP_LOGD(context->GetNodeName(), "GroupNormV2 no need to parse compile info.");
        auto platform_info = context->GetPlatformInfo();
        OP_CHECK_NULL_WITH_CONTEXT(context, platform_info);
        auto ascendc_platform = platform_ascendc::PlatformAscendC(platform_info);
        compile_info->coreNum = ascendc_platform.GetCoreNumAiv();
        uint64_t ubSize;
        ascendc_platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        compile_info->ubSize = static_cast<int64_t>(ubSize);
        return ge::GRAPH_SUCCESS;
    }
    return ge::GRAPH_FAILED;
}

// register tiling interface of the GroupNormV2 op.
IMPL_OP_OPTILING(GroupNormV2).Tiling(Tiling4GroupNormV2).TilingParse<GroupNormV2CompileInfo>(TilingPrepare4GroupNormV2);
} // namespace optiling