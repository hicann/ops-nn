/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file scatter_elements_v2_tiling.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_info.h"
#include "op_host/tiling_util.h"
#include "scatter_elements_v2_asc_tiling.h"
#include "scatter_elements_v2_tiling.h"

using namespace std;
using Ops::NN::Optiling::TilingRegistry;

namespace {
const int OPERATOR_TYPE = 1;

const int SIZE_OF_FP16 = 2;
const int SIZE_OF_FP32 = 4;
const int SIZE_OF_INT32 = 4;
const int SIZE_OF_INT64 = 8;
const int SIZE_OF_UINT8 = 1;
const int SIZE_OF_INT8 = 1;
const int SIZE_OF_BF16 = 2;

const int NONE = 1;
const int ADD = 2;
const int MUL = 3;
const int BUFFER_NUM = 1;
const int HALF_UB = 2;

const int INPUT_0 = 0;
const int INPUT_1 = 1;
const int INPUT_2 = 2;
const int WORK_SPACE_SIZE = 1024 * 1024 * 16;

const int SMALL_MODE = 1;
const int VAR_LIMIT = 128;
const int VAR_GROUPS = 64;

const size_t TWO_DIM = 2;
const int64_t NO_TRANSPOSE_DIM_MAX = 256;
const size_t NO_TRANSPOSE_TASKS_MIN = 768;
} // namespace

namespace optiling {
using namespace Ops::NN::OpTiling;

struct TilingDataStructScatterElementsV2310P{
  uint64_t M = 0;
  uint64_t varN = 0;
  uint64_t indicesN = 0;
  uint64_t updatesN = 0;
  uint64_t frontCoreNum = 0;
  uint64_t tailCoreNum = 0;
  uint64_t frontCoreData = 0;
  uint64_t tailCoreData = 0;
  uint64_t computeMode = 0;
  uint32_t usedCoreNum = 0;
  uint64_t MLeft = 0;

  uint32_t tilingKey = 0;
};

void coreSplit(uint64_t coreNums, uint64_t dataNums,
               uint64_t& frontCoreNums, uint64_t& tailCoreNums,
               uint64_t& frontCoreDataNums, uint64_t& tailCoreDataNums) {
    coreNums = coreNums == 0 ? 1 : coreNums;
    tailCoreDataNums = dataNums / coreNums;
    if (tailCoreDataNums == 0) {
        frontCoreNums = dataNums;
        tailCoreNums = 0;
        frontCoreDataNums = 1;
        tailCoreDataNums = 0;
    } else {
        if (dataNums % coreNums == 0) {
            frontCoreNums = coreNums;
            frontCoreDataNums = tailCoreDataNums;
            tailCoreNums = 0;
            tailCoreDataNums = 0;
        } else {
            frontCoreNums = dataNums - tailCoreDataNums * coreNums;
            frontCoreDataNums = tailCoreDataNums + 1;
            tailCoreNums = coreNums - frontCoreNums;
        }
    }
}

class ScatterElementsV2Tiling310P {
public:
  explicit ScatterElementsV2Tiling310P(gert::TilingContext* context_) : context(context_){};
  ge::graphStatus Init();
private:
  void SetKernelTiling();
  void TilingDataPrint() const;
  gert::TilingContext* context = nullptr;
  TilingDataStructScatterElementsV2310P tilingData;
};

void ScatterElementsV2Tiling310P::TilingDataPrint() const {
  OP_LOGD(context->GetNodeName(), "M: %ld.", tilingData.M);
  OP_LOGD(context->GetNodeName(), "varN: %ld.", tilingData.varN);
  OP_LOGD(context->GetNodeName(), "indicesN: %ld.", tilingData.indicesN);
  OP_LOGD(context->GetNodeName(), "updatesN: %ld.", tilingData.updatesN);
  OP_LOGD(context->GetNodeName(), "frontCoreNum: %ld.", tilingData.frontCoreNum);
  OP_LOGD(context->GetNodeName(), "tailCoreNum: %ld.", tilingData.tailCoreNum);
  OP_LOGD(context->GetNodeName(), "frontCoreData: %ld.", tilingData.frontCoreData);
  OP_LOGD(context->GetNodeName(), "tailCoreData: %ld.", tilingData.tailCoreData);
  OP_LOGD(context->GetNodeName(), "computeMode: %ld.", tilingData.computeMode);
  OP_LOGD(context->GetNodeName(), "MLeft: %ld.", tilingData.MLeft);
  OP_LOGD(context->GetNodeName(), "usedCoreNum: %d.", tilingData.usedCoreNum);
  OP_LOGD(context->GetNodeName(), "tilingKey: %d.", tilingData.tilingKey);
}

void ScatterElementsV2Tiling310P::SetKernelTiling() {
  ScatterElementsV2TilingData kernelTiling;
  kernelTiling.set_M(tilingData.M);
  kernelTiling.set_varN(tilingData.varN);
  kernelTiling.set_indicesN(tilingData.indicesN);
  kernelTiling.set_updatesN(tilingData.updatesN);
  kernelTiling.set_frontCoreNum(tilingData.frontCoreNum);
  kernelTiling.set_tailCoreNum(tilingData.tailCoreNum);
  kernelTiling.set_frontCoreData(tilingData.frontCoreData);
  kernelTiling.set_tailCoreData(tilingData.tailCoreData);
  kernelTiling.set_computeMode(tilingData.computeMode);
  kernelTiling.set_usedCoreNum(tilingData.usedCoreNum);
  kernelTiling.set_MLeft(tilingData.MLeft);

  context->SetBlockDim(tilingData.usedCoreNum);
  context->SetTilingKey(tilingData.tilingKey);

  kernelTiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(kernelTiling.GetDataSize());
}

ge::graphStatus ScatterElementsV2Tiling310P::Init() {
    const gert::StorageShape* varShape = context->GetInputShape(0);
    const gert::StorageShape* indicesShape = context->GetInputShape(1);
    const gert::StorageShape* updatesShape = context->GetInputShape(2);
    auto attrs = context->GetAttrs();

    int axis = *(attrs->GetAttrPointer<int>(0));
    const char* reduce = attrs->GetAttrPointer<char>(1);
    if (strcmp(reduce, "none") == 0) {
        tilingData.computeMode = 0;
    } else if (strcmp(reduce, "add") == 0) {
        tilingData.computeMode = 1;
    } else {
        OP_LOGE(context->GetNodeName(), "reduction only support none, add, or mul.");
        return ge::GRAPH_FAILED;
    }

    size_t varDims = static_cast<size_t>(varShape->GetStorageShape().GetDimNum());
    axis = axis >= 0 ? axis : axis + varDims; // do not support 0dim tensor.
    if (axis != static_cast<int>(varDims) - 1) {
        OP_LOGE(context->GetNodeName(), "only support last dim.");
        return ge::GRAPH_FAILED;
    }

    // var.shape reshape成[M, varN]
    tilingData.M = 1;
    tilingData.varN = varShape->GetStorageShape().GetDim(varDims - 1);
    for (size_t i = 0; i < varDims - 1; i++) {
        tilingData.M *= indicesShape->GetStorageShape().GetDim(i);
    }
    tilingData.indicesN = indicesShape->GetStorageShape().GetDim(varDims - 1);
    tilingData.updatesN = updatesShape->GetStorageShape().GetDim(varDims - 1);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
    tilingData.usedCoreNum = coreNum;
    if (tilingData.varN >= VAR_LIMIT) {
        // 列大
        tilingData.tilingKey = 0;
        coreSplit(tilingData.usedCoreNum, tilingData.M,
               tilingData.frontCoreNum, tilingData.tailCoreNum,
               tilingData.frontCoreData, tilingData.tailCoreData);
        tilingData.usedCoreNum = tilingData.frontCoreNum + tilingData.tailCoreNum;
    } else {
        // 列小
        // var每次搬64行，索引每次搬多少行根据空间定。
        tilingData.tilingKey = 1;
        tilingData.usedCoreNum -= 1;
        uint64_t MAligned = tilingData.M / VAR_GROUPS * VAR_GROUPS;
        tilingData.MLeft = tilingData.M - MAligned;

        if (MAligned != 0) {
            coreSplit(tilingData.usedCoreNum, MAligned / VAR_GROUPS,
               tilingData.frontCoreNum, tilingData.tailCoreNum,
               tilingData.frontCoreData, tilingData.tailCoreData);
        }
        tilingData.usedCoreNum = tilingData.frontCoreNum + tilingData.tailCoreNum;
        tilingData.usedCoreNum += 1;
    }

    size_t *currentWorkSpace = context->GetWorkspaceSizes(1);
    currentWorkSpace[0] = WORK_SPACE_SIZE;
    SetKernelTiling();
    TilingDataPrint();
    return ge::GRAPH_SUCCESS;
}

class ScatterElementsV2Tiling
{
public:
    explicit ScatterElementsV2Tiling(gert::TilingContext* context) : tilingContext(context) {};
    ge::graphStatus Init();
    ge::graphStatus RunKernelTiling();
    void TilingDataPrint() const;
    bool CacheOpSupport() const;
    bool CacheOpShapeCheck(size_t realDim, size_t inputDimNum) const;
    ge::graphStatus RunCacheOpTiling();
    ge::graphStatus SetCacheOpTiling();

private:
    ScatterElementsV2TilingData tilingData;
    gert::TilingContext* tilingContext = nullptr;
    int32_t needTranspose = 1;
    int32_t mode = 0;
    int32_t updatesIsScalar = 0;
    uint64_t xDim0 = 1;
    uint64_t xDim1 = 1;
    uint64_t indicesDim0 = 1;
    uint64_t indicesDim1 = 1;
    uint64_t updatesDim0 = 1;
    uint64_t updatesDim1 = 1;
    uint32_t tilingKey = 0;
    uint64_t usedCoreNum = 0;
    uint64_t eachNum = 1;
    uint64_t extraTaskCore = 0;
    uint64_t inputCount = 1;
    uint64_t indicesCount = 1;
    uint64_t updatesCount = 1;
    uint64_t inputOneTime = 0;
    uint64_t indicesOneTime = 0;
    uint64_t updatesOneTime = 0;
    uint64_t inputLoop = 0;
    uint64_t indicesLoop = 0;
    uint64_t inputEach = 0;
    uint64_t indicesEach = 0;
    uint64_t inputLast = 0;
    uint64_t indicesLast = 0;
    uint64_t eachPiece = 1;
    uint64_t inputAlign = 8;
    uint64_t indicesAlign = 8;
    uint64_t updatesAlign = 8;
    uint64_t inputOnePiece = 0;
    uint64_t modeFlag = 0;
    uint64_t lastIndicesLoop = 1;
    uint64_t lastIndicesEach = 1;
    uint64_t lastIndicesLast = 1;
    uint64_t oneTime = 1;
    uint64_t lastOneTime = 1;
    uint64_t workspaceSize = 1024 * 1024 * 16;
    uint64_t max_ub = 20480;
};

ge::graphStatus ScatterElementsV2Tiling::Init()
{
    if (tilingContext == nullptr) {
        OP_LOGE("ScatterElementsV2", "tilingContext is nullptr.");
        return ge::GRAPH_FAILED;
    }

    auto compileInfo = tilingContext->GetCompileInfo<ScatterElementsV2CompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, compileInfo);
    static uint32_t coreNum = static_cast<uint32_t>(compileInfo->totalCoreNum);
    if (coreNum == 0) {
        OP_LOGE(tilingContext, "coreNum must greater than 0.");
        return ge::GRAPH_FAILED;
    }
    workspaceSize = compileInfo->workspaceSize;
    auto ubSizePlatForm = compileInfo->ubSizePlatForm;
    max_ub = ubSizePlatForm / max_ub * max_ub / BUFFER_NUM;
    OP_LOGD(tilingContext, "ubSizePlatForm: %lu.", ubSizePlatForm);

    auto attrs = tilingContext->GetAttrs();
    if (attrs == nullptr || tilingContext->GetInputShape(INPUT_0) == nullptr ||
        tilingContext->GetInputShape(INPUT_1) == nullptr || tilingContext->GetInputShape(INPUT_2) == nullptr ||
        tilingContext->GetInputDesc(INPUT_0) == nullptr || tilingContext->GetRawTilingData() == nullptr) {
        OP_LOGE(tilingContext, "tilingContext inputshape or outputshape is nullptr.");
        return ge::GRAPH_FAILED;
    }
    auto inputDtype = tilingContext->GetInputDesc(INPUT_0)->GetDataType();
    uint32_t inputSize = 0;
    if (ge::DT_FLOAT == inputDtype) {
        inputSize = SIZE_OF_FP32;
    } else if (ge::DT_FLOAT16 == inputDtype) {
        inputSize = SIZE_OF_FP16;
    } else if (ge::DT_INT32 == inputDtype) {
        inputSize = SIZE_OF_INT32;
    } else if (ge::DT_UINT8 == inputDtype) {
        inputSize = SIZE_OF_UINT8;
    } else if (ge::DT_INT8 == inputDtype) {
        inputSize = SIZE_OF_INT8;
    } else if (ge::DT_BF16 == inputDtype) {
        inputSize = SIZE_OF_BF16;
    } else {
        OP_LOGE(tilingContext, "var only support float, float16, int32, uint8, int8, bf16.");
        return ge::GRAPH_FAILED;
    }
    uint32_t indicesSize = 0;
    auto indicesDtype = tilingContext->GetInputDesc(1)->GetDataType();
    if (ge::DT_INT32 == indicesDtype) {
        indicesSize = SIZE_OF_INT32;
    } else if (ge::DT_INT64 == indicesDtype) {
        indicesSize = SIZE_OF_INT64;
    } else {
        OP_LOGE(tilingContext, "indices only support int64, int32.");
        return ge::GRAPH_FAILED;
    }

    uint32_t dataAlign = 32;
    uint32_t inputDataAlign = dataAlign / inputSize;
    uint32_t indexDataAlign = dataAlign / indicesSize;

    const int* dim = (attrs->GetAttrPointer<int>(0));
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, dim);
    const char* reduce = attrs->GetAttrPointer<char>(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, reduce);
    auto inputShape = tilingContext->GetInputShape(INPUT_0)->GetStorageShape();
    auto indicesShape = tilingContext->GetInputShape(INPUT_1)->GetStorageShape();
    auto updatesShape = tilingContext->GetInputShape(INPUT_2)->GetStorageShape();
    auto inputDimNum = inputShape.GetDimNum();
    if (strcmp(reduce, "none") == 0) {
        tilingKey += OPERATOR_TYPE * NONE;
    } else if (strcmp(reduce, "add") == 0) {
        tilingKey += OPERATOR_TYPE * ADD;
    } else if (strcmp(reduce, "mul") == 0) {
        tilingKey += OPERATOR_TYPE * MUL;
        OP_LOGE(tilingContext, "scatter_elements_v2 not support mul.");
        return ge::GRAPH_FAILED;
    } else {
        OP_LOGE(tilingContext, "scatter_elements_v2 only support none, add.");
        return ge::GRAPH_FAILED;
    }

    if (inputDimNum != indicesShape.GetDimNum() ||
        inputDimNum != tilingContext->GetInputShape(INPUT_2)->GetStorageShape().GetDimNum()) {
        OP_LOGE(tilingContext, "the dimNum of input must equal the dimNum of indices.");
        return ge::GRAPH_FAILED;
    }

    uint32_t realDim = 0;
    if (*dim < 0) {
        realDim = *dim + inputDimNum;
    } else {
        realDim = *dim;
    }

    if (realDim != inputDimNum - 1) {
        OP_LOGE(tilingContext, "scatter_elements_v2 not support dim != -1.");
        return ge::GRAPH_FAILED;
    }

    for (uint32_t i = 0; i < inputDimNum; ++i) {
        auto dimInput = inputShape.GetDim(i);
        auto dimIndices = indicesShape.GetDim(i);
        auto dimUpdates = updatesShape.GetDim(i);
        if (dimUpdates < dimIndices) {
            OP_LOGE(tilingContext, "the dim of updates must greater than or equal the dim of indices.");
            return ge::GRAPH_FAILED;
        }
        if (realDim == i) {
            inputOneTime = dimInput;
            indicesOneTime = dimIndices;
            updatesOneTime = dimUpdates;
        }
        inputCount *= dimInput;
        indicesCount *= dimIndices;
        updatesCount *= dimUpdates;
    }

    if (inputOneTime == 0 || indicesOneTime == 0 || updatesOneTime == 0 || inputCount == 0 || indicesCount == 0 ||
        updatesCount == 0) {
        OP_LOGE(tilingContext, "shape cannot equal 0.");
        return ge::GRAPH_FAILED;
    }

    if (ge::DT_INT64 == indicesDtype) {
        indicesSize += SIZE_OF_INT32;
    }
    if ((strcmp(reduce, "add") == 0) && (ge::DT_FLOAT16 == inputDtype || ge::DT_BF16 == inputDtype)) {
        inputSize += sizeof(float) / BUFFER_NUM;
        indicesSize += sizeof(float) / BUFFER_NUM;
    }

    uint32_t times = indicesCount / indicesOneTime;
    uint32_t totalSize = inputSize * inputOneTime + indicesSize * (updatesOneTime + indicesOneTime);
    // 小包场景，一次性可以搬入一轮尾轴，按ub上限尽可能的搬多轮，特殊处理走small分支
    if (totalSize <= max_ub) {
        modeFlag = SMALL_MODE;
        max_ub = max_ub / totalSize;
        if (times < coreNum) {
            usedCoreNum = times;
            indicesEach = 1;
            indicesLoop = 1;
            indicesLast = 1;
        } else {
            oneTime = (times + coreNum - 1) / coreNum;
            usedCoreNum = (times + oneTime - 1) / oneTime;
            indicesLoop = (oneTime + max_ub - 1) / max_ub;
            indicesEach = indicesLoop == 1 ? oneTime : max_ub;
            indicesLast = oneTime - (indicesLoop - 1) * indicesEach;

            lastOneTime = times - oneTime * (usedCoreNum - 1);
            lastIndicesLoop = (lastOneTime + max_ub - 1) / max_ub;
            lastIndicesEach = lastIndicesLoop == 1 ? lastOneTime : max_ub;
            lastIndicesLast = lastOneTime - (lastIndicesLoop - 1) * lastIndicesEach;
        }
        indicesAlign = ((indicesEach - 1) / indexDataAlign + 1) * indexDataAlign;

        OP_LOGD(tilingContext, "Tiling inited.");
        return ge::GRAPH_SUCCESS;
    }

    inputOnePiece = inputOneTime;
    if (times < coreNum) {                                          // 一个任务可以分给多个核
        uint32_t need = (inputOneTime + dataAlign - 1) / dataAlign; // 每个核至少处理32个数，每个任务需要多少个核
        eachPiece = coreNum / times;                                // 每个任务可用的核数
        eachPiece = eachPiece > need ? need : eachPiece;
        eachNum = eachPiece == 1 ? 1 : 0;
        extraTaskCore = 0;
        inputOnePiece = (inputOneTime + eachPiece - 1) / eachPiece; // 每个核需要处理的数量
        usedCoreNum = (inputOneTime + inputOnePiece - 1) / inputOnePiece *
                      times; // 再次根据每个核需要处理的数量重新计算需要的核数
    } else {                 // 一个任务单独由一个核处理
        usedCoreNum = coreNum;
        eachNum = times / coreNum;
        extraTaskCore = times - eachNum * coreNum;
    }

    uint32_t indicesSum = indicesOneTime * (inputSize + indicesSize);
    uint32_t inputSum = inputOnePiece * inputSize;
    if (inputSum + indicesSum > static_cast<uint32_t>(max_ub)) {
        if (indicesSum < static_cast<uint32_t>(max_ub / HALF_UB)) {
            inputEach = (max_ub - indicesSum) / inputSize;
            inputEach = inputEach > inputOnePiece ? inputOnePiece : inputEach;
            inputLoop = (inputOnePiece - 1) / inputEach + 1;
            inputLast = inputOnePiece - inputEach * (inputLoop - 1);

            indicesLoop = 1;
            indicesEach = indicesLast = indicesOneTime;
        } else if (inputSum < static_cast<uint32_t>(max_ub / HALF_UB)) {
            indicesEach = (max_ub - inputSum) / (inputSize + indicesSize);
            indicesEach = indicesEach > indicesOneTime ? indicesOneTime : indicesEach;
            indicesLoop = (indicesOneTime - 1) / indicesEach + 1;
            indicesLast = indicesOneTime - indicesEach * (indicesLoop - 1);

            inputLoop = 1;
            inputEach = inputLast = inputOnePiece;
        } else {
            inputEach = max_ub / (HALF_UB * inputSize);
            inputEach = inputEach > inputOnePiece ? inputOnePiece : inputEach;
            inputLoop = (inputOnePiece - 1) / inputEach + 1;
            inputLast = inputOnePiece - inputEach * (inputLoop - 1);

            indicesEach = max_ub / (HALF_UB * (inputSize + indicesSize));
            indicesEach = indicesEach > indicesOneTime ? indicesOneTime : indicesEach;
            indicesLoop = (indicesOneTime - 1) / indicesEach + 1;
            indicesLast = indicesOneTime - indicesEach * (indicesLoop - 1);
        }
    } else {
        inputLoop = indicesLoop = 1;
        inputEach = inputLast = inputOnePiece;
        indicesEach = indicesLast = indicesOneTime;
    }

    inputAlign = ((inputEach - 1) / inputDataAlign + 1) * inputDataAlign;
    indicesAlign = ((indicesEach - 1) / indexDataAlign + 1) * indexDataAlign;
    updatesAlign = ((indicesEach - 1) / inputDataAlign + 1) * inputDataAlign;

    OP_LOGD(tilingContext, "Tiling inited.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterElementsV2Tiling::RunKernelTiling()
{
    OP_LOGD(tilingContext, "Tiling start.");

    tilingData.set_usedCoreNum(usedCoreNum);
    tilingData.set_eachNum(eachNum);
    tilingData.set_extraTaskCore(extraTaskCore);
    tilingData.set_eachPiece(eachPiece);
    tilingData.set_inputAlign(inputAlign);
    tilingData.set_indicesAlign(indicesAlign);
    tilingData.set_updatesAlign(updatesAlign);
    tilingData.set_inputCount(inputCount);
    tilingData.set_indicesCount(indicesCount);
    tilingData.set_updatesCount(updatesCount);
    tilingData.set_inputOneTime(inputOneTime);
    tilingData.set_indicesOneTime(indicesOneTime);
    tilingData.set_updatesOneTime(updatesOneTime);
    tilingData.set_inputEach(inputEach);
    tilingData.set_indicesEach(indicesEach);
    tilingData.set_inputLast(inputLast);
    tilingData.set_indicesLast(indicesLast);
    tilingData.set_inputLoop(inputLoop);
    tilingData.set_indicesLoop(indicesLoop);
    tilingData.set_inputOnePiece(inputOnePiece);
    tilingData.set_modeFlag(modeFlag);
    tilingData.set_lastIndicesLoop(lastIndicesLoop);
    tilingData.set_lastIndicesEach(lastIndicesEach);
    tilingData.set_lastIndicesLast(lastIndicesLast);
    tilingData.set_oneTime(oneTime);
    tilingData.set_lastOneTime(lastOneTime);
    tilingData.SaveToBuffer(
        tilingContext->GetRawTilingData()->GetData(), tilingContext->GetRawTilingData()->GetCapacity());
    tilingContext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    tilingContext->SetTilingKey(tilingKey);
    tilingContext->SetBlockDim(usedCoreNum);
    size_t* workspaces = tilingContext->GetWorkspaceSizes(1);
    workspaces[0] = workspaceSize;
    TilingDataPrint();
    OP_LOGD(tilingContext, "Tiling end.");
    return ge::GRAPH_SUCCESS;
}

void ScatterElementsV2Tiling::TilingDataPrint() const
{
    OP_LOGD(tilingContext, "usedCoreNum: %lu.", usedCoreNum);
    OP_LOGD(tilingContext, "eachNum: %lu.", eachNum);
    OP_LOGD(tilingContext, "extraTaskCore: %lu.", extraTaskCore);
    OP_LOGD(tilingContext, "eachPiece: %lu.", eachPiece);
    OP_LOGD(tilingContext, "inputAlign: %lu.", inputAlign);
    OP_LOGD(tilingContext, "indicesAlign: %lu.", indicesAlign);
    OP_LOGD(tilingContext, "updatesAlign: %lu.", updatesAlign);
    OP_LOGD(tilingContext, "inputCount: %lu.", inputCount);
    OP_LOGD(tilingContext, "indicesCount: %lu.", indicesCount);
    OP_LOGD(tilingContext, "updatesCount: %lu.", updatesCount);
    OP_LOGD(tilingContext, "inputOneTime: %lu.", inputOneTime);
    OP_LOGD(tilingContext, "indicesOneTime: %lu.", indicesOneTime);
    OP_LOGD(tilingContext, "updatesOneTime: %lu.", updatesOneTime);
    OP_LOGD(tilingContext, "inputEach: %lu.", inputEach);
    OP_LOGD(tilingContext, "indicesEach: %lu.", indicesEach);
    OP_LOGD(tilingContext, "inputLast: %lu.", inputLast);
    OP_LOGD(tilingContext, "indicesLast: %lu.", indicesLast);
    OP_LOGD(tilingContext, "inputLoop: %lu.", inputLoop);
    OP_LOGD(tilingContext, "indicesLoop: %lu.", indicesLoop);
    OP_LOGD(tilingContext, "inputOnePiece: %lu.", inputOnePiece);
    OP_LOGD(tilingContext, "modeFlag: %lu.", modeFlag);
    OP_LOGD(tilingContext, "lastIndicesLoop: %lu.", lastIndicesLoop);
    OP_LOGD(tilingContext, "lastIndicesEach: %lu.", lastIndicesEach);
    OP_LOGD(tilingContext, "lastIndicesLast: %lu.", lastIndicesLast);
    OP_LOGD(tilingContext, "oneTime: %lu.", oneTime);
    OP_LOGD(tilingContext, "lastOneTime: %lu.", lastOneTime);
    OP_LOGD(tilingContext, "tilingKey: %u.", tilingKey);
    OP_LOGD(tilingContext, "max_ub: %lu.", max_ub);
}

bool ScatterElementsV2Tiling::CacheOpShapeCheck(size_t realDim, size_t inputDimNum) const {
    auto inputShape = tilingContext->GetInputShape(INPUT_0)->GetStorageShape();
    auto indicesShape = tilingContext->GetInputShape(INPUT_1)->GetStorageShape();
    auto updatesShape = tilingContext->GetInputShape(INPUT_2)->GetStorageShape();
    if ((updatesShape.GetDimNum() == 1 && updatesShape.GetDim(0) == 1) || updatesShape.GetDimNum() == 0) {
        OP_LOGD("ScatterElementsV2", "updates is scalar Tensor.");
        updatesShape = indicesShape;
    }

    size_t tasks = 1;
    // 二维首轴
    if (realDim == 0 && inputDimNum == TWO_DIM) {
        tasks *= indicesShape.GetDim(1);
    }
    // 多维首轴 要求后面维度完全相同
    if (realDim == 0 && inputDimNum != TWO_DIM) {
        for (size_t i = 1; i < inputDimNum; i++) {
            auto varDimValue = inputShape.GetDim(i);
            auto indicesDimValue = indicesShape.GetDim(i);
            auto updatesDimValue = updatesShape.GetDim(i);
            if (!(varDimValue == indicesDimValue && varDimValue == updatesDimValue)) {
                OP_LOGD("ScatterElementsV2", "Low memory when realDim = 0, only support var、indices、updates dim equal.");
                return false;
            }
            tasks *= indicesDimValue;
        }
    }
    // 尾轴 要求前面维度除了第一维度外完全相同
    if (realDim == inputDimNum - 1) {
        tasks *= indicesShape.GetDim(0);
        for (size_t i = 1; i < inputDimNum - 1; i++) {
            auto varDimValue = inputShape.GetDim(i);
            auto indicesDimValue = indicesShape.GetDim(i);
            auto updatesDimValue = updatesShape.GetDim(i);
            if (!(varDimValue == indicesDimValue && varDimValue == updatesDimValue)) {
                OP_LOGD("ScatterElementsV2", "Low memory when realDim = -1, only support var、indices、updates dim equal.");
                return false;
            }
            tasks *= indicesDimValue;
        }
    }
    if (inputShape.GetDim(realDim) > NO_TRANSPOSE_DIM_MAX) {
        OP_LOGD("ScatterElementsV2", "Low memory only support var dim value in axis <= 256.");
        return false;
    }

    if (tasks < NO_TRANSPOSE_TASKS_MIN) {
        OP_LOGD("ScatterElementsV2", "Low memory only support var tasks >= 768.");
        return false;
    }
    if (indicesShape.GetDim(realDim) != updatesShape.GetDim(realDim)) {
        OP_LOGD("ScatterElementsV2", "Low memory only support indices and updates dim value equal in axis");
        return false;
    }
    return true;
}

bool ScatterElementsV2Tiling::CacheOpSupport() const {
    if (tilingContext == nullptr) {
        OP_LOGD("ScatterElementsV2", "tilingContext is nullptr.");
        return false;
    }

    auto attrs = tilingContext->GetAttrs();
    auto inputDtype = tilingContext->GetInputDesc(INPUT_0)->GetDataType();
    if (inputDtype != ge::DT_FLOAT && inputDtype != ge::DT_BOOL) {
        OP_LOGD("ScatterElementsV2", "Low memory only support var float or bool.");
        return false;
    }

    const int* dim = (attrs->GetAttrPointer<int>(0));
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, dim);
    const char* reduce = attrs->GetAttrPointer<char>(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, reduce);
    auto inputShape = tilingContext->GetInputShape(INPUT_0)->GetStorageShape();
    auto indicesShape = tilingContext->GetInputShape(INPUT_1)->GetStorageShape();
    auto updatesShape = tilingContext->GetInputShape(INPUT_2)->GetStorageShape();
    auto inputDimNum = inputShape.GetDimNum();
    uint32_t realDim = (*dim < 0 ? *dim + inputDimNum : *dim);
    if (updatesShape.GetDimNum() == indicesShape.GetDimNum() && strcmp(reduce, "none") == 0 && 
        realDim == inputDimNum - 1 && inputDtype == ge::DT_FLOAT) {
        OP_LOGD("ScatterElementsV2 when reduce=none, axis=-1 and fp32 will use original.");
        return false;
    }

    if ((updatesShape.GetDimNum() == 1 && updatesShape.GetDim(0) == 1) || updatesShape.GetDimNum() == 0) {
        OP_LOGD("ScatterElementsV2", "updates is scalar Tensor.");
        updatesShape = indicesShape;
    }
    if (inputDimNum < TWO_DIM) {
        OP_LOGD("ScatterElementsV2", "Low memory only support var dims >= 2.");
        return false;
    }
    if (strcmp(reduce, "mul") == 0) {
        OP_LOGD("ScatterElementsV2", "Low memory not reduce=mul.");
        return false;
    }
    if (realDim != 0 && realDim != inputDimNum - 1) {
        OP_LOGD("ScatterElementsV2", "Low memory only support axis in first dim or last dim.");
        return false;
    }
    return CacheOpShapeCheck(realDim, inputDimNum);
}

ge::graphStatus ScatterElementsV2Tiling::SetCacheOpTiling() {
    ScatterElementsV2TilingData tiling;
    tiling.set_needTranspose(needTranspose);
    tiling.set_mode(mode);
    tiling.set_coreNums(usedCoreNum);
    tiling.set_updatesIsScalar(updatesIsScalar);
    tiling.set_xDim0(xDim0);
    tiling.set_xDim1(xDim1);
    tiling.set_indicesDim0(indicesDim0);
    tiling.set_indicesDim1(indicesDim1);
    tiling.set_updatesDim0(updatesDim0);
    tiling.set_updatesDim1(updatesDim1);
    OP_LOGD(tilingContext, "needTranspose: %d.", needTranspose);
    OP_LOGD(tilingContext, "mode: %d.", mode);
    OP_LOGD(tilingContext, "coreNums: %d.", usedCoreNum);
    OP_LOGD(tilingContext, "updatesIsScalar: %d.", updatesIsScalar);
    OP_LOGD(tilingContext, "xDim0: %lu.", xDim0);
    OP_LOGD(tilingContext, "xDim1: %lu.", xDim1);
    OP_LOGD(tilingContext, "indicesDim0: %lu.", indicesDim0);
    OP_LOGD(tilingContext, "indicesDim1: %lu.", indicesDim1);
    OP_LOGD(tilingContext, "updatesDim0: %lu.", updatesDim0);
    OP_LOGD(tilingContext, "updatesDim1: %lu.", updatesDim1);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingContext->GetPlatformInfo());
    size_t libApiworkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t *currentWorkSpace = tilingContext->GetWorkspaceSizes(1);
    currentWorkSpace[0] = libApiworkspaceSize;
    tilingContext->SetBlockDim(usedCoreNum);
    tilingContext->SetTilingKey(0);
    tiling.SaveToBuffer(tilingContext->GetRawTilingData()->GetData(), tilingContext->GetRawTilingData()->GetCapacity());
    tilingContext->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterElementsV2Tiling::RunCacheOpTiling() {
    auto attrs = tilingContext->GetAttrs();
    const int* dim = (attrs->GetAttrPointer<int>(0));
    const char* reduce = attrs->GetAttrPointer<char>(1);
    if (strcmp(reduce, "none") == 0) {
        mode = 0;
    } else {
        mode = 1;
    }

    auto inputShape = tilingContext->GetInputShape(INPUT_0)->GetStorageShape();
    auto indicesShape = tilingContext->GetInputShape(INPUT_1)->GetStorageShape();
    auto updatesShape = tilingContext->GetInputShape(INPUT_2)->GetStorageShape();
    if ((updatesShape.GetDimNum() == 1 && updatesShape.GetDim(0) == 1) || updatesShape.GetDimNum() == 0) {
        updatesIsScalar = 1;
        updatesShape = indicesShape;
    } else {
        updatesIsScalar = 0;
    }
    auto inputDimNum = inputShape.GetDimNum();
    uint32_t realDim = (*dim < 0 ? *dim + inputDimNum : *dim);
    if (realDim == 0) {
        needTranspose = 1;
    } else {
        needTranspose = 0;
    }

    if (realDim == 0) {
        xDim0 = inputShape.GetDim(0);
        indicesDim0 = indicesShape.GetDim(0);
        updatesDim0 = updatesShape.GetDim(0);
        for (size_t i = 1; i < inputDimNum; i++) {
            xDim1 *= inputShape.GetDim(i);
            indicesDim1 *= indicesShape.GetDim(i);
            updatesDim1 *= updatesShape.GetDim(i);
        }
    } else {
        xDim1 = inputShape.GetDim(inputDimNum - 1);
        indicesDim1 = indicesShape.GetDim(inputDimNum - 1);
        updatesDim1 = updatesShape.GetDim(inputDimNum - 1);
        for (size_t i = 0; i < inputDimNum - 1; i++) {
            xDim0 *= inputShape.GetDim(i);
            indicesDim0 *= indicesShape.GetDim(i);
            updatesDim0 *= updatesShape.GetDim(i);
        }
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingContext->GetPlatformInfo());
    usedCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_LOGD(tilingContext, "ascendcPlatform CoreNum is: %d.", usedCoreNum);
    return SetCacheOpTiling();
}

ge::graphStatus TilingScatterElementsV2(gert::TilingContext* context)
{
    auto compile_info = reinterpret_cast<const ScatterElementsV2CompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);
    if (compile_info->is_regbase) {
        return Ops::NN::Optiling::TilingRegistry::GetInstance().DoTilingImpl(context);
    }

    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    auto socVersion = ascendcPlatform.GetSocVersion();
    if (socVersion == platform_ascendc::SocVersion::ASCEND310P) {
        ScatterElementsV2Tiling310P tilingObject(context);
        return tilingObject.Init();
    }  else {
        ScatterElementsV2Tiling tilingObject(context);
        if (tilingObject.CacheOpSupport()) {
            return tilingObject.RunCacheOpTiling();
        } else {
            if (tilingObject.Init() != ge::GRAPH_SUCCESS) {
                return ge::GRAPH_FAILED;
            }
            return tilingObject.RunKernelTiling();
        }
    }
}

ge::graphStatus TilingPrepareForScatterElementsV2(gert::TilingParseContext* context)
{
    OP_LOGD(context, "TilingPrepareForScatterElementsV2 start.");
    auto compileInfo = context->GetCompiledInfo<ScatterElementsV2CompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    compileInfo->workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSizePlatForm = static_cast<int64_t>(ubSizePlatForm);
    OP_CHECK_IF(
        (compileInfo->ubSizePlatForm <= 0), OP_LOGE(context, "Failed to get ub size."), return ge::GRAPH_FAILED);
    OP_LOGD(context, "ub_size_platform is %lu.", compileInfo->ubSizePlatForm);
    uint64_t totalUbSize = 0;
    platformInfo->GetLocalMemSize(fe::LocalMemType::UB, totalUbSize);
    compileInfo->is_regbase = IsRegbaseSocVersion(context);
    OP_LOGD(context, "total_ub_size is %lu.", totalUbSize);
    OP_LOGD(context, "TilingPrepareForScatterElementsV2 end.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ScatterElementsV2)
    .Tiling(TilingScatterElementsV2)
    .TilingParse<ScatterElementsV2CompileInfo>(TilingPrepareForScatterElementsV2);
} // namespace optiling
