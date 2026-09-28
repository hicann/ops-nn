/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
#include "op_host/tiling_templates_registry.h"
#include "scatter_elements_v2_tiling.h"

using namespace std;
using Ops::NN::Optiling::TilingRegistry;

namespace {
const int INPUT_TYPE = 100;
const int INDIC_TYPE = 10;
const int OPERATOR_TYPE = 1;

const int SIZE_OF_FP16 = 2;
const int SIZE_OF_FP32 = 4;
const int SIZE_OF_INT32 = 4;
const int SIZE_OF_INT64 = 8;
const int SIZE_OF_INT16 = 2;
const int SIZE_OF_UINT8 = 1;
const int SIZE_OF_INT8 = 1;
const int SIZE_OF_BF16 = 2;

const int DT_FLOAT32_TYPE = 1;
const int DT_FLOAT16_TYPE = 2;
const int DT_INT32_TYPE = 3;
const int DT_UINT8_TYPE = 4;
const int DT_INT8_TYPE = 5;
const int DT_BF16_TYPE = 6;
const int DT_INT16_TYPE = 7;
const int DT_INT64_TYPE = 8;
const int DT_DOUBLE_TYPE = 9;
const int DT_INT32_INDEX_TYPE = 1;
const int DT_INT64_INDEX_TYPE = 2;

const int NONE = 1;
const int ADD = 2;
const int MUL = 3;
const int MIN = 4;
const int MAX = 5;
const int MEAN = 6;
const int BUFFER_NUM = 1;
const int HALF_UB = 2;

const int INPUT_0 = 0;
const int INPUT_1 = 1;
const int INPUT_2 = 2;
const int SMALL_MODE = 1;
const int VAR_LIMIT = 128;

const size_t TWO_DIM = 2;
const int64_t NO_TRANSPOSE_DIM_MAX = 256;
const size_t NO_TRANSPOSE_TASKS_MIN = 768;

const uint32_t ALIGN_SIZE = 32;
const uint32_t TILE_SIZE = 5;
const uint32_t AGG_INDICES_NUM = 1024;
const uint32_t CACHE_OP_X_LOCAL_LENGTH = 40000;
const uint32_t CACHE_OP_X_LOCAL_LENGTH_REDUCE = 20480;
const uint32_t CACHE_OP_INDICES_LOCAL_LENGTH = 2048;
const uint32_t CACHE_OP_ALL_UB_SIZE = 16384 * 3 * SIZE_OF_FP32;
const uint32_t STABLE_BUCKET_MIN_INDICES = 256;
// Bucket construction has a fixed scalar classification cost. Keep it for
// wide rows where it can skip a meaningful number of output tiles.
const uint32_t STABLE_BUCKET_MIN_INPUT_LOOPS = 8;
const uint32_t STABLE_BUCKET_MAX_INPUT_LOOPS = 512;

const uint64_t WORKSPACE_GATHER_FOUR_BY_TWO = 1;
const uint64_t WORKSPACE_GATHER_FOUR_BY_ONE = 2;
// The single-owner plan avoids replaying source chunks for narrow index rows.
// The crossover was established from the max-unpool2d workspace-gather
// benchmark: three 128-element source groups (384) and a 768-element output
// row are the points where one owner remains faster than two destination
// owners while keeping all four source chunks resident.
const uint64_t WORKSPACE_SINGLE_OWNER_SOURCE_GROUP_WIDTH = 128;
const uint64_t WORKSPACE_SINGLE_OWNER_OUTPUT_GROUP_WIDTH = 256;
const uint64_t WORKSPACE_SINGLE_OWNER_MAX_SOURCE_WIDTH = 3 * WORKSPACE_SINGLE_OWNER_SOURCE_GROUP_WIDTH;
const uint64_t WORKSPACE_SINGLE_OWNER_MIN_OUTPUT_WIDTH = 3 * WORKSPACE_SINGLE_OWNER_OUTPUT_GROUP_WIDTH;

// 分桶散射分支参数（与 op_kernel/scatter_elements_v2_bucket_scatter.h 中的常量保持一致）。
// 注意与上方 STABLE_BUCKET_* 无关：那组是 legacy kernel 的稳定分桶，此处是稀疏散射分桶。
const uint64_t BUCKET_MIN_VAR_N = 65536;     // var 末轴长度下限：更小的规模主路径已足够高效
const uint64_t BUCKET_SPARSE_RATIO = 8;      // 稀疏判据：indicesN * RATIO <= varN 才启用
const uint64_t BUCKET_UPD_CHUNK_HOST = 8192; // 流式块元素数，须等于 kernel 的 BUCKET_UPD_CHUNK
// UB 余量：需覆盖 kernel 侧 metaBuf(4*numTiles*sizeof(int32)) 及对齐开销。numTiles 由 tileLen 反算、
// 与 tileLen 互为依赖，故不精确建模，改用足量固定余量（numTiles 上限约 512 时 metaBuf 约 8KB）。
const uint64_t BUCKET_UB_MARGIN = 12288;
const uint64_t BUCKET_FIFO_BUDGET = 24576; // 所有桶 FIFO 合计字节预算
const uint64_t BUCKET_ALIGN_HOST = 16;     // 桶起点/容量对齐粒度，须等于 kernel 的 BUCKET_ALIGN
const uint64_t BUCKET_FIFO_MAX = 64;       // 每桶 FIFO 深度上限
const uint64_t BUCKET_POW2_BASE = 2;       // tileLen 取 2 的幂时的底数
// 单桶在 GM 桶区的最坏额外开销（条目数）。kernel 侧每桶按
// padded = ceil(cnt, BUCKET_ALIGN) * BUCKET_ALIGN + BUCKET_ALIGN(guard) 预留，
// 而 ceil(cnt, 16) * 16 <= cnt + 15，故 padded <= cnt + 31；
// 取 2 * BUCKET_ALIGN_HOST = 32 作为对齐友好的上界（须 >= 31）。
const uint64_t BUCKET_PAD_PER_TILE_HOST = 2 * BUCKET_ALIGN_HOST;

} // namespace

namespace optiling {
using namespace Ops::NN::OpTiling;

bool IsRegbaseSocVersion4Scatter(const gert::TilingParseContext* context)
{
    return Ops::NN::OpTiling::IsRegbaseSocVersion(context);
}

static bool IsArch22DeterministicReduction(const char* reduce)
{
    return reduce != nullptr && (strcmp(reduce, "mul") == 0 || strcmp(reduce, "min") == 0 ||
                                 strcmp(reduce, "max") == 0 || strcmp(reduce, "mean") == 0);
}

static bool IsArch22DeterministicMode(const gert::TilingContext* context, const char* reduce)
{
    if (context == nullptr || !context->GetDeterministic() || !IsArch22DeterministicReduction(reduce)) {
        return false;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto socVersion = ascendcPlatform.GetSocVersion();
    return socVersion == platform_ascendc::SocVersion::ASCEND910B ||
           socVersion == platform_ascendc::SocVersion::ASCEND910_93;
}

bool IsRegbaseSocVersion4Scatter(const gert::TilingContext* context)
{
    return Ops::NN::OpTiling::IsRegbaseSocVersion(context);
}
class ScatterElementsV2Tiling {
public:
    explicit ScatterElementsV2Tiling(gert::TilingContext* context) : tilingContext(context) {};
    ge::graphStatus Init();
    ge::graphStatus RunKernelTiling();
    void TilingDataPrint() const;
    bool CacheOpSupport();
    ge::graphStatus RunCacheOpTiling();
    ge::graphStatus SetCacheOpTiling();
    bool BucketScatterSupport();
    ge::graphStatus RunBucketScatterTiling();

private:
    void SetTilingData(ScatterElementsV2TilingData& tiling);
    void LogTilingData() const;
    size_t CalculateWorkspaceSize();
    void ParseAttrs();
    void ProcessUpdatesShape(gert::Shape& updatesShape, const gert::Shape& indicesShape);
    void SetDimsForFirstAxis(const gert::Shape& inputShape, const gert::Shape& indicesShape,
                             const gert::Shape& updatesShape, size_t inputDimNum);
    void SetDimsForMiddleAxis(const gert::Shape& inputShape, const gert::Shape& indicesShape,
                              const gert::Shape& updatesShape, size_t inputDimNum);
    void SetDimsForLastAxis(const gert::Shape& inputShape, const gert::Shape& indicesShape,
                            const gert::Shape& updatesShape, size_t inputDimNum);
    void SetDimsByAxisType(const gert::Shape& inputShape, const gert::Shape& indicesShape,
                           const gert::Shape& updatesShape, size_t inputDimNum);
    // 从 context 重新归约 xDim*/indicesDim*/updatesDim*/batchSize/updatesIsScalar（幂等，可重复调用）
    bool ResolveScatterDims();
    bool CheckCacheOpShapeLimit(const gert::Shape& xShape, const char* reduce) const;
    bool CheckCacheOpDtype(ge::DataType inputDtype, const char* reduce) const;
    bool CheckLastAxisCacheOp(size_t inputDimNum, ge::DataType inputDtype, const char* reduce) const;
    bool CheckCacheOpXDim1Limit(const gert::Shape& inputShape, const gert::Shape& indicesShape,
                                const gert::Shape& updatesShape, size_t inputDimNum, ge::DataType inputDtype,
                                const char* reduce);
    uint64_t GetCacheOpMaxXDim1(ge::DataType inputDtype, const char* reduce) const;
    ScatterElementsV2TilingData tilingData;
    gert::TilingContext* tilingContext = nullptr;
    int32_t mode = 0;
    int32_t updatesIsScalar = 0;
    uint64_t xDim0 = 1;
    uint64_t xDim1 = 1;
    uint64_t indicesDim0 = 1;
    uint64_t indicesDim1 = 1;
    uint64_t updatesDim0 = 1;
    uint64_t updatesDim1 = 1;
    uint64_t batchSize = 1;
    uint64_t realDim = 0;
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
    uint64_t includeSelf = 1;
    uint64_t executionPlan = 0;
    uint64_t useStableBucket = 0;
    uint64_t lastIndicesLoop = 1;
    uint64_t lastIndicesEach = 1;
    uint64_t lastIndicesLast = 1;
    uint64_t oneTime = 1;
    uint64_t lastOneTime = 1;
    uint64_t workspaceSize = 1024 * 1024 * 16;
    uint64_t max_ub = 20480;
    bool isDeterministic = false;
};

ge::graphStatus ScatterElementsV2Tiling::Init()
{
    if (tilingContext == nullptr) {
        OP_LOGE("ScatterElementsV2", "tilingContext is nullptr.");
        return ge::GRAPH_FAILED;
    }

    auto compileInfo = tilingContext->GetCompileInfo<ScatterElementsV2CompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, compileInfo);
    uint32_t coreNum = static_cast<uint32_t>(compileInfo->totalCoreNum);
    if (coreNum == 0) {
        OP_LOGE(tilingContext, "coreNum must be greater than 0.");
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
        tilingKey += INPUT_TYPE * DT_FLOAT32_TYPE;
        inputSize = SIZE_OF_FP32;
    } else if (ge::DT_FLOAT16 == inputDtype) {
        tilingKey += INPUT_TYPE * DT_FLOAT16_TYPE;
        inputSize = SIZE_OF_FP16;
    } else if (ge::DT_INT32 == inputDtype) {
        tilingKey += INPUT_TYPE * DT_INT32_TYPE;
        inputSize = SIZE_OF_INT32;
    } else if (ge::DT_INT16 == inputDtype) {
        tilingKey += INPUT_TYPE * DT_INT16_TYPE;
        inputSize = SIZE_OF_INT16;
    } else if (ge::DT_INT64 == inputDtype) {
        tilingKey += INPUT_TYPE * DT_INT64_TYPE;
        inputSize = SIZE_OF_INT64;
    } else if (ge::DT_DOUBLE == inputDtype) {
        tilingKey += INPUT_TYPE * DT_DOUBLE_TYPE;
        inputSize = SIZE_OF_INT64;
    } else if (ge::DT_UINT8 == inputDtype) {
        tilingKey += INPUT_TYPE * DT_UINT8_TYPE;
        inputSize = SIZE_OF_UINT8;
    } else if (ge::DT_INT8 == inputDtype) {
        tilingKey += INPUT_TYPE * DT_INT8_TYPE;
        inputSize = SIZE_OF_INT8;
    } else if (ge::DT_BF16 == inputDtype) {
        tilingKey += INPUT_TYPE * DT_BF16_TYPE;
        inputSize = SIZE_OF_BF16;
    } else {
        OP_LOGE(tilingContext, "var only support float, float16, int32, int16, int64, double, uint8, int8, bf16.");
        return ge::GRAPH_FAILED;
    }
    uint32_t indicesSize = 0;
    auto indicesDtype = tilingContext->GetInputDesc(1)->GetDataType();
    if (ge::DT_INT32 == indicesDtype) {
        tilingKey += INDIC_TYPE * DT_INT32_INDEX_TYPE;
        indicesSize = SIZE_OF_INT32;
    } else if (ge::DT_INT64 == indicesDtype) {
        tilingKey += INDIC_TYPE * DT_INT64_INDEX_TYPE;
        indicesSize = SIZE_OF_INT64;
    } else {
        OP_LOGE(tilingContext, "indices only support int64, int32.");
        return ge::GRAPH_FAILED;
    }

    uint32_t dataAlign = 32;
    uint32_t inputDataAlign = dataAlign / inputSize;
    uint32_t indexDataAlign = dataAlign / indicesSize;

    const int64_t* dim = (attrs->GetAttrPointer<int64_t>(0));
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, dim);
    const char* reduce = attrs->GetAttrPointer<char>(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, reduce);
    isDeterministic = IsArch22DeterministicMode(tilingContext, reduce);
    const bool* includeSelfAttr = attrs->GetAttrPointer<bool>(2);
    includeSelf = includeSelfAttr == nullptr ? 1 : static_cast<uint64_t>(*includeSelfAttr);
    auto inputShape = tilingContext->GetInputShape(INPUT_0)->GetStorageShape();
    auto indicesShape = tilingContext->GetInputShape(INPUT_1)->GetStorageShape();
    auto updatesShape = tilingContext->GetInputShape(INPUT_2)->GetStorageShape();
    auto inputDimNum = inputShape.GetDimNum();
    if (strcmp(reduce, "none") == 0) {
        mode = NONE;
    } else if (strcmp(reduce, "add") == 0) {
        mode = ADD;
    } else if (strcmp(reduce, "min") == 0) {
        mode = MIN;
    } else if (strcmp(reduce, "max") == 0) {
        mode = MAX;
    } else if (strcmp(reduce, "mean") == 0) {
        mode = MEAN;
    } else if (strcmp(reduce, "mul") == 0) {
        mode = MUL;
    } else {
        OP_LOGE(tilingContext, "scatter_elements_v2 only support none, add, prod, min, max, mean.");
        return ge::GRAPH_FAILED;
    }

    if (inputDimNum != indicesShape.GetDimNum() ||
        inputDimNum != tilingContext->GetInputShape(INPUT_2)->GetStorageShape().GetDimNum()) {
        OP_LOGE(tilingContext, "the dimNum of input must equal the dimNum of indices.");
        return ge::GRAPH_FAILED;
    }

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
            OP_LOGE(tilingContext, "the dim of updates must be greater than or equal to the dim of indices.");
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
    if ((strcmp(reduce, "add") == 0 || strcmp(reduce, "mul") == 0 || strcmp(reduce, "min") == 0 ||
         strcmp(reduce, "max") == 0 || strcmp(reduce, "mean") == 0) &&
        (ge::DT_FLOAT16 == inputDtype || ge::DT_BF16 == inputDtype)) {
        inputSize += sizeof(float) / BUFFER_NUM;
        indicesSize += sizeof(float) / BUFFER_NUM;
    }
    if (strcmp(reduce, "add") == 0 || strcmp(reduce, "mul") == 0 || strcmp(reduce, "min") == 0 ||
        strcmp(reduce, "max") == 0 || strcmp(reduce, "mean") == 0) {
        inputSize += sizeof(int) / BUFFER_NUM;
    }

    if (isDeterministic && strcmp(reduce, "none") != 0) {
        OP_LOGD(
            tilingContext,
            "ScatterElementsV2 arch22 deterministic reduction enabled, force legacy kernel and single-core schedule.");
        coreNum = 1;
    } else if (isDeterministic) {
        // MaxUnpool2d uses unique indices with reduction=none. Keep the regular split so
        // large spatial rows can use all available cores without write conflicts.
        OP_LOGD(tilingContext, "ScatterElementsV2 arch22 deterministic none uses the regular multi-core schedule.");
    }

    uint32_t times = indicesCount / indicesOneTime;
    uint64_t totalSize = inputSize * (inputOneTime + updatesOneTime) + indicesSize * indicesOneTime;
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

    if (mode == NONE && modeFlag != SMALL_MODE && inputLoop >= STABLE_BUCKET_MIN_INPUT_LOOPS &&
        inputLoop <= STABLE_BUCKET_MAX_INPUT_LOOPS) {
        const uint64_t bucketMetaBytes = inputLoop * 2 * SIZE_OF_INT32;
        const uint64_t fixedBytes = inputAlign * inputSize + bucketMetaBytes;
        // updatesLocal, index storage (including the int64-to-int32 cast
        // buffer when needed), and one bucket position are resident for one
        // index element.
        const uint64_t bytesPerIndex = inputSize + indicesSize + SIZE_OF_INT32;
        if (fixedBytes < max_ub) {
            uint64_t bucketIndicesEach = (max_ub - fixedBytes) / bytesPerIndex;
            bucketIndicesEach = bucketIndicesEach > indicesOneTime ? indicesOneTime : bucketIndicesEach;
            bucketIndicesEach = bucketIndicesEach / indexDataAlign * indexDataAlign;
            if (bucketIndicesEach >= STABLE_BUCKET_MIN_INDICES) {
                indicesEach = bucketIndicesEach;
                indicesLoop = (indicesOneTime - 1) / indicesEach + 1;
                indicesLast = indicesOneTime - indicesEach * (indicesLoop - 1);
                indicesAlign = ((indicesEach - 1) / indexDataAlign + 1) * indexDataAlign;
                updatesAlign = ((indicesEach - 1) / inputDataAlign + 1) * inputDataAlign;
                useStableBucket = 1;
            }
        }
    }

    OP_LOGD(tilingContext, "Tiling inited.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterElementsV2Tiling::RunKernelTiling()
{
    OP_LOGD(tilingContext, "Tiling start.");

    // kernel 侧以 bktMode != 0 分流到分桶散射；此处显式置 0 走既有主路径，不依赖 tilingData 的默认零值
    tilingData.set_bktMode(0);
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
    tilingData.set_includeSelf(includeSelf);
    tilingData.set_mode(static_cast<uint64_t>(mode));
    // M carries the execution plan selected for the legacy kernel.
    tilingData.set_M(useStableBucket);
    tilingData.set_lastIndicesLoop(lastIndicesLoop);
    tilingData.set_lastIndicesEach(lastIndicesEach);
    tilingData.set_lastIndicesLast(lastIndicesLast);
    tilingData.set_oneTime(oneTime);
    tilingData.set_lastOneTime(lastOneTime);
    tilingData.SaveToBuffer(tilingContext->GetRawTilingData()->GetData(),
                            tilingContext->GetRawTilingData()->GetCapacity());
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
    OP_LOGD(tilingContext, "includeSelf: %lu.", includeSelf);
    OP_LOGD(tilingContext, "mode: %d.", mode);
    OP_LOGD(tilingContext, "useStableBucket: %lu.", useStableBucket);
    OP_LOGD(tilingContext, "lastIndicesLoop: %lu.", lastIndicesLoop);
    OP_LOGD(tilingContext, "lastIndicesEach: %lu.", lastIndicesEach);
    OP_LOGD(tilingContext, "lastIndicesLast: %lu.", lastIndicesLast);
    OP_LOGD(tilingContext, "oneTime: %lu.", oneTime);
    OP_LOGD(tilingContext, "lastOneTime: %lu.", lastOneTime);
    OP_LOGD(tilingContext, "tilingKey: %u.", tilingKey);
    OP_LOGD(tilingContext, "max_ub: %lu.", max_ub);
}

bool ScatterElementsV2Tiling::CacheOpSupport()
{
    if (tilingContext == nullptr) {
        OP_LOGD("ScatterElementsV2", "tilingContext is nullptr.");
        return false;
    }

    auto attrs = tilingContext->GetAttrs();
    auto inputDtype = tilingContext->GetInputDesc(INPUT_0)->GetDataType();
    const int64_t* dim = (attrs->GetAttrPointer<int64_t>(0));
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, dim);
    auto indicesShape = tilingContext->GetInputShape(INPUT_1)->GetStorageShape();
    auto inputDimNum = indicesShape.GetDimNum();
    realDim = (*dim < 0 ? *dim + inputDimNum : *dim);

    auto xShape = tilingContext->GetInputShape(INPUT_0)->GetStorageShape();
    const char* reduce = attrs->GetAttrPointer<char>(1);
    if (IsArch22DeterministicMode(tilingContext, reduce) && strcmp(reduce, "none") != 0) {
        OP_LOGD("ScatterElementsV2", "cache-op disabled for deterministic reduction mode on arch22.");
        return false;
    }
    if (!CheckCacheOpShapeLimit(xShape, reduce) || !CheckCacheOpDtype(inputDtype, reduce)) {
        return false;
    }

    auto updatesShape = tilingContext->GetInputShape(INPUT_2)->GetStorageShape();
    return CheckLastAxisCacheOp(inputDimNum, inputDtype, reduce) &&
           CheckCacheOpXDim1Limit(xShape, indicesShape, updatesShape, inputDimNum, inputDtype, reduce);
}

bool ScatterElementsV2Tiling::CheckCacheOpShapeLimit(const gert::Shape& xShape, const char* reduce) const
{
    if (xShape.GetDim(realDim) > CACHE_OP_X_LOCAL_LENGTH) {
        OP_LOGD("ScatterElementsV2", "new kernel only support x dim <= 40000.");
        return false;
    }
    bool needHitCount = reduce != nullptr && strcmp(reduce, "none") != 0;
    if (needHitCount && xShape.GetDim(realDim) > CACHE_OP_X_LOCAL_LENGTH_REDUCE) {
        OP_LOGD("ScatterElementsV2", "new kernel reduce mode only support x dim <= 20480.");
        return false;
    }
    return true;
}

bool ScatterElementsV2Tiling::CheckCacheOpDtype(ge::DataType inputDtype, const char* reduce) const
{
    if (inputDtype == ge::DT_BOOL || inputDtype == ge::DT_INT8 || inputDtype == ge::DT_UINT8) {
        if (reduce == nullptr) {
            return false;
        }
        if (strcmp(reduce, "none") != 0) {
            OP_LOGD("ScatterElementsV2", "when dtype is bool/int8/uint8, new kernel only support none mode.");
            return false;
        }
    }
    return true;
}

bool ScatterElementsV2Tiling::CheckLastAxisCacheOp(size_t inputDimNum, ge::DataType inputDtype,
                                                   const char* reduce) const
{
    auto updatesShape = tilingContext->GetInputShape(INPUT_2)->GetStorageShape();
    if (realDim == inputDimNum - 1 && updatesShape.GetDimNum() != 0 && inputDtype != ge::DT_BOOL) {
        if (reduce != nullptr && strcmp(reduce, "none") == 0) {
            if (inputDtype == ge::DT_INT32) {
                // The cache aggregation layout currently aliases the INT32
                // index row with its source-position workspace. Keep this
                // dtype on the established legacy last-axis tiling until a
                // dedicated deterministic cache layout is available.
                return false;
            }
            if (inputDtype == ge::DT_FLOAT &&
                tilingContext->GetInputShape(INPUT_0)->GetStorageShape().GetDim(inputDimNum - 1) <= 64) {
                // For tiny FP32 rows the cache kernel's fixed workspace and
                // synchronization prologue costs more than the legacy row
                // kernel. Keep this narrow shape family on the small-row path.
                return false;
            }
            // Each cache task owns complete rows. For overwrite semantics,
            // the scalar loop preserves update order inside a row, so a
            // last-axis non-scalar scatter is deterministic and safe here.
            return true;
        }
        bool allowLastAxisNonScalarFloat = (inputDtype == ge::DT_FLOAT) && (reduce != nullptr) &&
                                           (strcmp(reduce, "min") == 0 || strcmp(reduce, "mean") == 0);
        if (!allowLastAxisNonScalarFloat) {
            OP_LOGD("ScatterElementsV2",
                    "when realDim = -1 and updates not scalar, only bool or float min/mean will use new kernel.");
            return false;
        }
    }
    return true;
}

bool ScatterElementsV2Tiling::CheckCacheOpXDim1Limit(const gert::Shape& inputShape, const gert::Shape& indicesShape,
                                                     const gert::Shape& updatesShape, size_t inputDimNum,
                                                     ge::DataType inputDtype, const char* reduce)
{
    auto savedBatchSize = batchSize;
    auto savedRealDim = realDim;
    auto savedXDim0 = xDim0;
    auto savedXDim1 = xDim1;
    auto savedIndicesDim0 = indicesDim0;
    auto savedIndicesDim1 = indicesDim1;
    auto savedUpdatesDim0 = updatesDim0;
    auto savedUpdatesDim1 = updatesDim1;
    auto savedUpdatesIsScalar = updatesIsScalar;

    batchSize = 1;
    xDim0 = 1;
    xDim1 = 1;
    indicesDim0 = 1;
    indicesDim1 = 1;
    updatesDim0 = 1;
    updatesDim1 = 1;
    updatesIsScalar = 0;
    realDim = savedRealDim;

    auto updatesShapeCopy = updatesShape;
    ProcessUpdatesShape(updatesShapeCopy, indicesShape);
    SetDimsByAxisType(inputShape, indicesShape, updatesShapeCopy, inputDimNum);

    uint64_t maxXDim1 = GetCacheOpMaxXDim1(inputDtype, reduce);
    bool supported = xDim1 <= maxXDim1;
    if (!supported) {
        OP_LOGD(tilingContext, "cache-op disabled because xDim1(%lu) exceeds row UB capacity(%lu).", xDim1, maxXDim1);
    }

    batchSize = savedBatchSize;
    realDim = savedRealDim;
    xDim0 = savedXDim0;
    xDim1 = savedXDim1;
    indicesDim0 = savedIndicesDim0;
    indicesDim1 = savedIndicesDim1;
    updatesDim0 = savedUpdatesDim0;
    updatesDim1 = savedUpdatesDim1;
    updatesIsScalar = savedUpdatesIsScalar;
    return supported;
}

uint64_t ScatterElementsV2Tiling::GetCacheOpMaxXDim1(ge::DataType inputDtype, const char* reduce) const
{
    bool needHitCount = reduce != nullptr && strcmp(reduce, "none") != 0;
    if (!needHitCount) {
        return CACHE_OP_X_LOCAL_LENGTH - 1;
    }

    uint64_t ubTypeSize = ge::GetSizeByDataType(inputDtype);
    if (inputDtype == ge::DT_FLOAT16 || inputDtype == ge::DT_BF16) {
        ubTypeSize = SIZE_OF_FP32;
    }

    uint64_t tailBytes = static_cast<uint64_t>(CACHE_OP_INDICES_LOCAL_LENGTH) * SIZE_OF_INT64 +
                         static_cast<uint64_t>(CACHE_OP_INDICES_LOCAL_LENGTH) * ubTypeSize +
                         static_cast<uint64_t>(AGG_INDICES_NUM) * SIZE_OF_INT32 +
                         static_cast<uint64_t>(AGG_INDICES_NUM) * SIZE_OF_INT32;
    if (tailBytes >= CACHE_OP_ALL_UB_SIZE) {
        return 0;
    }
    return (CACHE_OP_ALL_UB_SIZE - tailBytes) / (ubTypeSize + SIZE_OF_INT32);
}

void ScatterElementsV2Tiling::SetTilingData(ScatterElementsV2TilingData& tiling)
{
    // 与 RunKernelTiling() 同理：cache-op 发射路径也显式置 0，确保 kernel 侧分流判据不读到脏值
    tiling.set_bktMode(0);
    tiling.set_batchSize(batchSize);
    tiling.set_realDim(realDim);
    tiling.set_coreNums(usedCoreNum);
    tiling.set_includeSelf(includeSelf);
    tiling.set_mode(static_cast<uint64_t>(mode));
    tiling.set_xDim0(xDim0);
    tiling.set_xDim1(xDim1);
    tiling.set_indicesDim0(indicesDim0);
    tiling.set_indicesDim1(indicesDim1);
    tiling.set_updatesDim0(updatesDim0);
    tiling.set_updatesDim1(updatesDim1);
    // M carries the execution plan selected for the cache kernel.
    tiling.set_M(executionPlan);
}

void ScatterElementsV2Tiling::LogTilingData() const
{
    OP_LOGD(tilingContext, "batchSize: %lu.", batchSize);
    OP_LOGD(tilingContext, "realDim: %lu.", realDim);
    OP_LOGD(tilingContext, "mode: %d.", mode);
    OP_LOGD(tilingContext, "coreNums: %d.", usedCoreNum);
    OP_LOGD(tilingContext, "updatesIsScalar: %d.", updatesIsScalar);
    OP_LOGD(tilingContext, "includeSelf: %lu.", includeSelf);
    OP_LOGD(tilingContext, "xDim0: %lu.", xDim0);
    OP_LOGD(tilingContext, "xDim1: %lu.", xDim1);
    OP_LOGD(tilingContext, "indicesDim0: %lu.", indicesDim0);
    OP_LOGD(tilingContext, "indicesDim1: %lu.", indicesDim1);
    OP_LOGD(tilingContext, "updatesDim0: %lu.", updatesDim0);
    OP_LOGD(tilingContext, "updatesDim1: %lu.", updatesDim1);
}

size_t ScatterElementsV2Tiling::CalculateWorkspaceSize()
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingContext->GetPlatformInfo());
    size_t libApiworkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    auto inputDtype = tilingContext->GetInputDesc(INPUT_0)->GetDataType();

    int32_t sizeOfVar = ge::GetSizeByDataType(inputDtype);
    if (inputDtype == ge::DT_FLOAT || inputDtype == ge::DT_INT32) {
        sizeOfVar = SIZE_OF_FP32;
    } else if (inputDtype == ge::DT_FLOAT16 || inputDtype == ge::DT_BF16) {
        sizeOfVar = SIZE_OF_FP16;
    } else if (inputDtype == ge::DT_UINT8 || inputDtype == ge::DT_INT8 || inputDtype == ge::DT_BOOL) {
        sizeOfVar = SIZE_OF_UINT8;
    }
    OP_LOGD(tilingContext, "sizeOfVar: %d.", sizeOfVar);
    libApiworkspaceSize += AGG_INDICES_NUM * 2 * sizeof(int32_t); // 2 is indices+updates

    if (batchSize == 1 && realDim == 1) { // 首轴需要转置
        auto targetParts = indicesDim1 / usedCoreNum;
        targetParts = targetParts > 0 ? targetParts : 1;
        targetParts = targetParts > TILE_SIZE ? TILE_SIZE : targetParts;
        auto partSize = indicesDim1 / targetParts;
        auto extraSize = this->indicesDim1 % targetParts; // 额外的大小
        auto maxSize = partSize > extraSize ? partSize : extraSize;

        libApiworkspaceSize += maxSize * xDim0 * sizeOfVar;
        libApiworkspaceSize += maxSize * indicesDim0 * sizeof(int32_t);
        libApiworkspaceSize += this->updatesIsScalar ? 0 : maxSize * updatesDim0 * sizeOfVar;
        // 用于转置x indices updates时，及用于x转置回来时，转连续
        libApiworkspaceSize += ALIGN_SIZE * VAR_LIMIT * sizeof(uint32_t) * SIZE_OF_FP32;
        // 用于整块处理时，直接使用gather做转置，无需pad/unpad，也无需TransDataTo5HD。（xForward indicesForward
        // updatesForward xBackward）
        libApiworkspaceSize += VAR_LIMIT * VAR_LIMIT * sizeof(uint32_t) * SIZE_OF_FP32;
    }
    if (batchSize > 1 && realDim == 1) { // 中轴需要转置
        int32_t targetParts = batchSize <= TILE_SIZE ? batchSize : TILE_SIZE;
        uint32_t partSize = batchSize / targetParts;
        uint32_t extraSize = batchSize % targetParts; // 额外的大小
        uint32_t maxSize = partSize > extraSize ? partSize : extraSize;

        libApiworkspaceSize += maxSize * xDim0 * xDim1 * sizeOfVar;
        libApiworkspaceSize += maxSize * indicesDim0 * indicesDim1 * sizeof(int32_t);
        libApiworkspaceSize += this->updatesIsScalar ? 0 : maxSize * updatesDim0 * updatesDim1 * sizeOfVar;

        // 正向，x indices, updates + 反向x
        libApiworkspaceSize += VAR_LIMIT * VAR_LIMIT * sizeof(uint32_t) * SIZE_OF_FP32;
    }
    if (executionPlan != 0) {
        const uint64_t sourceChunks = 4;
        const uint64_t destinationIntervals = executionPlan == WORKSPACE_GATHER_FOUR_BY_ONE ? 1 : 2;
        const uint64_t intervalWidth = (xDim1 + destinationIntervals - 1) / destinationIntervals;
        const uint64_t intervalStride = (intervalWidth + 7) / 8 * 8;
        libApiworkspaceSize += xDim0 * destinationIntervals * sourceChunks * intervalStride * sizeof(int32_t);
    }
    return libApiworkspaceSize;
}

ge::graphStatus ScatterElementsV2Tiling::SetCacheOpTiling()
{
    ScatterElementsV2TilingData tiling;
    SetTilingData(tiling);
    LogTilingData();

    size_t calculateResult = CalculateWorkspaceSize();
    size_t* currentWorkSpace = tilingContext->GetWorkspaceSizes(1);
    currentWorkSpace[0] = calculateResult;

    tilingContext->SetBlockDim(usedCoreNum);
    auto inputDtype = tilingContext->GetInputDesc(INPUT_0)->GetDataType();
    auto indicesDtype = tilingContext->GetInputDesc(INPUT_1)->GetDataType();
    uint32_t cacheTilingKey = 0;
    if (ge::DT_FLOAT == inputDtype) {
        cacheTilingKey += INPUT_TYPE * DT_FLOAT32_TYPE;
    } else if (ge::DT_FLOAT16 == inputDtype) {
        cacheTilingKey += INPUT_TYPE * DT_FLOAT16_TYPE;
    } else if (ge::DT_INT32 == inputDtype) {
        cacheTilingKey += INPUT_TYPE * DT_INT32_TYPE;
    } else if (ge::DT_UINT8 == inputDtype || ge::DT_BOOL == inputDtype) {
        cacheTilingKey += INPUT_TYPE * DT_UINT8_TYPE;
    } else if (ge::DT_INT8 == inputDtype) {
        cacheTilingKey += INPUT_TYPE * DT_INT8_TYPE;
    } else if (ge::DT_BF16 == inputDtype) {
        cacheTilingKey += INPUT_TYPE * DT_BF16_TYPE;
    } else if (ge::DT_INT16 == inputDtype) {
        cacheTilingKey += INPUT_TYPE * DT_INT16_TYPE;
    } else if (ge::DT_INT64 == inputDtype) {
        cacheTilingKey += INPUT_TYPE * DT_INT64_TYPE;
    } else if (ge::DT_DOUBLE == inputDtype) {
        cacheTilingKey += INPUT_TYPE * DT_DOUBLE_TYPE;
    }
    if (ge::DT_INT32 == indicesDtype) {
        cacheTilingKey += INDIC_TYPE * DT_INT32_INDEX_TYPE;
    } else if (ge::DT_INT64 == indicesDtype) {
        cacheTilingKey += INDIC_TYPE * DT_INT64_INDEX_TYPE;
    }
    tilingContext->SetTilingKey(cacheTilingKey);
    OP_LOGD(tilingContext, "scatterElementsV2 tilingKey: %u.", cacheTilingKey);
    tilingContext->SetScheduleMode(1); // kernel内涉及SyncAll()
    tiling.SaveToBuffer(tilingContext->GetRawTilingData()->GetData(), tilingContext->GetRawTilingData()->GetCapacity());
    tilingContext->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

void ScatterElementsV2Tiling::ParseAttrs()
{
    auto attrs = tilingContext->GetAttrs();
    const char* reduce = attrs->GetAttrPointer<char>(1);
    const bool* includeSelfAttr = attrs->GetAttrPointer<bool>(2);
    includeSelf = includeSelfAttr == nullptr ? 1 : static_cast<uint64_t>(*includeSelfAttr);
    if (strcmp(reduce, "none") == 0) {
        mode = NONE;
    } else if (strcmp(reduce, "add") == 0) {
        mode = ADD;
    } else if (strcmp(reduce, "mul") == 0) {
        mode = MUL;
    } else if (strcmp(reduce, "min") == 0) {
        mode = MIN;
    } else if (strcmp(reduce, "max") == 0) {
        mode = MAX;
    } else if (strcmp(reduce, "mean") == 0) {
        mode = MEAN;
    } else {
        mode = ADD;
    }
}

void ScatterElementsV2Tiling::ProcessUpdatesShape(gert::Shape& updatesShape, const gert::Shape& indicesShape)
{
    if ((updatesShape.GetDimNum() == 1 && updatesShape.GetDim(0) == 1) || updatesShape.GetDimNum() == 0) {
        updatesIsScalar = 1;
        updatesShape = indicesShape;
    } else {
        updatesIsScalar = 0;
    }
}

void ScatterElementsV2Tiling::SetDimsForFirstAxis(const gert::Shape& inputShape, const gert::Shape& indicesShape,
                                                  const gert::Shape& updatesShape, size_t inputDimNum)
{
    batchSize = 1;
    realDim = 1;
    xDim0 = inputShape.GetDim(0);
    indicesDim0 = indicesShape.GetDim(0);
    updatesDim0 = updatesShape.GetDim(0);
    for (size_t i = 1; i < inputDimNum; i++) {
        xDim1 *= inputShape.GetDim(i);
        indicesDim1 *= indicesShape.GetDim(i);
        updatesDim1 *= updatesShape.GetDim(i);
    }
}

void ScatterElementsV2Tiling::SetDimsForMiddleAxis(const gert::Shape& inputShape, const gert::Shape& indicesShape,
                                                   const gert::Shape& updatesShape, size_t inputDimNum)
{
    for (size_t i = 0; i < realDim; i++) {
        batchSize *= indicesShape.GetDim(i);
    }
    xDim0 = inputShape.GetDim(realDim);
    indicesDim0 = indicesShape.GetDim(realDim);
    updatesDim0 = updatesShape.GetDim(realDim);
    for (size_t i = realDim + 1; i < inputDimNum; i++) {
        xDim1 *= inputShape.GetDim(i);
        indicesDim1 *= indicesShape.GetDim(i);
        updatesDim1 *= updatesShape.GetDim(i);
    }
    realDim = 1;
}

void ScatterElementsV2Tiling::SetDimsForLastAxis(const gert::Shape& inputShape, const gert::Shape& indicesShape,
                                                 const gert::Shape& updatesShape, size_t inputDimNum)
{
    batchSize = 1;
    realDim = 2;
    xDim1 = inputShape.GetDim(inputDimNum - 1);
    indicesDim1 = indicesShape.GetDim(inputDimNum - 1);
    updatesDim1 = updatesShape.GetDim(inputDimNum - 1);
    for (size_t i = 0; i < inputDimNum - 1; i++) {
        xDim0 *= inputShape.GetDim(i);
        indicesDim0 *= indicesShape.GetDim(i);
        updatesDim0 *= updatesShape.GetDim(i);
    }
}

void ScatterElementsV2Tiling::SetDimsByAxisType(const gert::Shape& inputShape, const gert::Shape& indicesShape,
                                                const gert::Shape& updatesShape, size_t inputDimNum)
{
    if (realDim == 0) {
        SetDimsForFirstAxis(inputShape, indicesShape, updatesShape, inputDimNum);
    } else if (realDim > 0 && realDim < inputDimNum - 1) {
        SetDimsForMiddleAxis(inputShape, indicesShape, updatesShape, inputDimNum);
    } else {
        SetDimsForLastAxis(inputShape, indicesShape, updatesShape, inputDimNum);
    }
}

ge::graphStatus ScatterElementsV2Tiling::RunCacheOpTiling()
{
    ParseAttrs();

    auto inputShape = tilingContext->GetInputShape(INPUT_0)->GetStorageShape();
    auto indicesShape = tilingContext->GetInputShape(INPUT_1)->GetStorageShape();
    auto updatesShape = tilingContext->GetInputShape(INPUT_2)->GetStorageShape();

    ProcessUpdatesShape(updatesShape, indicesShape);

    auto inputDimNum = inputShape.GetDimNum();
    auto attrs = tilingContext->GetAttrs();
    const int64_t* dim = (attrs->GetAttrPointer<int64_t>(0));
    realDim = (*dim < 0 ? *dim + inputDimNum : *dim);

    SetDimsByAxisType(inputShape, indicesShape, updatesShape, inputDimNum);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingContext->GetPlatformInfo());
    // Cache tasks own whole output rows. Launching more AIV cores than rows
    // only adds prologue/epilogue cost and cannot create useful work.
    usedCoreNum = std::min<uint64_t>(ascendcPlatform.GetCoreNumAiv(), xDim0);

    // A zero-base overwrite does not depend on the input row. For sufficiently
    // wide output rows, partitioning by destination interval is deterministic:
    // every bucket scans the source row in its original order, and buckets
    // write disjoint ranges. This exposes row-internal parallelism without
    // introducing a cross-core duplicate-index race.
    constexpr uint64_t kMinBucketElements = 256;
    constexpr uint64_t kMaxBucketsPerRow = 4;
    constexpr uint64_t kWorkspaceSourceChunks = 4;
    constexpr uint64_t kWorkspaceDestinationIntervals = 2;
    constexpr uint64_t kWorkspaceMinOutputWidth = 512;
    constexpr uint64_t kWorkspaceMaxOutputWidth = 8192;
    constexpr uint64_t kWorkspaceMinSourceWidth = 64;
    const auto inputDtype = tilingContext->GetInputDesc(INPUT_0)->GetDataType();
    const auto indicesDtype = tilingContext->GetInputDesc(INPUT_1)->GetDataType();
    const bool nativeOutputType = inputDtype != ge::DT_BOOL && inputDtype != ge::DT_INT8 && inputDtype != ge::DT_UINT8;
    const bool workspaceGatherType = inputDtype == ge::DT_FLOAT16 || inputDtype == ge::DT_BF16;
    const bool workspacePlanCandidate = mode == NONE && includeSelf == 0 && updatesIsScalar == 0 &&
                                        indicesDtype == ge::DT_INT32 && workspaceGatherType &&
                                        indicesDim1 == updatesDim1 && indicesDim1 >= kWorkspaceMinSourceWidth &&
                                        indicesDim1 <= CACHE_OP_INDICES_LOCAL_LENGTH &&
                                        xDim1 >= kWorkspaceMinOutputWidth && xDim1 <= kWorkspaceMaxOutputWidth &&
                                        indicesDim1 * 2 <= xDim1 && xDim0 != 0 &&
                                        xDim0 * kWorkspaceSourceChunks * kWorkspaceDestinationIntervals <=
                                            ascendcPlatform.GetCoreNumAiv();
    if (workspacePlanCandidate) {
        // For short source rows, one owner per row avoids replaying every
        // source chunk for each destination interval. The owner still merges
        // four position tables with vector Max, preserving ordered overwrite
        // semantics. Wider source rows retain two destination owners so their
        // output gather work remains parallel.
        const bool useSingleOwner = indicesDim1 <= WORKSPACE_SINGLE_OWNER_MAX_SOURCE_WIDTH &&
                                    xDim1 >= WORKSPACE_SINGLE_OWNER_MIN_OUTPUT_WIDTH &&
                                    xDim0 * kWorkspaceSourceChunks <= ascendcPlatform.GetCoreNumAiv();
        executionPlan = useSingleOwner ? WORKSPACE_GATHER_FOUR_BY_ONE : WORKSPACE_GATHER_FOUR_BY_TWO;
        usedCoreNum = useSingleOwner ? xDim0 * kWorkspaceSourceChunks :
                                       xDim0 * kWorkspaceSourceChunks * kWorkspaceDestinationIntervals;
    }
    const bool outputBucketCandidate = mode == NONE && includeSelf == 0 && updatesIsScalar == 0 &&
                                       indicesDim1 == updatesDim1 && indicesDim1 <= CACHE_OP_INDICES_LOCAL_LENGTH &&
                                       xDim1 >= 2 * kMinBucketElements && nativeOutputType &&
                                       indicesDim1 * kMaxBucketsPerRow <= xDim1;
    if (executionPlan == 0 && outputBucketCandidate && xDim0 != 0) {
        uint64_t bucketCount = (xDim1 + kMinBucketElements - 1) / kMinBucketElements;
        bucketCount = std::min<uint64_t>(bucketCount, kMaxBucketsPerRow);
        bucketCount = std::min<uint64_t>(bucketCount, ascendcPlatform.GetCoreNumAiv() / xDim0);
        if (bucketCount > 1) {
            usedCoreNum = xDim0 * bucketCount;
        }
    }
    OP_LOGD(tilingContext, "ascendcPlatform CoreNum is: %d.", usedCoreNum);
    return SetCacheOpTiling();
}

// 重新归约 xDim*/indicesDim*/updatesDim*/batchSize/updatesIsScalar。这些成员原本只由 cache-op 路径
// （RunCacheOpTiling）填充；Init() 只计算旧 kernel 需要的切分字段，从不给它们赋值，故非 cache-op
// 路径下它们恒为默认值 1。分桶分支的形状判据必须先调用本函数补齐，否则判据恒拿默认值比较、
// 该分支永远不会被选中。SetDimsFor*Axis 内部对这些成员是累乘（*=），故此处先整体复位再重算，
// 使本函数可重复调用而结果不变（幂等）。
bool ScatterElementsV2Tiling::ResolveScatterDims()
{
    if (tilingContext == nullptr) {
        return false;
    }
    auto inputShapePtr = tilingContext->GetInputShape(INPUT_0);
    auto indicesShapePtr = tilingContext->GetInputShape(INPUT_1);
    auto updatesShapePtr = tilingContext->GetInputShape(INPUT_2);
    auto attrs = tilingContext->GetAttrs();
    if (inputShapePtr == nullptr || indicesShapePtr == nullptr || updatesShapePtr == nullptr || attrs == nullptr) {
        return false;
    }
    const int64_t* dim = attrs->GetAttrPointer<int64_t>(0);
    if (dim == nullptr) {
        return false;
    }

    auto inputShape = inputShapePtr->GetStorageShape();
    auto indicesShape = indicesShapePtr->GetStorageShape();
    auto updatesShape = updatesShapePtr->GetStorageShape();
    auto inputDimNum = inputShape.GetDimNum();
    if (inputDimNum == 0 || indicesShape.GetDimNum() != inputDimNum) {
        return false;
    }
    // SetDimsFor*Axis 按 inputDimNum 遍历 updates，故 updates 须与 input 同维，
    // 或为 ProcessUpdatesShape 可展开成 indices 形状的 scalar 形态
    auto updatesDimNum = updatesShape.GetDimNum();
    bool updatesScalarLike = (updatesDimNum == 0) || (updatesDimNum == 1 && updatesShape.GetDim(0) == 1);
    if (!updatesScalarLike && updatesDimNum != inputDimNum) {
        return false;
    }
    int64_t axis = (*dim < 0) ? (*dim + static_cast<int64_t>(inputDimNum)) : *dim;
    if (axis < 0 || axis >= static_cast<int64_t>(inputDimNum)) {
        return false;
    }

    batchSize = 1;
    xDim0 = 1;
    xDim1 = 1;
    indicesDim0 = 1;
    indicesDim1 = 1;
    updatesDim0 = 1;
    updatesDim1 = 1;
    updatesIsScalar = 0;
    realDim = static_cast<uint64_t>(axis);

    // 与 RunCacheOpTiling() 用同一套归约模型：scalar updates 先展开成 indices 形状，
    // 再按首轴 / 中轴 / 末轴折叠成二维 (xDim0, xDim1)
    ProcessUpdatesShape(updatesShape, indicesShape);
    SetDimsByAxisType(inputShape, indicesShape, updatesShape, inputDimNum);
    return true;
}

// 分桶散射分支的启用条件。严格收窄：仅 BFLOAT16 末轴 reduction=none 且 var 末轴远大于更新数的
// 稀疏场景才启用；其余 dtype / 场景的选路与既有实现完全一致，不受影响。
// 与 legacy kernel 的 ProcessNoneStableBucket（判据 M != 0，用于行内分块搬运）是两套独立机制：
// 此处的分桶是把稀疏更新点按输出 tile 归桶，把随机散射降为顺序突发写。
// 注意：本函数内会调用 ResolveScatterDims() 补齐维度成员，故不能声明为 const。
bool ScatterElementsV2Tiling::BucketScatterSupport()
{
    if (tilingContext == nullptr) {
        return false;
    }
    // 仅覆盖语义（none）适用“末次写赢”；带 reduction 的累积语义不适用
    if (mode != NONE) {
        return false;
    }
    if (includeSelf != 1) {
        return false;
    }
    // 先做 dtype 判定再补齐维度：ResolveScatterDims() 会改写 realDim/xDim* 等成员，
    // 把它挡在 BFLOAT16 之后可使该副作用面收窄到本分支真正可能命中的场景
    auto varDesc = tilingContext->GetInputDesc(INPUT_0);
    auto idxDesc = tilingContext->GetInputDesc(INPUT_1);
    if (varDesc == nullptr || idxDesc == nullptr) {
        return false;
    }
    // 仅本次新增支持的 BFLOAT16 启用，既有 dtype 行为保持不变
    if (varDesc->GetDataType() != ge::DT_BF16) {
        return false;
    }
    auto idxType = idxDesc->GetDataType();
    if (idxType != ge::DT_INT32 && idxType != ge::DT_INT64) {
        return false;
    }
    // ★ Init() 不给 xDim*/indicesDim*/updatesIsScalar 赋值，必须先补齐，
    //   否则下面的形状判据恒拿默认值 1 比较，本分支永远不会被选中。
    if (!ResolveScatterDims()) {
        return false;
    }
    if (updatesIsScalar != 0) {
        return false;
    }
    // 末轴无需再判：Init() 已强制 realDim == inputDimNum - 1，否则不会走到这里
    // 形状约束：var 末轴足够大，且更新数远小于它（稀疏），否则主路径已足够高效
    if (xDim1 < BUCKET_MIN_VAR_N || indicesDim1 == 0 || xDim0 == 0) {
        return false;
    }
    if (indicesDim1 * BUCKET_SPARSE_RATIO > xDim1) {
        return false;
    }
    // kernel 侧以同一个 bktIndicesN 索引 indices 与 updates，故两者末轴长度必须一致；
    // 行数也必须与 var 对齐，否则按 r*k 计算的行基址会越界
    if (updatesDim1 != indicesDim1 || indicesDim0 != xDim0 || updatesDim0 != xDim0) {
        return false;
    }
    return true;
}

ge::graphStatus ScatterElementsV2Tiling::RunBucketScatterTiling()
{
    auto platformInfo = tilingContext->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint64_t coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(ubSize == 0 || coreNum == 0, OP_LOGE(tilingContext, "invalid platform info for bucket scatter."),
                return ge::GRAPH_FAILED);

    // ResolveScatterDims() 幂等（内部先复位再重算），此处再调一次以解除对
    // “必须紧跟在 BucketScatterSupport() 之后被调用” 这一隐式顺序依赖
    OP_CHECK_IF(!ResolveScatterDims(), OP_LOGE(tilingContext, "failed to resolve dims for bucket scatter."),
                return ge::GRAPH_FAILED);
    uint64_t rows = xDim0;
    uint64_t m = xDim1;
    uint64_t k = indicesDim1;
    // var 为 BFLOAT16（2 字节），indices 缓冲按 int32 计
    const uint64_t vBytes = sizeof(uint16_t);
    const uint64_t idxBytes = sizeof(int32_t);

    // tileLen：在 UB 预算内取最大的 2 的幂
    // UB 占用 = tile(tileLen*vBytes) + idxBuf(CHUNK*4) + valBuf(CHUNK*vBytes) + margin
    uint64_t fixedBytes = BUCKET_UPD_CHUNK_HOST * idxBytes + BUCKET_UPD_CHUNK_HOST * vBytes + BUCKET_UB_MARGIN;
    uint64_t tileBytesMax = (ubSize > fixedBytes) ? (ubSize - fixedBytes) : vBytes;
    uint64_t tileLenMax = tileBytesMax / vBytes;
    if (tileLenMax < 1) {
        tileLenMax = 1;
    }
    uint64_t tileLen = 1;
    uint64_t shift = 0;
    while ((tileLen * BUCKET_POW2_BASE) <= tileLenMax) {
        tileLen *= BUCKET_POW2_BASE;
        shift++;
    }
    uint64_t numTiles = (m + tileLen - 1) / tileLen;
    OP_CHECK_IF(numTiles == 0, OP_LOGE(tilingContext, "bucket scatter numTiles is 0."), return ge::GRAPH_FAILED);

    // 每桶 FIFO 深度：numTiles*D*(4+2) <= 预算，取 16 的倍数并夹在 [16,64]
    uint64_t fifoDepth = BUCKET_FIFO_BUDGET / (numTiles * (idxBytes + vBytes));
    fifoDepth = (fifoDepth / BUCKET_ALIGN_HOST) * BUCKET_ALIGN_HOST;
    if (fifoDepth < BUCKET_ALIGN_HOST) {
        fifoDepth = BUCKET_ALIGN_HOST;
    }
    if (fifoDepth > BUCKET_FIFO_MAX) {
        fifoDepth = BUCKET_FIFO_MAX;
    }
    // 每核 GM 桶区容量（条目数）。kernel 侧每桶预留 padded = ceil(cnt,16)*16 + 16(guard)，
    // 由 ceil(cnt,16)*16 <= cnt + 15 得 padded <= cnt + 31，故 Σ_b padded <= k + 31*numTiles。
    // 取 32*numTiles 作对齐友好的上界：稀疏场景（k < 16*numTiles，正是本分支的启用前提）下
    // 旧式 k + 16*numTiles 会小于真实占用，导致桶 offset 越过本核桶区、踩写相邻核。
    uint64_t bktStride = k + BUCKET_PAD_PER_TILE_HOST * numTiles;

    // 行分核
    uint64_t usedCore = rows < coreNum ? rows : coreNum;
    if (usedCore < 1) {
        usedCore = 1;
    }
    uint64_t rowsPerCore = rows / usedCore;
    uint64_t frontCore = rows % usedCore;

    tilingData.set_bktMode(1);
    tilingData.set_bktRows(rows);
    tilingData.set_bktVarN(m);
    tilingData.set_bktIndicesN(k);
    tilingData.set_bktTileLen(tileLen);
    tilingData.set_bktNumTiles(numTiles);
    tilingData.set_bktShift(shift);
    tilingData.set_bktFifoDepth(fifoDepth);
    tilingData.set_bktStride(bktStride);
    tilingData.set_bktRowsPerCore(rowsPerCore);
    tilingData.set_bktFrontCore(frontCore);
    tilingData.set_usedCoreNum(usedCore);
    tilingData.set_includeSelf(includeSelf);
    tilingData.set_mode(static_cast<uint64_t>(mode));

    tilingData.SaveToBuffer(tilingContext->GetRawTilingData()->GetData(),
                            tilingContext->GetRawTilingData()->GetCapacity());
    tilingContext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    tilingContext->SetTilingKey(tilingKey);
    tilingContext->SetBlockDim(usedCore);

    // workspace：系统预留 + 每核 GM 桶区（int32 索引 + var dtype 值）。
    // workspaceSize 成员已在 TilingPrepare 中被赋为 GetLibApiWorkSpaceSize()，即系统预留部分；
    // kernel 侧用 GetUserWorkspace() 跳过该段后即为下面的桶区，故此处不可再叠加一次 lib api 大小。
    size_t* workspaces = tilingContext->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, workspaces);
    uint64_t bktBytes = usedCore * bktStride * (idxBytes + vBytes);
    workspaces[0] = static_cast<size_t>(workspaceSize + bktBytes);

    OP_LOGD(tilingContext,
            "bucket scatter tiling: rows=%lu, varN=%lu, indicesN=%lu, tileLen=%lu, numTiles=%lu, fifoDepth=%lu, "
            "bktStride=%lu, usedCore=%lu.",
            rows, m, k, tileLen, numTiles, fifoDepth, bktStride, usedCore);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingScatterElementsV2(gert::TilingContext* context)
{
    auto compile_info = reinterpret_cast<const ScatterElementsV2CompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);
    if (compile_info->is_regbase) {
        return Ops::NN::Optiling::TilingRegistry::GetInstance().DoTilingImpl(context);
    }

    ScatterElementsV2Tiling tilingObject(context);
    // The experimental entry implements the legacy row/tile kernel only.
    // Cache tiling populates a different set of fields and must not be sent
    // to that entry: its legacy loop bounds would be zero.
    if (tilingObject.Init() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // BFLOAT16 稀疏大 var 的末轴覆盖场景走分桶散射分支；其余场景与既有实现完全一致
    if (tilingObject.BucketScatterSupport()) {
        return tilingObject.RunBucketScatterTiling();
    }
    return tilingObject.RunKernelTiling();
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
    OP_CHECK_IF((compileInfo->ubSizePlatForm <= 0), OP_LOGE(context, "Failed to get ub size."),
                return ge::GRAPH_FAILED);
    OP_LOGD(context, "ub_size_platform is %lu.", compileInfo->ubSizePlatForm);
    uint64_t totalUbSize = 0;
    platformInfo->GetLocalMemSize(fe::LocalMemType::UB, totalUbSize);
    compileInfo->is_regbase = IsRegbaseSocVersion4Scatter(context);
    OP_LOGD(context, "total_ub_size is %lu.", totalUbSize);
    OP_LOGD(context, "TilingPrepareForScatterElementsV2 end.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ScatterElementsV2)
    .Tiling(TilingScatterElementsV2)
    .TilingParse<ScatterElementsV2CompileInfo>(TilingPrepareForScatterElementsV2);
} // namespace optiling
