/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file rotate_quant_tiling.cpp
 * \brief
 */
#include "rotate_quant_tiling.h"
#include "error_util.h"
#include "log/log.h"
#include "matmul/common/op_host/op_tiling/debug_tiling.h"
#include "platform/platform_infos_def.h"
#include "register/op_def_registry.h"
#include "op_host/tiling_key.h"
#include "op_host/tiling_templates_registry.h"
#include "matmul/common/op_host/math_util.h"

using namespace optiling;

namespace {
constexpr uint32_t X_INDEX = 0;
constexpr uint32_t ROT_INDEX = 1;
constexpr uint32_t Y_INDEX = 0;
constexpr uint32_t SCALE_INDEX = 1;

constexpr uint32_t BASE_M = 128;
constexpr uint32_t BASE_N = 256;
constexpr uint32_t BASE_K = 64;
constexpr uint32_t VEC_NUM_PER_CUBE = 2;
constexpr int32_t N_MAX = 16000;
constexpr int32_t N_MIN = 128;
constexpr int32_t K_MAX = 1024;
constexpr int32_t K_MIN = 16;

constexpr uint32_t RESERVED_LENGTH = 1024;
constexpr uint32_t NUM_TWO_FIVE_SIX = 256;
constexpr uint32_t ONE_REPEAT_ELE = 64;
constexpr uint32_t MEMORY_ALLOC_COEFFICIENT = 12;
constexpr uint32_t MULTI_ROW_NO_SMOOTH_SPLIT = 11;
constexpr uint32_t MAX_UB_REPEAT_COUNT = 255;

constexpr uint32_t STEP_LOOP_SMALL_K_THRESHOLD = 384;
constexpr uint32_t STEP_LOOP_SMALL_GROUP_SIZE = 15;
constexpr uint32_t STEP_LOOP_LARGE_GROUP_SIZE = 60;
constexpr uint32_t STEP_LOOP_CUBE_CORE_THRESHOLD = 54;
constexpr uint32_t STEP_LOOP_UB_COEFF = 2;
constexpr uint32_t STEP_LOOP_MIN_ITERS_PER_CORE = 15;

constexpr int64_t SYS_WORKSPACE_SIZE = 16777216;
} // namespace

namespace optiling {
ge::graphStatus RotateQuantTiling::GetPlatformInfo()
{
    return ge::GRAPH_SUCCESS;
}

void RotateQuantTiling::InitCompileInfo()
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        OP_LOGE(context_->GetNodeName(), "platformInfoPtr is null");
        return;
    }

    const auto &ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo_.ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfo_.l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, compileInfo_.l2Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfo_.l0ASize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfo_.l0BSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfo_.l0CSize);
    compileInfo_.aicNum = ascendcPlatform.GetCoreNumAic();
    compileInfo_.aivNum = ascendcPlatform.GetCoreNumAiv();

    if (compileInfo_.aicNum <= 0) {
        OP_LOGE(context_->GetNodeName(), "aicNum <= 0");
        return;
    }
    OP_LOGD(context_->GetNodeName(), "RotateQuant InitCompileInfo Success");
}

ge::graphStatus RotateQuantTiling::GetShapeAttrsInfo()
{
    tilingDataSize_ = sizeof(RotateQuantTilingData);
    if (inputParams_.initFlag) {
        OP_LOGD(inputParams_.opName, "No need to get shape and attrs from tiling context again");
        return ge::GRAPH_SUCCESS;
    }

    inputParams_.opName = context_->GetNodeName();
    OP_LOGI(inputParams_.opName, "TilingContext: %s", Ops::NN::DebugTilingContext(context_).c_str());
    OP_TILING_CHECK(
        CheckContext() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid context."),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        AnalyzeAttrs() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid attrs."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        AnalyzeDtype() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid dtypes."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        AnalyzeShapes() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid shapes."),
        return ge::GRAPH_FAILED);

    OP_LOGD(
        inputParams_.opName, "Input params: MNK[%ld, %ld, %ld], numBlocks[%ld].", inputParams_.M,
        inputParams_.N, inputParams_.K, inputParams_.numBlocks);

    inputParams_.initFlag = true;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RotateQuantTiling::CheckContext()
{
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(X_INDEX));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(X_INDEX));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(ROT_INDEX));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(ROT_INDEX));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputShape(Y_INDEX));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputDesc(Y_INDEX));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputShape(SCALE_INDEX));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputDesc(SCALE_INDEX));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData()->GetData());
    OP_TILING_CHECK(
        context_->GetRawTilingData()->GetCapacity() < tilingDataSize_,
        CUBE_INNER_ERR_REPORT(
            inputParams_.opName, "context tiling data capacity %zu < actual tiling data size %zu.",
            context_->GetRawTilingData()->GetCapacity(), tilingDataSize_),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RotateQuantTiling::AnalyzeAttrs()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RotateQuantTiling::AnalyzeDtype()
{
    inputParams_.xDtype = context_->GetInputDesc(X_INDEX)->GetDataType();
    auto rotDtype = context_->GetInputDesc(ROT_INDEX)->GetDataType();
    inputParams_.yDtype = context_->GetOutputDesc(Y_INDEX)->GetDataType();
    auto scaleDtype = context_->GetOutputDesc(SCALE_INDEX)->GetDataType();

    OP_TILING_CHECK(
        inputParams_.xDtype != ge::DT_FLOAT16 && inputParams_.xDtype != ge::DT_BF16,
        OP_LOGE(inputParams_.opName, "x dtype should be fp16 or bf16."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        rotDtype != ge::DT_FLOAT16 && rotDtype != ge::DT_BF16,
        OP_LOGE(inputParams_.opName, "rotation dtype should be fp16 or bf16."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        inputParams_.xDtype != rotDtype, OP_LOGE(inputParams_.opName, "x and rotation dtype should be same."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        inputParams_.yDtype != ge::DT_INT8 && inputParams_.yDtype != ge::DT_INT4,
        OP_LOGE(inputParams_.opName, "y dtype should be int8 or int4."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        scaleDtype != ge::DT_FLOAT, OP_LOGE(inputParams_.opName, "scale dtype should be float."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RotateQuantTiling::AnalyzeShapes()
{
    const auto &xShape = context_->GetInputShape(X_INDEX)->GetStorageShape();
    const auto &rotShape = context_->GetInputShape(ROT_INDEX)->GetStorageShape();
    const auto &yShape = context_->GetOutputShape(Y_INDEX)->GetStorageShape();
    const auto &scaleShape = context_->GetOutputShape(SCALE_INDEX)->GetStorageShape();

    inputParams_.M = xShape.GetDim(0);
    inputParams_.N = xShape.GetDim(1);
    inputParams_.K = rotShape.GetDim(0);
    int64_t rotK = rotShape.GetDim(1);

    OP_TILING_CHECK(
        inputParams_.N % inputParams_.K != 0,
        OP_LOGE(inputParams_.opName, "N(%ld) must be divisible by K(%ld).", inputParams_.N, inputParams_.K),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        inputParams_.K != rotK,
        OP_LOGE(inputParams_.opName, "RotK(%ld) must be the same as K(%ld).", rotK, inputParams_.K),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        inputParams_.N > N_MAX || inputParams_.N < N_MIN,
        OP_LOGE(inputParams_.opName, "N(%ld) must be between 128 and 16000.", inputParams_.N),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        inputParams_.K > K_MAX || inputParams_.K < K_MIN,
        OP_LOGE(inputParams_.opName, "K(%ld) must be between 16 and 1024.", inputParams_.K),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        inputParams_.M != yShape.GetDim(0),
        OP_LOGE(inputParams_.opName, "x dim0(%ld) must be the same as y dim0(%ld).", inputParams_.M, yShape.GetDim(0)),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        inputParams_.N != yShape.GetDim(1),
        OP_LOGE(inputParams_.opName, "x dim1(%ld) must be the same as y dim1(%ld).", inputParams_.N, yShape.GetDim(1)),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        inputParams_.M != scaleShape.GetDim(0),
        OP_LOGE(inputParams_.opName, "x dim0(%ld) must be the same as scale dim0(%ld).", inputParams_.M, scaleShape.GetDim(0)),
        return ge::GRAPH_FAILED);

    inputParams_.numBlocks = inputParams_.N / inputParams_.K;

    return ge::GRAPH_SUCCESS;
}

void RotateQuantTiling::PrintTilingData()
{
    auto& tiling = tilingData_;
    OP_LOGD(inputParams_.opName, "=== RotateQuant Tiling Data ===");
    OP_LOGD(inputParams_.opName, "M:                    [%d]", tiling.M);
    OP_LOGD(inputParams_.opName, "N:                    [%d]", tiling.N);
    OP_LOGD(inputParams_.opName, "K:                    [%d]", tiling.K);
    OP_LOGD(inputParams_.opName, "loopM:                [%d]", tiling.loopM);
    OP_LOGD(inputParams_.opName, "dstType:              [%ld]", tiling.dstType);
    OP_LOGD(inputParams_.opName, "stepLoop:             [%ld]", tiling.stepLoop);
    OP_LOGD(inputParams_.opName, "--- AIC Core Parameters ---");
    OP_LOGD(inputParams_.opName, "aicCoreNum:           [%d]", tiling.aicCoreNum);
    OP_LOGD(inputParams_.opName, "numBlocks:            [%d]", tiling.numBlocks);
    OP_LOGD(inputParams_.opName, "rowPerCubeHeadCore:   [%u]", tiling.rowPerCubeHeadCore);
    OP_LOGD(inputParams_.opName, "rowPerCubeTailCore:   [%u]", tiling.rowPerCubeTailCore);
    OP_LOGD(inputParams_.opName, "--- AIV Core Parameters ---");
    OP_LOGD(inputParams_.opName, "aivCoreNum:           [%d]", tiling.aivCoreNum);
    OP_LOGD(inputParams_.opName, "headCoreNum:          [%u]", tiling.headCoreNum);
    OP_LOGD(inputParams_.opName, "multiRowNumHeadCore:  [%u]", tiling.multiRowNumHeadCore);
    OP_LOGD(inputParams_.opName, "rowPerHeadCore:       [%u]", tiling.rowPerHeadCore);
    OP_LOGD(inputParams_.opName, "rowPerVectorTailCore: [%u]", tiling.rowPerVectorTailCore);
    OP_LOGD(inputParams_.opName, "rowPerVectorLastCore: [%u]", tiling.rowPerVectorLastCore);
    OP_LOGD(inputParams_.opName, "lastUbRows:           [%u]", tiling.lastUbRows);
    OP_LOGD(inputParams_.opName, "ubSize:               [%u]", tiling.ubSize);
    OP_LOGD(inputParams_.opName, "=== End Tiling Data ===");
}

ge::graphStatus RotateQuantTiling::CaclVecTilingData()
{
    uint32_t aivCoreNum = compileInfo_.aivNum;
    uint32_t aicCoreNum = compileInfo_.aicNum;
    uint32_t ubSize = static_cast<uint32_t>(compileInfo_.ubSize);

    uint32_t maxHandleRowsPerUb = CalcMaxHandleRowsPerUb(ubSize);

    uint32_t loopM = (maxHandleRowsPerUb == 0) ? 0 :
                     ops::CeilDiv<uint32_t>(static_cast<uint32_t>(inputParams_.M), maxHandleRowsPerUb);
    uint32_t loopPerHeadCore = ops::CeilDiv<uint32_t>(loopM, aivCoreNum);
    uint32_t loopPerHeadCubeCore = ops::CeilDiv<uint32_t>(loopM, aicCoreNum);
    uint32_t rowPerHeadCore = loopPerHeadCore * maxHandleRowsPerUb;
    uint32_t usedCoreNum = ops::CeilDiv<uint32_t>(loopM, loopPerHeadCore);
    uint32_t headCoreNum = usedCoreNum - 1;
    uint32_t usedAicCoreNum = ops::CeilDiv<uint32_t>(usedCoreNum, VEC_NUM_PER_CUBE);
    uint32_t usedAivCoreNum = usedAicCoreNum * VEC_NUM_PER_CUBE;
    uint32_t rowPerCubeHeadCore = rowPerHeadCore * VEC_NUM_PER_CUBE;
    uint32_t rowPerCubeTailCore = static_cast<uint32_t>(inputParams_.M) - (usedAicCoreNum - 1) * rowPerCubeHeadCore;
    uint32_t rowPerVectorTailCore = ops::CeilDiv<uint32_t>(rowPerCubeTailCore, VEC_NUM_PER_CUBE);
    uint32_t rowPerVectorLastCore = rowPerCubeTailCore - rowPerVectorTailCore;

    uint32_t tailMod = rowPerVectorTailCore % maxHandleRowsPerUb;
    uint32_t lastMod = rowPerVectorLastCore % maxHandleRowsPerUb;
    uint32_t lastUbRows = (tailMod == 0 || lastMod == 0) ? maxHandleRowsPerUb : tailMod;

    uint32_t stepLoop;
    OP_TILING_CHECK(CalcStepLoop(maxHandleRowsPerUb, loopPerHeadCore, loopPerHeadCubeCore, stepLoop) != ge::GRAPH_SUCCESS,
                    OP_LOGE(inputParams_.opName, "CalcStepLoop failed."), return ge::GRAPH_FAILED);

    tilingData_.headCoreNum = headCoreNum;
    tilingData_.rowPerHeadCore = rowPerHeadCore;
    tilingData_.rowPerVectorTailCore = rowPerVectorTailCore;
    tilingData_.rowPerVectorLastCore = rowPerVectorLastCore;
    tilingData_.rowPerCubeTailCore = rowPerCubeTailCore;
    tilingData_.rowPerCubeHeadCore = rowPerCubeHeadCore;
    tilingData_.multiRowNumHeadCore = maxHandleRowsPerUb;
    tilingData_.ubSize = ubSize;
    tilingData_.M = static_cast<int32_t>(inputParams_.M);
    tilingData_.N = static_cast<int32_t>(inputParams_.N);
    tilingData_.K = static_cast<int32_t>(inputParams_.K);
    tilingData_.loopM = loopM;
    tilingData_.dstType = inputParams_.yDtype;
    tilingData_.aicCoreNum = usedAicCoreNum;
    tilingData_.aivCoreNum = usedAivCoreNum;
    tilingData_.numBlocks = static_cast<int32_t>(inputParams_.numBlocks);
    tilingData_.stepLoop = stepLoop;
    tilingData_.lastUbRows = lastUbRows;

    return ge::GRAPH_SUCCESS;
}

uint32_t RotateQuantTiling::CalcMaxHandleRowsPerUb(uint32_t ubSize)
{
    uint32_t maxUbLen = ubSize - RESERVED_LENGTH -
                        (NUM_TWO_FIVE_SIX + ONE_REPEAT_ELE + MEMORY_ALLOC_COEFFICIENT * ONE_REPEAT_ELE * sizeof(float));
    maxUbLen = maxUbLen / MULTI_ROW_NO_SMOOTH_SPLIT / ONE_REPEAT_ELE * ONE_REPEAT_ELE;
    maxUbLen = std::min(maxUbLen, static_cast<uint32_t>(MAX_UB_REPEAT_COUNT * ONE_REPEAT_ELE));
    uint32_t maxHandleRowsPerUb = maxUbLen / static_cast<uint32_t>(inputParams_.N);
    return std::min(maxHandleRowsPerUb, static_cast<uint32_t>(inputParams_.M));
}

ge::graphStatus RotateQuantTiling::CalcStepLoop(uint32_t maxHandleRowsPerUb, uint32_t loopPerHeadCore,
                                                 uint32_t loopPerHeadCubeCore, uint32_t& stepLoop)
{
    if (inputParams_.K <= STEP_LOOP_SMALL_K_THRESHOLD) {
        uint32_t groupSize = (loopPerHeadCubeCore <= STEP_LOOP_CUBE_CORE_THRESHOLD)
            ? STEP_LOOP_SMALL_GROUP_SIZE
            : STEP_LOOP_LARGE_GROUP_SIZE;
        stepLoop = std::max(ops::CeilDiv<uint32_t>(loopPerHeadCubeCore, groupSize), 1U);
    } else {
        stepLoop = std::max(BASE_M / (static_cast<uint32_t>(inputParams_.numBlocks) * maxHandleRowsPerUb * STEP_LOOP_UB_COEFF), 1U);
        if (stepLoop == 0) {
            OP_LOGE(context_->GetNodeName(), "stepLoop cannot be zero.");
            return ge::GRAPH_FAILED;
        }
        if (loopPerHeadCore / stepLoop >= STEP_LOOP_MIN_ITERS_PER_CORE) {
            stepLoop = BASE_M / (static_cast<uint32_t>(inputParams_.numBlocks) * maxHandleRowsPerUb);
        }
    }
    inputParams_.stepLoop = stepLoop;
    return ge::GRAPH_SUCCESS;
}

bool RotateQuantTiling::SetMatmulTiling()
{
    auto& mt = tilingData_.matmulTiling;
    matmul_tiling::MultiCoreMatmulTiling mm;

    matmul_tiling::DataType dataType = (inputParams_.xDtype == ge::DT_BF16) ?
        matmul_tiling::DataType::DT_BF16 : matmul_tiling::DataType::DT_FLOAT16;

    mm.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType, false);
    mm.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType, false);
    mm.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);

    mm.SetBias(false);
    mm.SetDim(compileInfo_.aicNum);

    int32_t rowsPerCore = tilingData_.stepLoop * 2 * tilingData_.multiRowNumHeadCore;
    int32_t numBlocks = tilingData_.numBlocks;
    int32_t K = tilingData_.K;

    mm.SetShape(rowsPerCore * numBlocks, K, K);

    if (mm.GetTiling(mt) == -1) {
        OP_LOGE(context_->GetNodeName(), "RotateQuant Get Tiling Failed!");
        return false;
    }

    mt.iterateOrder = 1;
    mt.usedCoreNum = compileInfo_.aicNum;
    mt.shareMode = 0;
    mt.dbL0C = 1;
    mt.dbL0A = 2;
    mt.dbL0B = 2;
    mt.baseM = BASE_M;
    mt.baseN = BASE_N;
    mt.baseK = BASE_K;
    mt.stepKa = 8;
    mt.stepKb = 4;
    mt.depthA1 = 16;
    mt.depthB1 = 8;
    mt.stepM = 1;
    mt.stepN = 1;

    return true;
}

bool RotateQuantTiling::SetRotateQuantTiling()
{
    return CaclVecTilingData() == ge::GRAPH_SUCCESS;
}

ge::graphStatus RotateQuantTiling::DoOpTiling()
{
    OP_LOGE_IF(!SetRotateQuantTiling(), ge::GRAPH_FAILED, "RotateQuant", "SetRotateQuantTiling failed.");
    inputParams_.usedCoreNum = tilingData_.aicCoreNum;
    OP_LOGE_IF(!SetMatmulTiling(), ge::GRAPH_FAILED, "RotateQuant", "SetMatmulTiling failed.");
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RotateQuantTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t RotateQuantTiling::GetTilingKey() const
{
    return 0;
}

ge::graphStatus RotateQuantTiling::GetWorkspaceSize()
{
    workspaceSize_ = SYS_WORKSPACE_SIZE +
                     static_cast<size_t>(inputParams_.M) * static_cast<size_t>(inputParams_.N) * sizeof(float);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RotateQuantTiling::PostTiling()
{
    context_->SetBlockDim(tilingData_.aicCoreNum);
    context_->SetScheduleMode(1);
    errno_t ret = memcpy_s(
        context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
        reinterpret_cast<void*>(&tilingData_), tilingDataSize_);
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    context_->GetRawTilingData()->SetDataSize(tilingDataSize_);
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = workspaceSize_;
    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("RotateQuant", RotateQuantTiling, 0);

static ge::graphStatus RotateQuantTilingFunc(gert::TilingContext* context)
{
    OP_LOGE_IF(context == nullptr, ge::GRAPH_FAILED, "RotateQuant", "TilingContext is null!");
    return Ops::NN::Optiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingParseForRotateQuant(gert::TilingParseContext* context)
{
    OP_LOGE_IF(context == nullptr, ge::GRAPH_FAILED, "RotateQuant", "TilingParseContext is null!");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(RotateQuant)
    .Tiling(RotateQuantTilingFunc)
    .TilingParse<RotateQuantCompileInfo>(TilingParseForRotateQuant);
} // namespace optiling
