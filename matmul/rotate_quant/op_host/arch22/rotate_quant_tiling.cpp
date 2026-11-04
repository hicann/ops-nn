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
#include "matmul/common/op_host/log_format_util.h"
#include "matmul/common/op_host/matmul_error_report.h"
#include "error_util.h"
#include "log/log.h"
#include "matmul/common/op_host/op_tiling/debug_tiling.h"
#include "platform/platform_infos_def.h"
#include "register/op_def_registry.h"
#include "op_host/tiling_key.h"
#include "op_host/tiling_templates_registry.h"
#include "matmul/common/op_host/math_util_nn.h"

namespace {
constexpr uint32_t X_INDEX = 0;
constexpr uint32_t ROT_INDEX = 1;
constexpr uint32_t GROUP_INDEX = 3;
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

namespace Ops {
namespace NN {
namespace RotateQuant {
ge::graphStatus RotateQuantTiling::GetPlatformInfo()
{
    if (compileInfo_.aicNum == 0) {
        OP_LOGE_WITHOUT_REPORT(context_->GetNodeName(), "Platform initialization failed; cannot calculate tiling.");
        return ge::GRAPH_FAILED;
    }
    if (compileInfo_.aivNum == 0) {
        Ops::NN::ReportMatmulTilingError(
            context_->GetNodeName(),
            "Platform vector core count must be positive, got 0; check platform initialization.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

void RotateQuantTiling::InitCompileInfo()
{
    if (context_ == nullptr) {
        Ops::NN::ReportMatmulTilingError("RotateQuant", "Tiling context is null; check the tiling invocation.");
        return;
    }
    auto platformInfoPtr = context_->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        Ops::NN::ReportMatmulTilingError(context_->GetNodeName(),
                                         "Platform information is null; check platform initialization.");
        return;
    }

    const auto& ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo_.ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfo_.l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, compileInfo_.l2Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfo_.l0ASize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfo_.l0BSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfo_.l0CSize);
    compileInfo_.aicNum = ascendcPlatform.GetCoreNumAic();
    compileInfo_.aivNum = ascendcPlatform.GetCoreNumAiv();

    if (compileInfo_.aicNum <= 0) {
        Ops::NN::ReportMatmulTilingError(
            context_->GetNodeName(),
            Ops::NN::FormatString("Platform cube core count must be positive, got %lu; check platform initialization.",
                                  static_cast<unsigned long>(compileInfo_.aicNum))
                .c_str());
        return;
    }
    OP_LOGD(context_->GetNodeName(), "RotateQuant InitCompileInfo Success");
}

ge::graphStatus RotateQuantTiling::GetShapeAttrsInfo()
{
    tilingDataSize_ = sizeof(RotateQuantOpt::RotateQuantTilingData);
    if (inputParams_.initFlag) {
        OP_LOGD(inputParams_.opName, "No need to get shape and attrs from tiling context again");
        return ge::GRAPH_SUCCESS;
    }

    inputParams_.opName = context_->GetNodeName();
    OP_LOGI(inputParams_.opName, "TilingContext: %s", Ops::NN::DebugTilingContext(context_).c_str());
    OP_TILING_CHECK(CheckContext() != ge::GRAPH_SUCCESS,
                    OP_LOGE_WITHOUT_REPORT(inputParams_.opName, "Invalid context."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(AnalyzeAttrs() != ge::GRAPH_SUCCESS, OP_LOGE_WITHOUT_REPORT(inputParams_.opName, "Invalid attrs."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(AnalyzeDtype() != ge::GRAPH_SUCCESS, OP_LOGE_WITHOUT_REPORT(inputParams_.opName, "Invalid dtypes."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(AnalyzeShapes() != ge::GRAPH_SUCCESS,
                    OP_LOGE_WITHOUT_REPORT(inputParams_.opName, "Invalid shapes."), return ge::GRAPH_FAILED);

    OP_LOGD(inputParams_.opName, "Input params: MNK[%ld, %ld, %ld], numBlocks[%ld].", inputParams_.M, inputParams_.N,
            inputParams_.K, inputParams_.numBlocks);

    inputParams_.initFlag = true;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RotateQuantTiling::CheckContext()
{
    OP_TILING_CHECK(context_->GetOptionalInputDesc(GROUP_INDEX) != nullptr,
                    OP_LOGE(inputParams_.opName, "groupList must be nullptr on this platform."),
                    return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->GetInputShape(X_INDEX) == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "x"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->GetInputDesc(X_INDEX) == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "x"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->GetInputShape(ROT_INDEX) == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "rotation"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->GetInputDesc(ROT_INDEX) == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "rotation"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->GetOutputShape(Y_INDEX) == nullptr,
                Ops::NN::ReportMatmulTilingError(context_->GetNodeName(),
                                                 "Output y shape storage is null; check tiling output initialization."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->GetOutputDesc(Y_INDEX) == nullptr,
                Ops::NN::ReportMatmulTilingError(context_->GetNodeName(),
                                                 "Output y descriptor is null; check tiling output initialization."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->GetOutputShape(SCALE_INDEX) == nullptr,
                Ops::NN::ReportMatmulTilingError(
                    context_->GetNodeName(), "Output scale shape storage is null; check tiling output initialization."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->GetOutputDesc(SCALE_INDEX) == nullptr,
                Ops::NN::ReportMatmulTilingError(
                    context_->GetNodeName(), "Output scale descriptor is null; check tiling output initialization."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->GetRawTilingData() == nullptr,
                Ops::NN::ReportMatmulTilingError(
                    context_->GetNodeName(), "Raw tiling data storage is null; check tiling output initialization."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->GetRawTilingData()->GetData() == nullptr,
                Ops::NN::ReportMatmulTilingError(context_->GetNodeName(),
                                                 "Raw tiling data buffer is null; check tiling output initialization."),
                return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        context_->GetRawTilingData()->GetCapacity() < tilingDataSize_,
        Ops::NN::ReportMatmulTilingError(
            inputParams_.opName, Ops::NN::FormatString("Tiling buffer capacity %zu bytes is smaller than the "
                                                       "required %zu bytes; check the tiling output buffer.",
                                                       context_->GetRawTilingData()->GetCapacity(), tilingDataSize_)
                                     .c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RotateQuantTiling::AnalyzeAttrs() { return ge::GRAPH_SUCCESS; }

ge::graphStatus RotateQuantTiling::AnalyzeDtype()
{
    inputParams_.xDtype = context_->GetInputDesc(X_INDEX)->GetDataType();
    auto rotDtype = context_->GetInputDesc(ROT_INDEX)->GetDataType();
    inputParams_.yDtype = context_->GetOutputDesc(Y_INDEX)->GetDataType();
    auto scaleDtype = context_->GetOutputDesc(SCALE_INDEX)->GetDataType();

    OP_TILING_CHECK(inputParams_.xDtype != ge::DT_FLOAT16 && inputParams_.xDtype != ge::DT_BF16,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(inputParams_.opName, "x",
                                                          Ops::Base::ToString(inputParams_.xDtype).c_str(),
                                                          "Supported dtypes are FLOAT16 and BF16."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        rotDtype != ge::DT_FLOAT16 && rotDtype != ge::DT_BF16,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(inputParams_.opName, "rotation", Ops::Base::ToString(rotDtype).c_str(),
                                              "Supported dtypes are FLOAT16 and BF16."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(inputParams_.xDtype != rotDtype,
                    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                        inputParams_.opName, "x, rotation",
                        Ops::NN::FormatString("%s, %s", Ops::Base::ToString(inputParams_.xDtype).c_str(),
                                              Ops::Base::ToString(rotDtype).c_str())
                            .c_str(),
                        "x and rotation must have the same dtype."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(inputParams_.yDtype != ge::DT_INT8 && inputParams_.yDtype != ge::DT_INT4,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(inputParams_.opName, "y",
                                                          Ops::Base::ToString(inputParams_.yDtype).c_str(),
                                                          "Supported output dtypes are INT8 and INT4."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        scaleDtype != ge::DT_FLOAT,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(inputParams_.opName, "scale", Ops::Base::ToString(scaleDtype).c_str(),
                                              "scale must have dtype FLOAT."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RotateQuantTiling::AnalyzeShapes()
{
    const auto& xShape = context_->GetInputShape(X_INDEX)->GetStorageShape();
    const auto& rotShape = context_->GetInputShape(ROT_INDEX)->GetStorageShape();
    const auto& yShape = context_->GetOutputShape(Y_INDEX)->GetStorageShape();
    const auto& scaleShape = context_->GetOutputShape(SCALE_INDEX)->GetStorageShape();

    inputParams_.M = xShape.GetDim(0);
    inputParams_.N = xShape.GetDim(1);
    inputParams_.K = rotShape.GetDim(0);
    int64_t rotK = rotShape.GetDim(1);

    OP_TILING_CHECK(
        inputParams_.N % inputParams_.K != 0,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            inputParams_.opName, "x, rotation",
            Ops::NN::FormatString("%s, %s", Ops::Base::ToString(xShape).c_str(), Ops::Base::ToString(rotShape).c_str())
                .c_str(),
            "x N must be divisible by rotation K."),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        inputParams_.K != rotK,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(inputParams_.opName, "rotation", Ops::Base::ToString(rotShape).c_str(),
                                              "rotation must be a square matrix."),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(inputParams_.N > N_MAX || inputParams_.N < N_MIN,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(inputParams_.opName, "x", Ops::Base::ToString(xShape).c_str(),
                                                          "x N must be between 128 and 16000."),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        inputParams_.K > K_MAX || inputParams_.K < K_MIN,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(inputParams_.opName, "rotation", Ops::Base::ToString(rotShape).c_str(),
                                              "rotation K must be between 16 and 1024."),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        inputParams_.M != yShape.GetDim(0),
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            inputParams_.opName, "x, y",
            Ops::NN::FormatString("%s, %s", Ops::Base::ToString(xShape).c_str(), Ops::Base::ToString(yShape).c_str())
                .c_str(),
            "x and y must have equal first dimensions."),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        inputParams_.N != yShape.GetDim(1),
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            inputParams_.opName, "x, y",
            Ops::NN::FormatString("%s, %s", Ops::Base::ToString(xShape).c_str(), Ops::Base::ToString(yShape).c_str())
                .c_str(),
            "x and y must have equal second dimensions."),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        inputParams_.M != scaleShape.GetDim(0),
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(inputParams_.opName, "x, scale",
                                               Ops::NN::FormatString("%s, %s", Ops::Base::ToString(xShape).c_str(),
                                                                     Ops::Base::ToString(scaleShape).c_str())
                                                   .c_str(),
                                               "x and scale must have equal first dimensions."),
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

    uint32_t loopM = (maxHandleRowsPerUb == 0) ?
                         0 :
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
    OP_TILING_CHECK(
        CalcStepLoop(maxHandleRowsPerUb, loopPerHeadCore, loopPerHeadCubeCore, stepLoop) != ge::GRAPH_SUCCESS,
        OP_LOGE_WITHOUT_REPORT(inputParams_.opName, "CalcStepLoop failed."), return ge::GRAPH_FAILED);

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
        uint32_t groupSize = (loopPerHeadCubeCore <= STEP_LOOP_CUBE_CORE_THRESHOLD) ? STEP_LOOP_SMALL_GROUP_SIZE :
                                                                                      STEP_LOOP_LARGE_GROUP_SIZE;
        stepLoop = std::max(ops::CeilDiv<uint32_t>(loopPerHeadCubeCore, groupSize), 1U);
    } else {
        stepLoop = std::max(
            BASE_M / (static_cast<uint32_t>(inputParams_.numBlocks) * maxHandleRowsPerUb * STEP_LOOP_UB_COEFF), 1U);
        if (stepLoop == 0) {
            Ops::NN::ReportMatmulTilingError(
                context_->GetNodeName(), "Calculated tiling stepLoop is zero; check the tiling loop configuration.");
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

    matmul_tiling::DataType dataType = (inputParams_.xDtype == ge::DT_BF16) ? matmul_tiling::DataType::DT_BF16 :
                                                                              matmul_tiling::DataType::DT_FLOAT16;

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
        Ops::NN::ReportMatmulTilingError(
            context_->GetNodeName(),
            "Matrix multiplication GetTiling returned -1; check the tiling shape and platform resources.");
        return false;
    }

    constexpr uint32_t kDbL0A = 2;
    constexpr uint32_t kDbL0B = 2;
    constexpr uint32_t kStepKa = 8;
    constexpr uint32_t kStepKb = 4;
    constexpr uint32_t kDepthA1 = 16;
    constexpr uint32_t kDepthB1 = 8;
    mt.iterateOrder = 1;
    mt.usedCoreNum = compileInfo_.aicNum;
    mt.shareMode = 0;
    mt.dbL0C = 1;
    mt.dbL0A = kDbL0A;
    mt.dbL0B = kDbL0B;
    mt.baseM = BASE_M;
    mt.baseN = BASE_N;
    mt.baseK = BASE_K;
    mt.stepKa = kStepKa;
    mt.stepKb = kStepKb;
    mt.depthA1 = kDepthA1;
    mt.depthB1 = kDepthB1;
    mt.stepM = 1;
    mt.stepN = 1;

    return true;
}

bool RotateQuantTiling::SetRotateQuantTiling() { return CaclVecTilingData() == ge::GRAPH_SUCCESS; }

ge::graphStatus RotateQuantTiling::DoOpTiling()
{
    OP_CHECK_IF(!SetRotateQuantTiling(), OP_LOGE_WITHOUT_REPORT("RotateQuant", "SetRotateQuantTiling failed."),
                return ge::GRAPH_FAILED);
    inputParams_.usedCoreNum = tilingData_.aicCoreNum;
    OP_CHECK_IF(!SetMatmulTiling(), OP_LOGE_WITHOUT_REPORT("RotateQuant", "SetMatmulTiling failed."),
                return ge::GRAPH_FAILED);
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RotateQuantTiling::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

uint64_t RotateQuantTiling::GetTilingKey() const { return 0; }

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
    errno_t ret = memcpy_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                           static_cast<void*>(&tilingData_), tilingDataSize_);
    if (ret != EOK) {
        Ops::NN::ReportMatmulTilingError(
            context_->GetNodeName(),
            Ops::NN::FormatString("Failed to serialize tiling data: memcpy_s returned %d, requiredBytes=%zu, "
                                  "capacityBytes=%zu; check the tiling output buffer.",
                                  ret, tilingDataSize_, context_->GetRawTilingData()->GetCapacity())
                .c_str());
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
    OP_CHECK_IF(context == nullptr,
                Ops::NN::ReportMatmulTilingError("RotateQuant", "Tiling context is null; check the tiling invocation."),
                return ge::GRAPH_FAILED);
    return Ops::NN::Optiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingParseForRotateQuant(gert::TilingParseContext* context)
{
    OP_CHECK_IF(context == nullptr,
                Ops::NN::ReportMatmulTilingError("RotateQuant",
                                                 "Tiling parse context is null; check the tiling prepare invocation."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(RotateQuant)
    .Tiling(RotateQuantTilingFunc)
    .TilingParse<RotateQuantCompileInfo>(TilingParseForRotateQuant);
} // namespace RotateQuant
} // namespace NN
} // namespace Ops
