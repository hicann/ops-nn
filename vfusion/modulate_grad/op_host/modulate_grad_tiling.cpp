/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file modulate_grad_tiling.cc
 * \brief
 */
#include "modulate_grad_tiling.h"
#include <cassert>
#include <string>
#include <vector>
#include "error_util.h"
#include "log/log.h"
#include "op_host/tiling_util.h"
#include "platform/platform_infos_def.h"
#include "platform/platform_info.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "util/math_util.h"
#include "modulate_grad_tiling_arch35.h"

namespace optiling {
struct ModulateGradTilingData {
    uint64_t B;
    uint64_t L;
    uint64_t D;
    uint32_t dataType = 2;
    uint32_t block_dim;
    uint32_t dataTypeSize = 4;

    uint32_t splitB;
    uint32_t coresPerB;
    uint32_t usedCores;
    uint32_t formerNum;
    uint32_t formerLength;
    uint32_t tailNum;
    uint32_t tailLength;
    uint32_t has_scale;
    uint32_t has_shift;
};

constexpr uint32_t DATA_TYPE_SIZE[] = {2, 2, 4};
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t UB_BLOCK_NUM = 384;
constexpr uint32_t MAX_AVAILABLE_UB_BLOCK_NUM = UB_BLOCK_NUM / BUFFER_NUM * BUFFER_NUM;

// 合法输入下 tiling 内部/环境失败统一上报正式码 EZ1008（Execution_Error），携带 opName 与 reason。
static ge::graphStatus ReportTilingFailure(const char* opName, const std::string& reason)
{
    const std::vector<const char*> keys = {"opName", "reason"};
    const std::vector<const char*> values = {opName, reason.c_str()};
    REPORT_PREDEFINED_ERR_MSG("EZ1008", keys, values);
    OP_LOGE_WITHOUT_REPORT(opName, "%s", reason.c_str());
    return ge::GRAPH_FAILED;
}

static bool IsOptionalInputPresent(const gert::StorageShape* input)
{
    if (input == nullptr) {
        return false;
    }
    const gert::Shape shape = input->GetStorageShape();
    return shape.GetDimNum() > 0 && shape.GetShapeSize() > 0;
}

void PrintTilingData(gert::TilingContext* context, ModulateGradTilingData& tiling)
{
    OP_LOGD(context, "B:%lu.", tiling.B);
    OP_LOGD(context, "L:%lu", tiling.L);
    OP_LOGD(context, "D:%lu.", tiling.D);
    OP_LOGD(context, "dataType:%u.", tiling.dataType);
    OP_LOGD(context, "block_dim:%u.", tiling.block_dim);
    OP_LOGD(context, "dataTypeSize:%u.", tiling.dataTypeSize);
    OP_LOGD(context, "splitB:%u.", tiling.splitB);
    OP_LOGD(context, "coresPerB:%u.", tiling.coresPerB);
    OP_LOGD(context, "usedCores:%u.", tiling.usedCores);
    OP_LOGD(context, "formerLength:%u.", tiling.formerLength);
    OP_LOGD(context, "tailNum:%u.", tiling.tailNum);
    OP_LOGD(context, "tailLength:%u.", tiling.tailLength);
    OP_LOGD(context, "has_scale:%u.", tiling.has_scale);
    OP_LOGD(context, "has_shift:%u.", tiling.has_shift);
}
void GenerateTilingData(ModulateGradTilingData* tiling, uint32_t numBlocks)
{
    uint64_t B = tiling->B;
    uint64_t D = tiling->D;
    if (numBlocks == 0 || B == 0 || D == 0 || B * D == 0) {
        tiling->splitB = 0;
        tiling->coresPerB = 1;
        tiling->usedCores = 1;
        tiling->formerNum = 0;
        tiling->formerLength = 0;
        tiling->tailNum = 1;
        tiling->tailLength = 0;
        return;
    }
    numBlocks = numBlocks < B * D ? numBlocks : B * D;
    uint32_t splitBlock = numBlocks / 2;
    tiling->splitB = (B < splitBlock) ? 1 : 0;

    if (tiling->splitB) {
        tiling->coresPerB = numBlocks / B;
        tiling->usedCores = tiling->coresPerB * B;

        uint32_t dPerCore = D / tiling->coresPerB;
        uint32_t remainderD = D % tiling->coresPerB;

        tiling->formerNum = remainderD;
        tiling->formerLength = dPerCore + 1;
        tiling->tailNum = tiling->coresPerB - remainderD;
        tiling->tailLength = dPerCore;
    } else {
        numBlocks = numBlocks < B ? numBlocks : B;
        tiling->coresPerB = 1;
        uint32_t bPerCore = B / numBlocks;
        tiling->usedCores = numBlocks;
        uint32_t remainderB = B % numBlocks;

        tiling->formerNum = (remainderB == 0) ? 0 : remainderB;
        tiling->formerLength = (remainderB == 0) ? 0 : bPerCore + 1;
        tiling->tailNum = (remainderB == 0) ? numBlocks : (numBlocks - remainderB);
        tiling->tailLength = bPerCore;
    }
}
static ge::graphStatus ValidateModulateGradTilingInputs(gert::TilingContext* context)
{
    constexpr uint32_t GRAD_OUTPUT_INDEX = 0;
    constexpr uint32_t INPUT_INDEX = 1;
    constexpr uint32_t SCALE_INDEX = 2;
    constexpr uint32_t SHIFT_INDEX = 3;
    const gert::StorageShape* gradOutputShape = context->GetInputShape(GRAD_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradOutputShape);
    const gert::StorageShape* inputShape = context->GetInputShape(INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    const gert::Shape gradOutputStorage = gradOutputShape->GetStorageShape();
    const gert::Shape inputStorage = inputShape->GetStorageShape();
    if (gradOutputStorage.GetDimNum() != 3 || inputStorage.GetDimNum() != 3) {
        OP_LOGE(context, "modulate_grad requires grad_output and input rank 3, got %zu and %zu.",
                gradOutputStorage.GetDimNum(), inputStorage.GetDimNum());
        return ge::GRAPH_FAILED;
    }
    for (size_t i = 0; i < 3; ++i) {
        // Tiling runs on concrete runtime shapes; negative dims are unresolved or invalid.
        if (gradOutputStorage.GetDim(i) < 0 || inputStorage.GetDim(i) < 0) {
            OP_LOGE(context, "modulate_grad requires non-negative dims at tiling, grad_output dim %ld, input dim %ld.",
                    gradOutputStorage.GetDim(i), inputStorage.GetDim(i));
            return ge::GRAPH_FAILED;
        }
        if (gradOutputStorage.GetDim(i) != inputStorage.GetDim(i)) {
            OP_LOGE(context, "modulate_grad requires grad_output shape equal to input shape, dim %zu: %ld vs %ld.", i,
                    gradOutputStorage.GetDim(i), inputStorage.GetDim(i));
            return ge::GRAPH_FAILED;
        }
    }
    const int64_t B = inputStorage.GetDim(0);
    const int64_t D = inputStorage.GetDim(2);
    const bool inputEmpty = (inputStorage.GetShapeSize() == 0);
    auto checkOptional = [&](const gert::StorageShape* opt, const char* name) -> ge::graphStatus {
        if (opt == nullptr) {
            return ge::GRAPH_SUCCESS;
        }
        const gert::Shape s = opt->GetStorageShape();
        // An absent optional input is kept as a rank-0 or zero-size empty placeholder;
        // treat every zero-size optional as absent and skip further validation.
        if (s.GetDimNum() == 0 || s.GetShapeSize() == 0) {
            if (!inputEmpty) {
                OP_LOGD(context, "modulate_grad %s is empty (treated as absent).", name);
            }
            return ge::GRAPH_SUCCESS;
        }
        if (inputEmpty) {
            OP_LOGE(context, "modulate_grad requires %s empty or absent when grad_output/input is empty.", name);
            return ge::GRAPH_FAILED;
        }
        if (s.GetDimNum() != 2) {
            OP_LOGE(context, "modulate_grad requires %s rank 2, got %zu.", name, s.GetDimNum());
            return ge::GRAPH_FAILED;
        }
        if (s.GetDim(0) < 0 || s.GetDim(1) < 0) {
            OP_LOGE(context, "modulate_grad requires non-negative %s dims at tiling.", name);
            return ge::GRAPH_FAILED;
        }
        if (static_cast<int64_t>(s.GetDim(0)) != B || static_cast<int64_t>(s.GetDim(1)) != D) {
            OP_LOGE(context, "modulate_grad requires %s shape [%ld, %ld], got [%ld, %ld].", name, B, D, s.GetDim(0),
                    s.GetDim(1));
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    };
    const gert::StorageShape* scaleShape = context->GetOptionalInputShape(SCALE_INDEX);
    const gert::StorageShape* shiftShape = context->GetOptionalInputShape(SHIFT_INDEX);
    if (checkOptional(scaleShape, "scale") != ge::GRAPH_SUCCESS ||
        checkOptional(shiftShape, "shift") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const ge::DataType inputDtype = context->GetInputDesc(INPUT_INDEX)->GetDataType();
    if (inputDtype != ge::DT_FLOAT16 && inputDtype != ge::DT_BF16 && inputDtype != ge::DT_FLOAT) {
        OP_LOGE(context, "modulate_grad unsupported input dtype %d.", static_cast<int32_t>(inputDtype));
        return ge::GRAPH_FAILED;
    }
    const ge::DataType gradOutputDtype = context->GetInputDesc(GRAD_OUTPUT_INDEX)->GetDataType();
    if (gradOutputDtype != inputDtype) {
        OP_LOGE(context, "modulate_grad requires grad_output dtype %d equal to input dtype %d.",
                static_cast<int32_t>(gradOutputDtype), static_cast<int32_t>(inputDtype));
        return ge::GRAPH_FAILED;
    }
    // 输入 format 校验（仅必选输入，其 IR 索引不受可选输入压缩影响）。
    const ge::Format gradOutputFormat = context->GetInputDesc(GRAD_OUTPUT_INDEX)->GetFormat().GetStorageFormat();
    const ge::Format inputFormat = context->GetInputDesc(INPUT_INDEX)->GetFormat().GetStorageFormat();
    if (gradOutputFormat != ge::FORMAT_ND || inputFormat != ge::FORMAT_ND) {
        OP_LOGE(context, "modulate_grad requires ND format, got grad_output format %d, input format %d.",
                static_cast<int32_t>(gradOutputFormat), static_cast<int32_t>(inputFormat));
        return ge::GRAPH_FAILED;
    }
    // 可选输入 dtype/format 校验：仅校验在场（非零尺寸）的 scale/shift，缺席或零尺寸占位
    // 按“视为缺席”约定跳过；desc 经 IR 索引获取，规避 TTK 压缩列表的运行时索引漂移，
    // 在场但 desc 未实例化时无校验依据，跳过。
    auto checkOptionalDesc = [&](const gert::StorageShape* opt, uint32_t irIndex, const char* name) -> ge::graphStatus {
        if (!IsOptionalInputPresent(opt)) {
            return ge::GRAPH_SUCCESS;
        }
        const auto* desc = context->GetOptionalInputDesc(irIndex);
        if (desc == nullptr) {
            return ge::GRAPH_SUCCESS;
        }
        const ge::DataType dtype = desc->GetDataType();
        if (dtype != inputDtype) {
            OP_LOGE(context, "modulate_grad requires %s dtype %d equal to input dtype %d.", name,
                    static_cast<int32_t>(dtype), static_cast<int32_t>(inputDtype));
            return ge::GRAPH_FAILED;
        }
        const ge::Format format = desc->GetFormat().GetStorageFormat();
        if (format != ge::FORMAT_ND) {
            OP_LOGE(context, "modulate_grad requires %s ND format, got format %d.", name, static_cast<int32_t>(format));
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    };
    if (checkOptionalDesc(scaleShape, SCALE_INDEX, "scale") != ge::GRAPH_SUCCESS ||
        checkOptionalDesc(shiftShape, SHIFT_INDEX, "shift") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // 输出 dtype/shape 校验（可选输出仅在对应可选输入在场时校验）。可选输出可能未被实例化：
    // GE 图模式保留空占位（输出总数=3，IR 索引即运行时索引），TTK kernel 直连模式按在场压缩
    // 运行时输出列表（输出总数=在场数，在场的可选输出依次排在 grad_input 之后）。
    // tiling 运行时不提供输出实例信息（GetIrOutputInstanceInfo 为空），按输出总数推导运行时索引。
    const bool scalePresent = IsOptionalInputPresent(scaleShape);
    const bool shiftPresent = IsOptionalInputPresent(shiftShape);
    const size_t outputNum = context->GetComputeNodeOutputNum();
    const uint32_t gradScaleIdx = 1U;
    const uint32_t gradShiftIdx = (outputNum < 3U) ? (1U + (scalePresent ? 1U : 0U)) : 2U;
    auto checkOutput = [&](uint32_t index, const gert::StorageShape* expect, bool required,
                           const char* name) -> ge::graphStatus {
        const auto* desc = context->GetOutputDesc(index);
        const gert::StorageShape* out = context->GetOutputShape(index);
        if (desc == nullptr || out == nullptr) {
            if (required) {
                OP_LOGE(context, "modulate_grad missing required output %s desc/shape.", name);
                return ge::GRAPH_FAILED;
            }
            return ge::GRAPH_SUCCESS;
        }
        const ge::DataType dtype = desc->GetDataType();
        if (dtype != ge::DT_UNDEFINED && dtype != inputDtype) {
            OP_LOGE(context, "modulate_grad requires %s dtype %d equal to input dtype %d.", name,
                    static_cast<int32_t>(dtype), static_cast<int32_t>(inputDtype));
            return ge::GRAPH_FAILED;
        }
        if (expect != nullptr) {
            const gert::Shape outShape = out->GetStorageShape();
            const gert::Shape expectShape = expect->GetStorageShape();
            if (outShape.GetDimNum() != expectShape.GetDimNum()) {
                OP_LOGE(context, "modulate_grad requires %s rank %zu, got %zu.", name, expectShape.GetDimNum(),
                        outShape.GetDimNum());
                return ge::GRAPH_FAILED;
            }
            for (size_t i = 0; i < outShape.GetDimNum(); ++i) {
                if (outShape.GetDim(i) != expectShape.GetDim(i)) {
                    OP_LOGE(context, "modulate_grad requires %s dim %zu equal to %ld, got %ld.", name, i,
                            expectShape.GetDim(i), outShape.GetDim(i));
                    return ge::GRAPH_FAILED;
                }
            }
        }
        return ge::GRAPH_SUCCESS;
    };
    if (checkOutput(0, inputShape, true, "grad_input") != ge::GRAPH_SUCCESS ||
        checkOutput(gradScaleIdx, scalePresent ? scaleShape : nullptr, scalePresent, "grad_scale") !=
            ge::GRAPH_SUCCESS ||
        checkOutput(gradShiftIdx, shiftPresent ? shiftShape : nullptr, shiftPresent, "grad_shift") !=
            ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    constexpr uint32_t INPUT_INDEX = 1;
    constexpr uint32_t SCALE_INDEX = 2;
    constexpr uint32_t SHIFT_INDEX = 3;
    if (ValidateModulateGradTilingInputs(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const gert::StorageShape* scaleInput = context->GetOptionalInputShape(SCALE_INDEX);
    const gert::StorageShape* shiftInput = context->GetOptionalInputShape(SHIFT_INDEX);
    const bool scalePresent = IsOptionalInputPresent(scaleInput);
    const bool shiftPresent = IsOptionalInputPresent(shiftInput);
    if (Ops::NN::OpTiling::IsRegbaseSocVersion(context)) {
        return ModulateGradTilingArch35(context, scalePresent, shiftPresent);
    }
    OP_LOGD(context->GetNodeName(), "Begin the tiling process for the non-regbase architecture");
    ModulateGradTiling tilingData;
    ModulateGradTilingData tiling{};
    const gert::StorageShape* input_shape = context->GetInputShape(INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, input_shape);
    const gert::Shape inputStorageShape = input_shape->GetStorageShape();
    if (inputStorageShape.GetDimNum() != 3) {
        OP_LOGE(context, "modulate_grad input rank must be 3, got %zu.", inputStorageShape.GetDimNum());
        return ge::GRAPH_FAILED;
    }
    const uint64_t B = inputStorageShape.GetDim(0);
    const uint64_t L = inputStorageShape.GetDim(1);
    const uint64_t D = inputStorageShape.GetDim(2);
    const ge::DataType inputDtype = context->GetInputDesc(INPUT_INDEX)->GetDataType();
    switch (inputDtype) {
        case ge::DT_FLOAT16:
            tiling.dataType = 0;
            tiling.dataTypeSize = 2;
            break;
        case ge::DT_BF16:
            tiling.dataType = 1;
            tiling.dataTypeSize = 2;
            break;
        case ge::DT_FLOAT:
            tiling.dataType = 2;
            tiling.dataTypeSize = 4;
            break;
        default:
            OP_LOGE(context, "modulate_grad unsupported input dtype %d.", static_cast<int32_t>(inputDtype));
            return ge::GRAPH_FAILED;
    }
    tiling.B = B;
    tiling.L = L;
    tiling.D = D;
    tiling.has_scale = scalePresent ? 1U : 0U;
    tiling.has_shift = shiftPresent ? 1U : 0U;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t block_dim = ascendcPlatform.GetCoreNumAiv();
    OP_LOGD(context, "ascendcPlatform CoreNum: %u.", block_dim);
    tiling.block_dim = block_dim;
    GenerateTilingData(&tiling, block_dim);
    context->SetBlockDim(tiling.usedCores);

    tilingData.set_B(tiling.B);
    tilingData.set_L(tiling.L);
    tilingData.set_D(tiling.D);
    tilingData.set_dataType(tiling.dataType);
    tilingData.set_block_dim(tiling.block_dim);
    tilingData.set_dataTypeSize(tiling.dataTypeSize);
    tilingData.set_splitB(tiling.splitB);
    tilingData.set_coresPerB(tiling.coresPerB);
    tilingData.set_usedCores(tiling.usedCores);
    tilingData.set_formerNum(tiling.formerNum);
    tilingData.set_formerLength(tiling.formerLength);
    tilingData.set_tailNum(tiling.tailNum);
    tilingData.set_tailLength(tiling.tailLength);
    tilingData.set_has_scale(tiling.has_scale);
    tilingData.set_has_shift(tiling.has_shift);

    PrintTilingData(context, tiling);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = ascendcPlatform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus TilingPrepareForModulateGrad(gert::TilingParseContext* context)
{
    OP_LOGD(context, "Tiling Prepare For ModulateGrad start.");
    auto compileInfo = context->GetCompiledInfo<ModulateGradCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    if (compileInfo->totalCoreNum == 0) {
        return ReportTilingFailure(context->GetNodeName(), "the platform reports 0 available AIV cores (coreNum is 0)");
    }
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSizePlatForm = ubSizePlatForm;
    OP_LOGD(context, "ub_size_platform is %lu.", compileInfo->ubSizePlatForm);
    OP_LOGD(context, "Tiling Prepare For ModulateGrad end.");
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(ModulateGrad).Tiling(TilingFunc).TilingParse<ModulateGradCompileInfo>(TilingPrepareForModulateGrad);
} // namespace optiling
