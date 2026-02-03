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
 * \file weight_quant_batch_matmul_experiment_tiling.cpp
 * \brief
 */

#include "../op_kernel/weight_quant_batch_matmul_experiment_tiling_data.h"
#include "../op_kernel/weight_quant_batch_matmul_experiment_tiling_key.h"
#include "log/log.h"
#include "tiling/tiling_api.h"
#include "tiling_base/tiling_templates_registry.h"
#include "tiling_base/tiling_util.h"
#include "util/math_util.h"

using WeightQuantBatchMatmulExperimental::WeightQuantBatchMatmulExperimentTilingData;
namespace optiling {
namespace weight_quant_batch_matmul_experiment {
constexpr uint32_t WS_SYS_SIZE = 16U * 1024U * 1024U;  // 16MB
constexpr uint32_t DB_SIZE = 2UL;
constexpr uint32_t BLOCK_BYTE_SIZE = 32UL;
constexpr uint32_t H_ALIGNED = 16UL;
constexpr uint32_t BASIC_BLOCK_SIZE_256 = 256UL;
constexpr uint32_t BASIC_BLOCK_SIZE_128 = 128UL;
constexpr uint32_t BASIC_BLOCK_K_128_BYTE = 128UL;
constexpr uint32_t DB_OFF_SIZE = 1UL;
constexpr uint32_t ITER_COL_FIRST = 0UL;
constexpr uint32_t DATA_SIZE_FP32 = 4UL;
constexpr uint32_t CO_SIZE_FP32 = 8UL;
constexpr uint32_t BIAS_TABLE_NUM = 256UL;
constexpr uint32_t NUM_HALF = 2UL;
constexpr uint32_t BASE_STEP = 1UL;
constexpr uint32_t BASIC_ALIGN_FP32 = 128UL;
constexpr uint32_t BASIC_ALIGN_16 = 16;
constexpr uint32_t BASIC_ALIGN_256 = 256;

struct WeightQuantBatchMatmulExperimentCompileInfo {
    uint64_t aicNum{0UL};
    uint64_t aivNum{0UL};
    uint64_t ubSize{0UL};
    uint64_t l1Size{0UL};
    uint64_t l2Size{0UL};
    uint64_t l0ASize{0UL};
    uint64_t l0BSize{0UL};
    uint64_t l0CSize{0UL};
    uint64_t btSize{0UL};
    platform_ascendc::SocVersion socVersion;
};

struct WeightQuantBatchMatmulExperimentArgs {
    const char *opName = nullptr;
    ge::DataType xType = ge::DT_FLOAT;
    ge::DataType weightType = ge::DT_FLOAT;
    ge::DataType yType = ge::DT_FLOAT;
    ge::Format aFormat = ge::FORMAT_ND;
    ge::Format weightFormat = ge::FORMAT_ND;
    ge::Format yFormat = ge::FORMAT_ND;
    uint64_t mValue = 0;
    uint64_t nValue = 0;
    uint64_t kValue = 0;
};

inline uint64_t CeilDiv(uint64_t x, uint64_t align)
{
    if (align == 0) {
        return x;
    }
    return (x + align - 1) / align;
}

inline uint64_t CeilAlign(uint64_t x, uint64_t align) { return CeilDiv(x, align) * align; }

static ge::graphStatus GetPlatformInfo(gert::TilingContext *context,
                                       WeightQuantBatchMatmulExperimentCompileInfo *compileInfoPtr)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context->GetNodeName(), "context is null"), return ge::GRAPH_FAILED);
    fe::PlatFormInfos *platformInfos = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfos == nullptr, OP_LOGE(context->GetNodeName(), "platform info is null"),
                return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfos);
    compileInfoPtr->aicNum = ascendcPlatform.GetCoreNumAic();
    if (compileInfoPtr->aicNum == 0) {
        OP_LOGE(context->GetNodeName(), "aicNum is zero.");
        return ge::GRAPH_FAILED;
    }
    compileInfoPtr->socVersion = ascendcPlatform.GetSocVersion();
    // compileInfoPtr->btSize = 1024UL;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoPtr->l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfoPtr->l0ASize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfoPtr->l0BSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfoPtr->l0CSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, compileInfoPtr->l2Size);
    OP_LOGI(context->GetNodeName(), "parse compile info soc:%d, l1Size:%lu, l2Size:%lu, coreNum:%lu",
            static_cast<int32_t>(compileInfoPtr->socVersion), compileInfoPtr->l1Size, compileInfoPtr->l2Size,
            compileInfoPtr->aicNum);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetShapeAttrsInfo(gert::TilingContext *context, WeightQuantBatchMatmulExperimentArgs *argsPtr)
{
    argsPtr->opName = context->GetNodeName();
    OP_CHECK_IF(argsPtr->opName == nullptr, OP_LOGE("weight_quant_batch_matmul_experiment", "get op name invalid"),
                return ge::GRAPH_FAILED);
    // 输入张亮shape与描述信息
    auto shapeX = context->GetInputShape(0);
    auto shapeWeight = context->GetInputShape(1);
    auto descX = context->GetInputDesc(0);
    auto descWeight = context->GetInputDesc(1);
    auto descOut = context->GetOutputDesc(0);
    OP_CHECK_IF(
        shapeX == nullptr || shapeWeight == nullptr || descX == nullptr || descWeight == nullptr || descOut == nullptr,
        OP_LOGE(argsPtr->opName, "the input is invalid"), return ge::GRAPH_FAILED);

    // 输入、输出格式信息
    argsPtr->aFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(descX->GetStorageFormat()));
    argsPtr->weightFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(descWeight->GetStorageFormat()));
    argsPtr->yFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(descOut->GetStorageFormat()));
    OP_CHECK_IF(argsPtr->aFormat != ge::FORMAT_ND || argsPtr->weightFormat != ge::FORMAT_ND ||
                    argsPtr->yFormat != ge::FORMAT_ND,
                OP_LOGE(argsPtr->opName, "input and output only support nd format"), return ge::GRAPH_FAILED);

    // 输入、输出数据类型
    argsPtr->xType = descX->GetDataType();
    argsPtr->weightType = descWeight->GetDataType();
    argsPtr->yType = descOut->GetDataType();
    OP_CHECK_IF(
        argsPtr->xType != ge::DT_FLOAT16 || argsPtr->weightType != ge::DT_INT4 || argsPtr->yType != ge::DT_FLOAT16,
        OP_LOGE(argsPtr->opName, "current only support x is FLOAT16, weight is int4 and y is float16"),
        return ge::GRAPH_FAILED);

    // 获取input的m/n/k
    const size_t xDimNum = shapeX->GetStorageShape().GetDimNum();
    const size_t weightDimNum = shapeWeight->GetStorageShape().GetDimNum();
    OP_CHECK_IF(xDimNum != 2 || weightDimNum != 2,
                OP_LOGE(argsPtr->opName, "the input shape dimensions are not equal to 2"), return ge::GRAPH_FAILED);

    argsPtr->mValue = shapeX->GetStorageShape().GetDim(0);
    argsPtr->kValue = shapeX->GetStorageShape().GetDim(1);
    argsPtr->nValue = shapeWeight->GetStorageShape().GetDim(1);
    OP_LOGD(context->GetNodeName(), "parse op shape info m:%d, n:%d, k:%d", static_cast<int32_t>(argsPtr->mValue),
            static_cast<int32_t>(argsPtr->nValue), static_cast<int32_t>(argsPtr->kValue));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus DoOpTiling(WeightQuantBatchMatmulExperimentTilingData *tilingDataPtr,
                                  WeightQuantBatchMatmulExperimentCompileInfo *compileInfoPtr,
                                  WeightQuantBatchMatmulExperimentArgs *argsPtr)
{
    tilingDataPtr->matmulTiling.usedCoreNum = compileInfoPtr->aicNum;
    tilingDataPtr->matmulTiling.M = argsPtr->mValue;
    tilingDataPtr->matmulTiling.N = argsPtr->nValue;
    tilingDataPtr->matmulTiling.Ka = argsPtr->kValue;
    tilingDataPtr->matmulTiling.Kb = argsPtr->kValue;
    tilingDataPtr->matmulTiling.singleCoreM = 3 * argsPtr->mValue;  // 实际执行mm的shape，m轴展开了3倍
    tilingDataPtr->matmulTiling.singleCoreN = 512;                  // 示例代码，n默认按照512切分
    tilingDataPtr->matmulTiling.singleCoreK = argsPtr->kValue;
    tilingDataPtr->matmulTiling.baseM = CeilAlign(tilingDataPtr->matmulTiling.singleCoreM, 16);
    tilingDataPtr->matmulTiling.baseN = tilingDataPtr->matmulTiling.singleCoreN;
    tilingDataPtr->matmulTiling.baseK = 128;  // 示例代码，group size 固定128
    tilingDataPtr->matmulTiling.depthA1 = 2;
    tilingDataPtr->matmulTiling.depthB1 = 2;
    tilingDataPtr->matmulTiling.stepM = 1;
    tilingDataPtr->matmulTiling.stepN = 1;
    tilingDataPtr->matmulTiling.stepKa = 1;
    tilingDataPtr->matmulTiling.stepKb = 1;
    tilingDataPtr->matmulTiling.iterateOrder = 0;
    tilingDataPtr->matmulTiling.dbL0C = 1;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus PostTiling(gert::TilingContext *context,
                                  WeightQuantBatchMatmulExperimentCompileInfo *compileInfoPtr,
                                  WeightQuantBatchMatmulExperimentTilingData *tilingDataPtr)
{
    size_t tilingDataSize = sizeof(WeightQuantBatchMatmulExperimentTilingData);
    auto ret = memcpy_s(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity(),
                        reinterpret_cast<void *>(tilingDataPtr), tilingDataSize);
    if (ret != EOK) {
        OP_LOGE(context->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    context->GetRawTilingData()->SetDataSize(tilingDataSize);
    context->SetBlockDim(tilingDataPtr->matmulTiling.usedCoreNum);
    context->SetScheduleMode(1);

    uint64_t msdTemplate = BASIC_MSD; // 可切换至PRELOAD_MSD流水
    // 生成tilingkey
    uint64_t tilingKey = GET_TPL_TILING_KEY(msdTemplate);
    OP_LOGI(context->GetNodeName(), "Tiling Key is 0x%x", tilingKey);
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    size_t usrSize = 64 * 1024 * 1024;  // workspace统一预留64M
    currentWorkspace[0] = usrSize + sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

// 1.Tiling需要获取的运行环境信息，包括可用核数、UB大小等，并将获取到的信息传递给'CompileInfo'
// 由于自动生成的aclnn接口实现不调用该函数，直接返回'ge::GRAPH_SUCCESS'即可
static ge::graphStatus TilingParseForWeightQuantBatchMatmulExperiment(
    [[maybe_unused]] gert::TilingParseContext *context)
{
    return ge::GRAPH_SUCCESS;
}

// 2. Tiling计算主入口
static ge::graphStatus WeightQuantBatchMatmulExperimentTilingFunc(gert::TilingContext *context)
{
    // 2.1 平台信息
    WeightQuantBatchMatmulExperimentCompileInfo *compileInfoPtr = new WeightQuantBatchMatmulExperimentCompileInfo;
    auto ret = GetPlatformInfo(context, compileInfoPtr);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    // 2.2 输入信息
    WeightQuantBatchMatmulExperimentArgs *argsPtr = new WeightQuantBatchMatmulExperimentArgs;
    ret = GetShapeAttrsInfo(context, argsPtr);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    // 2.3 计算Tiling参数
    WeightQuantBatchMatmulExperimentTilingData *tilingDataPtr =
        context->GetTilingData<WeightQuantBatchMatmulExperimentTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tilingDataPtr);
    ret = DoOpTiling(tilingDataPtr, compileInfoPtr, argsPtr);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    // 2.4 设置tilingkey
    ret = PostTiling(context, compileInfoPtr, tilingDataPtr);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    // 2.5设置workspaceSize
    GetWorkspaceSize(context);
    return ge::GRAPH_SUCCESS;
}

// 3. Tiling注册入口
IMPL_OP_OPTILING(WeightQuantBatchMatmulExperiment)
    .Tiling(WeightQuantBatchMatmulExperimentTilingFunc)
    .TilingParse<WeightQuantBatchMatmulExperimentCompileInfo>(TilingParseForWeightQuantBatchMatmulExperiment);
}  // namespace weight_quant_batch_matmul_experiment
}  // namespace optiling