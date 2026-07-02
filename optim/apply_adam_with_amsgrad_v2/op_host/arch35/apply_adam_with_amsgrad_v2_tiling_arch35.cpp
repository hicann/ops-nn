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
 * \file apply_adam_with_amsgrad_v2_tiling.cpp
 * \brief ApplyAdamWithAmsgradV2 Tiling 实现（arch35 / DAV_3510 / Ascend950PR-DT / UB ≈ 248KB）
 *
 * fp32 逐元素 5 步路径。
 *   核数 GetCoreNumAiv 动态获取，UB GetCoreMemSize 动态获取，均禁止硬编码。
 *   UB 占用：5 in(VECIN) + 4 out(VECOUT) × 4B + 7 fp32 buf × 4B = 16×4=64 B/elem，含安全余量取 72。
 */
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"

#include <cstring>

#include "../../op_kernel/arch35/apply_adam_with_amsgrad_v2_tiling_data.h"
#include "../../op_kernel/arch35/apply_adam_with_amsgrad_v2_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr size_t WORKSPACE_NUM = 1;
// fp32 直算：5 in(VECIN) + 4 out(VECOUT) × 4B + 7 fp32 buf × 4B = 16×4=64；含安全余量取 72
constexpr int64_t FP32_BYTES_PER_ELEM = 72;

// 输入索引（与 op_def Input 注册顺序严格对齐）
constexpr int IDX_VAR = 0;
constexpr int IDX_M = 1;
constexpr int IDX_V = 2;
constexpr int IDX_VHAT = 3;
constexpr int IDX_BETA1_POWER = 4;
constexpr int IDX_BETA2_POWER = 5;
constexpr int IDX_LR = 6;
constexpr int IDX_BETA1 = 7;
constexpr int IDX_BETA2 = 8;
constexpr int IDX_EPSILON = 9;
constexpr int IDX_GRAD = 10;

static const gert::Shape g_scalar_shape = {1};

static inline gert::Shape EnsureNotScalar(const gert::Shape& in_shape)
{
    if (in_shape.GetDimNum() == 0) {
        return g_scalar_shape;
    }
    return in_shape;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t* ubSize, int64_t* coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    *coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(*coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, *ubSize);
    OP_CHECK_IF(*ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context, int64_t* totalNum, ge::DataType* dataType)
{
    auto varShapeDesc = context->GetInputShape(IDX_VAR);
    OP_CHECK_NULL_WITH_CONTEXT(context, varShapeDesc);
    auto varShape = EnsureNotScalar(varShapeDesc->GetStorageShape());

    auto mShapeDesc = context->GetInputShape(IDX_M);
    OP_CHECK_NULL_WITH_CONTEXT(context, mShapeDesc);
    auto mShape = EnsureNotScalar(mShapeDesc->GetStorageShape());

    auto vShapeDesc = context->GetInputShape(IDX_V);
    OP_CHECK_NULL_WITH_CONTEXT(context, vShapeDesc);
    auto vShape = EnsureNotScalar(vShapeDesc->GetStorageShape());

    auto vhatShapeDesc = context->GetInputShape(IDX_VHAT);
    OP_CHECK_NULL_WITH_CONTEXT(context, vhatShapeDesc);
    auto vhatShape = EnsureNotScalar(vhatShapeDesc->GetStorageShape());

    auto gradShapeDesc = context->GetInputShape(IDX_GRAD);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradShapeDesc);
    auto gradShape = EnsureNotScalar(gradShapeDesc->GetStorageShape());

    OP_CHECK_IF(
        varShape.GetShapeSize() != mShape.GetShapeSize() ||
            varShape.GetShapeSize() != vShape.GetShapeSize() ||
            varShape.GetShapeSize() != vhatShape.GetShapeSize() ||
            varShape.GetShapeSize() != gradShape.GetShapeSize(),
        OP_LOGE(context,
                "ApplyAdamWithAmsgradV2: var/m/v/vhat/grad shape size mismatch: "
                "var=%ld, m=%ld, v=%ld, vhat=%ld, grad=%ld",
                varShape.GetShapeSize(), mShape.GetShapeSize(), vShape.GetShapeSize(),
                vhatShape.GetShapeSize(), gradShape.GetShapeSize()),
        return ge::GRAPH_FAILED);

    *totalNum = varShape.GetShapeSize();

    auto inputDesc = context->GetInputDesc(IDX_VAR);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    *dataType = inputDesc->GetDataType();
    // 仅支持 fp32
    OP_CHECK_IF(*dataType != ge::DT_FLOAT,
                OP_LOGE(context,
                        "ApplyAdamWithAmsgradV2: only float32 supported, var dtype=%d.",
                        static_cast<int>(*dataType)),
                return ge::GRAPH_FAILED);

    // m/v/vhat/grad 必须与 var 同 dtype：kernel 统一 DTYPE_X 单模板，异构 dtype 会按错误字宽错读字节
    const int consistencyIdx[] = {IDX_M, IDX_V, IDX_VHAT, IDX_GRAD};
    for (int idx : consistencyIdx) {
        auto desc = context->GetInputDesc(idx);
        OP_CHECK_NULL_WITH_CONTEXT(context, desc);
        OP_CHECK_IF(desc->GetDataType() != *dataType,
                    OP_LOGE(context,
                            "ApplyAdamWithAmsgradV2: input[%d] dtype=%d must equal var dtype=%d.",
                            idx, static_cast<int>(desc->GetDataType()), static_cast<int>(*dataType)),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(WORKSPACE_NUM);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

// 标量直读为 float。契约层面：proto/def 已将 6 个标量锁定为 DT_FLOAT，
// 算子整体收窄为 fp32-only，标量恒为 float，直接按 float 读取。
static float ReadScalarTensor(gert::TilingContext* context, int inputIdx)
{
    const gert::Tensor* tensor = context->GetInputTensor(inputIdx);
    if (tensor == nullptr) {
        return 0.0f;
    }
    const float* data = tensor->GetData<float>();
    if (data == nullptr) {
        return 0.0f;
    }
    return data[0];
}

// 标量读取：6 个标量按各自 dtype 解码写入 tiling
static void ReadScalars(gert::TilingContext* context, ApplyAdamWithAmsgradV2TilingData* tiling)
{
    tiling->beta1Power = ReadScalarTensor(context, IDX_BETA1_POWER);
    tiling->beta2Power = ReadScalarTensor(context, IDX_BETA2_POWER);
    tiling->lr = ReadScalarTensor(context, IDX_LR);
    tiling->beta1 = ReadScalarTensor(context, IDX_BETA1);
    tiling->beta2 = ReadScalarTensor(context, IDX_BETA2);
    tiling->epsilon = ReadScalarTensor(context, IDX_EPSILON);
}

// 切分计算：blockFactor（每核元素数）+ ubFactor（每轮 UB 元素数），写回 tiling，返回 usedCoreNum
static int64_t CalcBlockTiling(gert::TilingContext* context, ApplyAdamWithAmsgradV2TilingData* tiling,
                               int64_t totalNum, int64_t coreNum, uint64_t ubSize)
{
    tiling->totalNum = totalNum;
    if (totalNum == 0) {  // 空 tensor
        tiling->blockFactor = 0;
        tiling->ubFactor = 0;
        return 1;
    }

    // 算子已收窄为 fp32-only：typeSize 恒为 4，bytesPerElem 恒为 FP32_BYTES_PER_ELEM
    int64_t typeSize = 4;
    int64_t bytesPerElem = FP32_BYTES_PER_ELEM;
    int64_t ubBlockSize = Ops::Base::GetUbBlockSize(context);  // 32 字节
    int64_t alignElems = ubBlockSize / typeSize;  // 对齐元素数（fp32=8, 半=16），与 ubFactor 一致

    tiling->blockFactor = CeilAlign(CeilDiv(totalNum, coreNum), alignElems);
    int64_t usedCoreNum = Ops::Base::CeilDiv(totalNum, tiling->blockFactor);

    int64_t maxElemsByUb = Ops::Base::FloorDiv(static_cast<int64_t>(ubSize), bytesPerElem);
    int64_t ubFactor = Ops::Base::FloorAlign(maxElemsByUb, alignElems);
    if (ubFactor <= 0) {
        ubFactor = alignElems;
    }
    if (ubFactor > tiling->blockFactor) {
        ubFactor = tiling->blockFactor;
    }
    tiling->ubFactor = ubFactor;
    return usedCoreNum;
}

static ge::graphStatus ApplyAdamWithAmsgradV2TilingFunc(gert::TilingContext* context)
{
    OP_LOGI(context, "Enter ApplyAdamWithAmsgradV2TilingFunc");
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(GetPlatformInfo(context, &ubSize, &coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    int64_t totalNum = 0;
    ge::DataType dataType = ge::DT_FLOAT;
    OP_CHECK_IF(GetShapeAttrsInfo(context, &totalNum, &dataType) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetWorkspaceSize error"), return ge::GRAPH_FAILED);

    ApplyAdamWithAmsgradV2TilingData* tiling = context->GetTilingData<ApplyAdamWithAmsgradV2TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(ApplyAdamWithAmsgradV2TilingData), 0,
                         sizeof(ApplyAdamWithAmsgradV2TilingData)) != EOK,
                OP_LOGE(context, "memset tiling data error"), return ge::GRAPH_FAILED);

    ReadScalars(context, tiling);  // 标量读取（ValueDepend(REQUIRED, TILING) 保证 host 可读）
    int64_t usedCoreNum = CalcBlockTiling(context, tiling, totalNum, coreNum, ubSize);

    context->SetBlockDim(usedCoreNum);
    // def 驱动 dtype：dtype 由构建系统按 op_def DataType 列表展开为 DTYPE_VAR（现仅 float 单变体），
    // tiling key 无需再编码 dtype，仅设置单值占位轴 schMode=0。
    // dataType 仍在 GetShapeAttrsInfo 中用于 fp32 校验。
    ASCENDC_TPL_SEL_PARAM(context, APPLY_ADAM_WITH_AMSGRAD_V2_TPL_SCH_MODE_0);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForApplyAdamWithAmsgradV2([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ApplyAdamWithAmsgradV2CompileInfo {};  // 入图场景依赖

IMPL_OP_OPTILING(ApplyAdamWithAmsgradV2)
    .Tiling(ApplyAdamWithAmsgradV2TilingFunc)
    .TilingParse<ApplyAdamWithAmsgradV2CompileInfo>(TilingParseForApplyAdamWithAmsgradV2);

} // namespace optiling
