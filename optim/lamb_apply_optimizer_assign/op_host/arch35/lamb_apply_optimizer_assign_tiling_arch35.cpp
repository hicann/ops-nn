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
 * \file lamb_apply_optimizer_assign_tiling_arch35.cpp
 * \brief lamb_apply_optimizer_assign_tiling_arch35 source file
 */

#include "lamb_apply_optimizer_assign_tiling_arch35.h"
#include "../../../lamb_apply_common/op_host/arch35/lamb_apply_check_util.h"
#include <graph/utils/type_utils.h>
#include <securec.h>
#include <algorithm>
#include <string>
#include "infershape_broadcast_util.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_templates_registry.h"

using namespace ge;

namespace optiling {

constexpr static uint64_t LAMB_APPLY_OPTIMIZER_ASSIGN_TILING_PRIORITY = 0;
constexpr static uint64_t TILING_KEY_FP32 = 100;
constexpr static uint64_t TILING_KEY_FP16 = 200;
constexpr static int32_t INPUT_NUM = 12;
constexpr static int32_t OUTPUT_NUM = 3;
constexpr static int32_t INPUTV_IDX = 1; // inputv: ref(原地)输出
constexpr static int32_t INPUTM_IDX = 2; // inputm: ref(原地)输出
static const char* const kInputNames[] = {"grad",   "inputv", "inputm", "input3", "mul0_x",        "mul1_x",
                                          "mul2_x", "mul3_x", "add2_y", "steps",  "do_use_weight", "weight_decay_rate"};
static const char* const kOutputNames[] = {"output0", "inputv", "inputm"};

static ge::graphStatus TilingPrepareForLambApplyOptimizerAssign(gert::TilingParseContext* context)
{
    auto compileInfoPtr = context->GetCompiledInfo<LambApplyOptimizerAssignCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LambApplyOptimizerAssignTiling::GetShapeAttrsInfo()
{
    if (CheckLambApplyDtypeConsistency(context_, INPUT_NUM, kInputNames, OUTPUT_NUM, kOutputNames) !=
        ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return CheckInplaceShapeConstraint();
}

// inputv、inputm 是 in-place 更新的动量输出(next_v/next_m 原地写回它们的输入 buffer,见 proto "(in-place)"),
// 两者形状必须相同, 且必须 == 全部输入广播的完整网格。其余输入(含绑在 In0 的 grad)可广播进这个网格。
ge::graphStatus LambApplyOptimizerAssignTiling::CheckInplaceShapeConstraint()
{
    auto inputvShape = context_->GetInputShape(INPUTV_IDX);
    auto inputmShape = context_->GetInputShape(INPUTM_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputvShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputmShape);
    const auto& vs = inputvShape->GetStorageShape();
    const auto& ms = inputmShape->GetStorageShape();
    if (!(vs == ms)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context_->GetNodeName(), "inputv and inputm",
            (Ops::Base::ToString(vs) + " and " + Ops::Base::ToString(ms)).c_str(),
            "inputv and inputm are in-place updated moments and must have the same shape");
        return ge::GRAPH_FAILED;
    }
    return CheckLambApplyBroadcastIntoRef(context_, INPUT_NUM, INPUTV_IDX, "inputv");
}

bool LambApplyOptimizerAssignTiling::IsCapable() { return true; }

ge::graphStatus LambApplyOptimizerAssignTiling::DoOpTiling()
{
    auto rawTilingData = context_->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context_, rawTilingData);
    auto input0Desc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, input0Desc);

    ge::DataType input0DType = input0Desc->GetDataType();
    uint32_t dtSize = 0;
    if (input0DType == ge::DT_FLOAT) {
        tilingKey = TILING_KEY_FP32;
        dtSize = sizeof(float);
    } else if (input0DType == ge::DT_FLOAT16) {
        tilingKey = TILING_KEY_FP16;
        dtSize = sizeof(uint16_t);
    } else {
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "grad", Ops::Base::ToString(input0DType).c_str(),
                                  "fp16 or fp32");
        return ge::GRAPH_FAILED;
    }

    using PlanTiling = LambBrcTilingData<12, 3>;
    td_ = PlanTiling{};
    // 空进空出: 输出 0 元素时 tiling 全 0, kernel 按 usedCoreNum=0 直接退出。
    auto outShape0 = context_->GetOutputShape(0);
    if (outShape0 == nullptr || outShape0->GetStorageShape().GetShapeSize() != 0) {
        // 先取返回值再判: 模板实参里的逗号会被预处理器当成 OP_CHECK_IF 的参数分隔符。
        ge::graphStatus planRet = BuildLambBrcPlan<12, 3>(context_, coreNum_, ubSize_, dtSize, td_);
        OP_CHECK_IF(planRet != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "build broadcast plan failed"),
                    return ge::GRAPH_FAILED);
    }

    auto ret = memcpy_s(rawTilingData->GetData(), rawTilingData->GetCapacity(), &td_, sizeof(td_));
    OP_CHECK_IF(ret != EOK, OP_LOGE(context_->GetNodeName(), "copy tiling data failed, ret %d", ret),
                return ge::GRAPH_FAILED);
    rawTilingData->SetDataSize(sizeof(td_));
    context_->SetBlockDim(std::max<uint32_t>(td_.usedCoreNum, 1));
    size_t* ws = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, ws);
    ws[0] = 0U;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LambApplyOptimizerAssignTiling::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

uint64_t LambApplyOptimizerAssignTiling::GetTilingKey() const { return tilingKey; }

ge::graphStatus LambApplyOptimizerAssignTiling::GetWorkspaceSize() { return ge::GRAPH_SUCCESS; }

ge::graphStatus LambApplyOptimizerAssignTiling::PostTiling() { return ge::GRAPH_SUCCESS; }

ge::graphStatus LambApplyOptimizerAssignTiling::GetPlatformInfo()
{
    auto compileInfo = static_cast<const LambApplyOptimizerAssignCompileInfo*>(context_->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);
    coreNum_ = compileInfo->coreNum;
    ubSize_ = compileInfo->ubSize;
    OP_CHECK_IF(coreNum_ == 0 || ubSize_ == 0,
                OP_LOGE(context_->GetNodeName(), "invalid platform info: coreNum %lu ubSize %lu", coreNum_, ubSize_),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingForLambApplyOptimizerAssign(gert::TilingContext* context)
{
    OP_LOGD("LambApplyOptimizerAssignTiling", "Enter TilingForLambApplyOptimizerAssign");
    if (context == nullptr) {
        OP_LOGE("LambApplyOptimizerAssignTiling", "Tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }

    auto compileInfo = static_cast<const LambApplyOptimizerAssignCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    OP_LOGD(context, "Enter ascendc LambApplyOptimizerAssignTiling");
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

IMPL_OP_OPTILING(LambApplyOptimizerAssign)
    .Tiling(TilingForLambApplyOptimizerAssign)
    .TilingParse<LambApplyOptimizerAssignCompileInfo>(TilingPrepareForLambApplyOptimizerAssign);

REGISTER_OPS_TILING_TEMPLATE(LambApplyOptimizerAssign, LambApplyOptimizerAssignTiling,
                             LAMB_APPLY_OPTIMIZER_ASSIGN_TILING_PRIORITY);
} // namespace optiling
