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
#include "../../../lamb_apply_common/lamb_apply_check_util.h"
#include <graph/utils/type_utils.h>
#include <string>
#include "infershape_broadcast_util.h"
#include "../../op_kernel/arch35/lamb_apply_optimizer_assign_dag.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_templates_registry.h"

using namespace AscendC;
using namespace ge;

namespace optiling {

constexpr static uint64_t LAMB_APPLY_OPTIMIZER_ASSIGN_TILING_PRIORITY = 0;
constexpr static int32_t INPUT_NUM = 12;
constexpr static int32_t OUTPUT_NUM = 3;
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
    static const int32_t kScalarInputIdx[] = {4, 5, 6, 7, 8, 9, 10, 11};
    if (CheckLambApplyDtypeConsistency(context_, INPUT_NUM, kInputNames, OUTPUT_NUM, kOutputNames) !=
        ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckLambApplyScalarNotEmpty(context_, kScalarInputIdx, sizeof(kScalarInputIdx) / sizeof(kScalarInputIdx[0]),
                                     kInputNames) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return CheckInplaceShapeConstraint();
}

// inputv、inputm 是 in-place 更新的动量输出(next_v/next_m 原地写回它们的输入 buffer,见 proto "(in-place)"),
// 内核按 grad/inputv/inputm/input3 广播出的最大网格计算并写回,故 inputv、inputm 形状必须 == 该广播网格。
// 等价充要条件:inputv 与 inputm 同形状,且 grad、input3 均能广播进 inputv(非 in-place 与标量可向上广播)。
ge::graphStatus LambApplyOptimizerAssignTiling::CheckInplaceShapeConstraint()
{
    auto gradShape = context_->GetInputShape(0);
    auto inputvShape = context_->GetInputShape(1);
    auto inputmShape = context_->GetInputShape(2);
    auto input3Shape = context_->GetInputShape(3);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputvShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputmShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, input3Shape);
    const auto& gs = gradShape->GetStorageShape();
    const auto& vs = inputvShape->GetStorageShape();
    const auto& ms = inputmShape->GetStorageShape();
    const auto& ps = input3Shape->GetStorageShape();
    gert::Shape bcShape;
    if (!(vs == ms)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context_->GetNodeName(), "inputv and inputm",
            (Ops::Base::ToString(vs) + " and " + Ops::Base::ToString(ms)).c_str(),
            "inputv and inputm are in-place updated moments and must have the same shape equal to the broadcast "
            "output shape");
        return ge::GRAPH_FAILED;
    }
    // grad/input3 能广播进 inputv <=> broadcast(x, inputv) == inputv
    if (!Ops::Base::BroadcastShape(&gs, &vs, &bcShape) || !(bcShape == vs)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            context_->GetNodeName(), "grad", Ops::Base::ToString(gs).c_str(),
            "grad must be broadcastable into the in-place moment shape inputv/inputm");
        return ge::GRAPH_FAILED;
    }
    if (!Ops::Base::BroadcastShape(&ps, &vs, &bcShape) || !(bcShape == vs)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            context_->GetNodeName(), "input3", Ops::Base::ToString(ps).c_str(),
            "input3 must be broadcastable into the in-place moment shape inputv/inputm");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

bool LambApplyOptimizerAssignTiling::IsCapable() { return true; }

ge::graphStatus LambApplyOptimizerAssignTiling::DoOpTiling()
{
    // 空 tensor 应对(空进空出): 输出为空(0元素)时设 1 核(空转), 配合全0 tiling 数据(blockFormer=0)使 kernel 空转退出,
    // 直接成功。
    auto emptyTensorOutShape0 = context_->GetOutputShape(0);
    if (emptyTensorOutShape0 != nullptr && emptyTensorOutShape0->GetStorageShape().GetShapeSize() == 0) {
        auto emptyRawTiling = context_->GetRawTilingData();
        if (emptyRawTiling != nullptr && emptyRawTiling->GetData() != nullptr) {
            size_t emptyCap = emptyRawTiling->GetCapacity();
            uint8_t* emptyPtr = reinterpret_cast<uint8_t*>(emptyRawTiling->GetData());
            for (size_t emptyIdx = 0; emptyIdx < emptyCap; ++emptyIdx) {
                emptyPtr[emptyIdx] = 0;
            }
            emptyRawTiling->SetDataSize(emptyCap);
        }
        size_t* emptyWs = context_->GetWorkspaceSizes(1);
        if (emptyWs != nullptr) {
            emptyWs[0] = 0;
        }
        context_->SetBlockDim(1);
        tilingKey = GET_TPL_TILING_KEY(1); // schMode=1(已编译), 配合全0 tiling(blockFormer=0)空转
        return ge::GRAPH_SUCCESS;
    }
    auto input0Desc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, input0Desc);
    ge::DataType input0DType = input0Desc->GetDataType();
    if (input0DType == ge::DT_FLOAT16) {
        BroadcastBaseTiling<LambApplyOptimizerAssignOp::LambApplyOptimizerAssignCompute<half, float>::OpDag>
            brcBaseTiling(context_, static_cast<uint32_t>(BROADCAST_KERNEL_TYPE::KERNEL_TYPE_NDDMA));
        OP_CHECK_IF(brcBaseTiling.DoTiling() == ge::GRAPH_FAILED,
                    OP_LOGE(context_->GetNodeName(), "Do tiling failed. Please check the detailed log."),
                    return ge::GRAPH_FAILED);
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0DType == ge::DT_FLOAT) {
        BroadcastBaseTiling<LambApplyOptimizerAssignOp::LambApplyOptimizerAssignCompute<float, float>::OpDag>
            brcBaseTiling(context_, static_cast<uint32_t>(BROADCAST_KERNEL_TYPE::KERNEL_TYPE_NDDMA));
        OP_CHECK_IF(brcBaseTiling.DoTiling() == ge::GRAPH_FAILED,
                    OP_LOGE(context_->GetNodeName(), "Do tiling failed. Please check the detailed log."),
                    return ge::GRAPH_FAILED);
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else {
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "grad", Ops::Base::ToString(input0DType).c_str(),
                                  "fp16 or fp32");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LambApplyOptimizerAssignTiling::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

uint64_t LambApplyOptimizerAssignTiling::GetTilingKey() const { return tilingKey; }

ge::graphStatus LambApplyOptimizerAssignTiling::GetWorkspaceSize() { return ge::GRAPH_SUCCESS; }

ge::graphStatus LambApplyOptimizerAssignTiling::PostTiling() { return ge::GRAPH_SUCCESS; }

ge::graphStatus LambApplyOptimizerAssignTiling::GetPlatformInfo() { return ge::GRAPH_SUCCESS; }

static ge::graphStatus TilingForLambApplyOptimizerAssign(gert::TilingContext* context)
{
    OP_LOGD("LambApplyOptimizerAssignTiling", "Enter TilingForLambApplyOptimizerAssign");
    if (context == nullptr) {
        OP_LOGE("LambApplyOptimizerAssignTiling", "Tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }

    auto compileInfo = reinterpret_cast<const LambApplyOptimizerAssignCompileInfo*>(context->GetCompileInfo());
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
