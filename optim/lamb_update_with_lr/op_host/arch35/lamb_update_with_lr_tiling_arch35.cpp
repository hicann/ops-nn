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
 * \file lamb_update_with_lr_tiling_arch35.cpp
 * \brief lamb_update_with_lr_tiling_arch35 source file
 */

#include "lamb_update_with_lr_tiling_arch35.h"
#include <graph/utils/type_utils.h>
#include "../../op_kernel/arch35/lamb_update_with_lr_dag.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_templates_registry.h"

using namespace AscendC;
using namespace ge;

namespace optiling {

constexpr static uint64_t LAMB_UPDATE_WITH_LR_TILING_PRIORITY = 0;
constexpr static int32_t INPUT_NUM = 9;
constexpr static int32_t OUTPUT_NUM = 1;

static ge::graphStatus TilingPrepareForLambUpdateWithLr(gert::TilingParseContext* context)
{
    auto compileInfoPtr = context->GetCompiledInfo<LambUpdateWithLrCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LambUpdateWithLrTiling::GetShapeAttrsInfo()
{
    auto input0Desc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, input0Desc);
    ge::DataType input0DType = input0Desc->GetDataType();
    static const char* kInputNames[] = {"input_greater1", "input_greater_realdiv",
                                        "input_realdiv",  "input_mul0",
                                        "input_mul1",     "input_sub",
                                        "greater_y",      "select_e",
                                        "minimum_y"};
    static const char* kOutputNames[] = {"y"};
    for (int32_t inputIdx = 1; inputIdx < INPUT_NUM; inputIdx++) {
        auto inputDesc = context_->GetInputDesc(inputIdx);
        OP_CHECK_NULL_WITH_CONTEXT(context_, inputDesc);

        auto curDtype = inputDesc->GetDataType();
        if (curDtype != input0DType) {
            std::string paramNames = std::string(kInputNames[inputIdx]) + " and input0";
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                context_->GetNodeName(), paramNames.c_str(),
                (Ops::Base::ToString(curDtype) + " and " + Ops::Base::ToString(input0DType)).c_str(),
                "Their dtypes should be the same");
            return ge::GRAPH_FAILED;
        }
    }
    for (int32_t outputIdx = 0; outputIdx < OUTPUT_NUM; outputIdx++) {
        auto outputDesc = context_->GetOutputDesc(outputIdx);
        OP_CHECK_NULL_WITH_CONTEXT(context_, outputDesc);

        auto curDtype = outputDesc->GetDataType();
        if (curDtype != input0DType) {
            std::string paramNames = std::string(kOutputNames[outputIdx]) + " and input0";
            std::string incorrectDtypes = Ops::Base::ToString(curDtype) + " and " + Ops::Base::ToString(input0DType);
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), paramNames.c_str(), incorrectDtypes.c_str(),
                                                   "Their dtypes should be the same");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

bool LambUpdateWithLrTiling::IsCapable() { return true; }

ge::graphStatus LambUpdateWithLrTiling::DoOpTiling()
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
        BroadcastBaseTiling<LambUpdateWithLrOp::LambUpdateWithLrCompute<half, float>::OpDag> brcBaseTiling(
            context_, static_cast<uint32_t>(BROADCAST_KERNEL_TYPE::KERNEL_TYPE_NDDMA));
        OP_CHECK_IF(brcBaseTiling.DoTiling() == ge::GRAPH_FAILED,
                    OP_LOGE(context_->GetNodeName(), "Do tiling failed. Please check the detailed log."),
                    return ge::GRAPH_FAILED);
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0DType == ge::DT_FLOAT) {
        BroadcastBaseTiling<LambUpdateWithLrOp::LambUpdateWithLrCompute<float, float>::OpDag> brcBaseTiling(
            context_, static_cast<uint32_t>(BROADCAST_KERNEL_TYPE::KERNEL_TYPE_NDDMA));
        OP_CHECK_IF(brcBaseTiling.DoTiling() == ge::GRAPH_FAILED,
                    OP_LOGE(context_->GetNodeName(), "Do tiling failed. Please check the detailed log."),
                    return ge::GRAPH_FAILED);
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else {
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "input_greater1", Ops::Base::ToString(input0DType).c_str(),
                                  "fp16 or fp32");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LambUpdateWithLrTiling::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

uint64_t LambUpdateWithLrTiling::GetTilingKey() const { return tilingKey; }

ge::graphStatus LambUpdateWithLrTiling::GetWorkspaceSize() { return ge::GRAPH_SUCCESS; }

ge::graphStatus LambUpdateWithLrTiling::PostTiling() { return ge::GRAPH_SUCCESS; }

ge::graphStatus LambUpdateWithLrTiling::GetPlatformInfo() { return ge::GRAPH_SUCCESS; }

static ge::graphStatus TilingForLambUpdateWithLr(gert::TilingContext* context)
{
    OP_LOGD("LambUpdateWithLrTiling", "Enter TilingForLambUpdateWithLr");
    if (context == nullptr) {
        OP_LOGE("LambUpdateWithLrTiling", "Tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }

    auto compileInfo = reinterpret_cast<const LambUpdateWithLrCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    OP_LOGD(context, "Enter ascendc LambUpdateWithLrTiling");
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

IMPL_OP_OPTILING(LambUpdateWithLr)
    .Tiling(TilingForLambUpdateWithLr)
    .TilingParse<LambUpdateWithLrCompileInfo>(TilingPrepareForLambUpdateWithLr);

REGISTER_OPS_TILING_TEMPLATE(LambUpdateWithLr, LambUpdateWithLrTiling, LAMB_UPDATE_WITH_LR_TILING_PRIORITY);
} // namespace optiling
