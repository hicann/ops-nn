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
 * \file sigmoid_cross_entropy_with_logits_grad_v2_arch35_tiling.h
 * \brief
 */
#ifndef OP_TILING_SIGMOIDCROSSENTROPYWITHLOGITSGRADV2_ARCH35_H_
#define OP_TILING_SIGMOIDCROSSENTROPYWITHLOGITSGRADV2_ARCH35_H_

#include "atvoss/broadcast/broadcast_tiling.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"
#include "register/tilingdata_base.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "op_host/tiling_templates_registry.h"
#include "register/op_impl_registry.h"
#include "../../op_kernel/arch35/sigmoid_cross_entropy_with_logits_grad_v2_dag.h"
#include "../../op_kernel/arch35/sigmoid_cross_entropy_with_logits_grad_v2_struct.h"
#include "register/op_def_registry.h"
#include "op_api/runtime2_util.h"

namespace optiling {
using Ops::NN::Optiling::TilingBaseClass;
using namespace Ops::Base;

struct SigmoidCrossEntropyWithLogitsGradV2CompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSize = 0;
};

class SigmoidCrossEntropyWithLogitsGradV2Tiling : public TilingBaseClass {
public:
    explicit SigmoidCrossEntropyWithLogitsGradV2Tiling(gert::TilingContext* context) : TilingBaseClass(context) {}

protected:
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;

private:
    uint64_t tilingKey_ = 0;
    ge::graphStatus GetDtype();
    ge::graphStatus GetScale();
    void DumpTilingInfo();
    platform_ascendc::SocVersion socVersion_ = platform_ascendc::SocVersion::ASCEND910B;
    bool hasWeight_ = false;
    bool hasPosWeight_ = false;
    bool isMean_ = false;
    float scale_ = 0;
    ge::DataType dtype_ = ge::DT_FLOAT;
};
} // namespace optiling
#endif // OP_TILING_SIGMOIDCROSSENTROPYWITHLOGITSGRADV2_ARCH35_H_
