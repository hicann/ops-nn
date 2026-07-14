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
 * \file lamb_apply_weight_assign_tiling_arch35.h
 * \brief lamb_apply_weight_assign_tiling_arch35 head file
 */

#ifndef OPS_OPTIM_ADAM_APPLY_ONE_OP_HOST_ADAM_APPLY_ONE_TILING_ARCH35_H
#define OPS_OPTIM_ADAM_APPLY_ONE_OP_HOST_ADAM_APPLY_ONE_TILING_ARCH35_H

#include "../../op_kernel/arch35/lamb_apply_weight_assign_tiling_key.h"
#include "op_host/tiling_base.h"

using namespace Ops::NN::Optiling;

namespace optiling {

struct LambApplyWeightAssignCompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSize = 0;
};

class LambApplyWeightAssignTiling : public TilingBaseClass {
public:
    explicit LambApplyWeightAssignTiling(gert::TilingContext* context) : TilingBaseClass(context) {}

protected:
    bool IsCapable() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

private:
    // 校验 in-place 更新的 input_param 形状 == broadcast(input3, input_param)(input3 可向上广播)。
    // dtype 一致性、标量非空的通用校验见 lamb_apply_common/lamb_apply_check_util.h。
    ge::graphStatus CheckInplaceShapeConstraint();
    uint64_t tilingKey = 0;
};

} // namespace optiling

#endif // OPS_OPTIM_ADAM_APPLY_ONE_OP_HOST_ADAM_APPLY_ONE_TILING_ARCH35_H
