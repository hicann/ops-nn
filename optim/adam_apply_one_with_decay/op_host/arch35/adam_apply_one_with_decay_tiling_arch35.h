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
 * \file adam_apply_one_with_decay_tiling_arch35.h
 * \brief adam_apply_one_with_decay_tiling_arch35 head file
 */

#ifndef OPS_OPTIM_ADAM_APPLY_ONE_WITH_DECAY_OP_HOST_ADAM_APPLY_ONE_WITH_DEACY_TILING_ARCH35_H
#define OPS_OPTIM_ADAM_APPLY_ONE_WITH_DECAY_OP_HOST_ADAM_APPLY_ONE_WITH_DEACY_TILING_ARCH35_H

#include "../../op_kernel/arch35/adam_apply_one_with_decay_tiling_key.h"
#include "tiling_base/tiling_base.h"

using namespace Ops::NN::Optiling;

namespace optiling {

struct AdamApplyOneWithDecayCompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSize = 0;
};

class AdamApplyOneWithDecayTiling : public TilingBaseClass {
public:
    explicit AdamApplyOneWithDecayTiling(gert::TilingContext* context) : TilingBaseClass(context)
    {}

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
    uint64_t _tilingKey = 0;
};

} // namespace optiling

#endif // OPS_OPTIM_ADAM_APPLY_ONE_WITH_DECAY_OP_HOST_ADAM_APPLY_ONE_WITH_DEACY_TILING_ARCH35_H