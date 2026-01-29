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
 * \file conv2d_mdc_tiling.cpp
 * \brief
 */

#include "arch35/conv2d_v2_base_tiling.h"

#include "tiling_base/tiling_templates_registry.h"
#include "register/op_def_registry.h"
#include "common/op_host/op_tiling/conv_tiling_templates_registry.h"

using namespace optiling::conv_ops_tiling;

namespace optiling {
    // using op_tiling register capability in "tiling_templates_registry" for AscendC conv2d operator
    CONV_REGISTER_TILING_TEMPLATE(Conv2DV2, Conv2dBaseTiling, static_cast<int32_t>(NpuArch::DAV_5102), 1);
}