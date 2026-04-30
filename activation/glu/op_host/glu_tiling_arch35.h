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
 * \file glu_tiling_arch35.h
 * \brief Tiling implementation for Ascend950 (arch35)
 */
#ifndef GLU_TILING_ARCH35_H
#define GLU_TILING_ARCH35_H

#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/math_util.h"
#include "glu_tiling.h"

namespace optiling {

class GluRegbaseTiling
{
public:
    GluRegbaseTiling() = default;
    ~GluRegbaseTiling() = default;
    GluTilingData tilingData;
    ge::graphStatus RunFusionKernelTiling(gert::TilingContext* context);
};

} // namespace optiling

#endif // GLU_TILING_ARCH35_H
