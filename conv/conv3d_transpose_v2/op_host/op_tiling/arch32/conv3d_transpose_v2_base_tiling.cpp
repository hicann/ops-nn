/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file conv3d_transpose_v2_base_tiling.cpp
 * \brief
 */

#include <log/log.h>
#include "tiling_base/tiling_templates_registry.h"
#include "conv3d_transpose_v2_base_tiling.h"

namespace Ops {
namespace NN {
namespace Conv {

REGISTER_TILING_TEMPLATE("Conv3DTransposeV2", Conv3DTransposeV2Tiling, 0);

} // namespace Conv
} // namespace NN
} // namespace Ops
