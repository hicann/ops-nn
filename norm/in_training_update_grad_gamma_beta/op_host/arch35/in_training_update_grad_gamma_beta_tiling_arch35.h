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
 * \file in_training_update_grad_gamma_beta_tiling_arch35.h
 * \brief Host tiling declarations for Ascend 950.
 */

#ifndef IN_TRAINING_UPDATE_GRAD_GAMMA_BETA_TILING_ARCH35_H_
#define IN_TRAINING_UPDATE_GRAD_GAMMA_BETA_TILING_ARCH35_H_

#include "exe_graph/runtime/tiling_context.h"
#include "graph/types.h"

namespace optiling {
struct INTrainingUpdateGradGammaBetaCompileInfo {};

ge::graphStatus INTrainingUpdateGradGammaBetaTilingFunc(gert::TilingContext* context);
} // namespace optiling

#endif // IN_TRAINING_UPDATE_GRAD_GAMMA_BETA_TILING_ARCH35_H_
