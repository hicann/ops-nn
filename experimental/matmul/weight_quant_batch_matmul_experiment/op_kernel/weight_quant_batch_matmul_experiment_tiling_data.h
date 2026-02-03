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
 * \file weight_quant_batch_matmul_experiment_tiling_data.h
 * \brief tiling data struct
 */

#ifndef __WEIGHT_QUANT_BATCH_MATMUL_EXPERIMENT_TILLING_DATA_H__
#define __WEIGHT_QUANT_BATCH_MATMUL_EXPERIMENT_TILLING_DATA_H__

#include "tiling/tiling_api.h"

namespace WeightQuantBatchMatmulExperimental {
#pragma pack(push, 8)
struct WeightQuantBatchMatmulExperimentTilingData {
    TCubeTiling matmulTiling;
};
#pragma pack(pop)
}  // namespace WeightQuantBatchMatmulExperimental
#endif
