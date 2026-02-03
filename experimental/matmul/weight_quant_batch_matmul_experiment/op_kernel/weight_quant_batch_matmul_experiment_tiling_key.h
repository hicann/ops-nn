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
 * \file weight_quant_batch_matmul_experiment_tiling_key.h
 * \brief weight_quant_batch_matmul_experiment tiling key declare
 */

#ifndef __WEIGHT_QUANT_BATCH_MATMUL_EXPERIMENT_TILING_KEY_H__
#define __WEIGHT_QUANT_BATCH_MATMUL_EXPERIMENT_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

#define BASIC_MSD 0
#define PRELOAD_MSD 1

ASCENDC_TPL_ARGS_DECL(WeightQuantBatchMatmulExperiment,
                      ASCENDC_TPL_UINT_DECL(MSD_MODE, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, BASIC_MSD, PRELOAD_MSD));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2),
                                     ASCENDC_TPL_UINT_SEL(MSD_MODE, ASCENDC_TPL_UI_LIST, BASIC_MSD, PRELOAD_MSD), ));
#endif