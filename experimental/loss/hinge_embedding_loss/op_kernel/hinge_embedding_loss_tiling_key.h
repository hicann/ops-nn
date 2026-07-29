/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef HINGE_EMBEDDING_LOSS_TILING_KEY_H_
#define HINGE_EMBEDDING_LOSS_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h"

#define HINGE_EMBEDDING_LOSS_REDUCTION_NONE 0
#define HINGE_EMBEDDING_LOSS_REDUCTION_SUM 1
#define HINGE_EMBEDDING_LOSS_REDUCTION_MEAN 2

ASCENDC_TPL_ARGS_DECL(HingeEmbeddingLoss,
                      ASCENDC_TPL_UINT_DECL(reductionMode, 2, ASCENDC_TPL_UI_LIST, HINGE_EMBEDDING_LOSS_REDUCTION_NONE,
                                            HINGE_EMBEDDING_LOSS_REDUCTION_SUM, HINGE_EMBEDDING_LOSS_REDUCTION_MEAN));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(reductionMode, ASCENDC_TPL_UI_LIST,
                                                          HINGE_EMBEDDING_LOSS_REDUCTION_NONE,
                                                          HINGE_EMBEDDING_LOSS_REDUCTION_SUM,
                                                          HINGE_EMBEDDING_LOSS_REDUCTION_MEAN)), );

#endif
