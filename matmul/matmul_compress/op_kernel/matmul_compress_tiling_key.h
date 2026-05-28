/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file matmul_compress_tiling_key.h
 * \brief
 */

#ifndef MATMUL_COMPRESS_TILING_KEY_H
#define MATMUL_COMPRESS_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

#define MATMUL_COMPRESS_PP_MAT_MUL_MODE_FALSE 0
#define MATMUL_COMPRESS_PP_MAT_MUL_MODE_TRUE 1

#define MATMUL_COMPRESS_NOT_TRANS 0
#define MATMUL_COMPRESS_A_TRANS 1
#define MATMUL_COMPRESS_B_TRANS 2
#define MATMUL_COMPRESS_ALL_TRANS 3

ASCENDC_TPL_ARGS_DECL(
    MatmulCompress,
    ASCENDC_TPL_UINT_DECL(
        PP_MAT_MUL_MODE, ASCENDC_TPL_1_BW, ASCENDC_TPL_UI_LIST, 
        MATMUL_COMPRESS_PP_MAT_MUL_MODE_FALSE,
        MATMUL_COMPRESS_PP_MAT_MUL_MODE_TRUE),
    ASCENDC_TPL_UINT_DECL(
        TRANS, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, 
        MATMUL_COMPRESS_NOT_TRANS, MATMUL_COMPRESS_A_TRANS,
        MATMUL_COMPRESS_B_TRANS, MATMUL_COMPRESS_ALL_TRANS), );


ASCENDC_TPL_SEL(
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2002)
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(PP_MAT_MUL_MODE, ASCENDC_TPL_UI_LIST, MATMUL_COMPRESS_PP_MAT_MUL_MODE_TRUE),
        ASCENDC_TPL_UINT_SEL(TRANS, ASCENDC_TPL_UI_LIST, MATMUL_COMPRESS_B_TRANS), ),
#endif
);
#endif // MATMUL_COMPRESS_TILING_KEY_H