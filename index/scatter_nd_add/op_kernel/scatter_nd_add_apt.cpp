/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file scatter_nd_add.cpp
 * \brief scatter_nd_add
 */

#include "./arch35/scatter_nd_add_simt.h"
#include "./arch35/scatter_nd_add_simd.h"
#include "./arch35/scatter_nd_add_common.h"
#include "./arch35/scatter_nd_add_simt_sort.h"
#include "./arch35/scatter_nd_add_simd_non_sort.h"
#include "./arch35/scatter_nd_add_determinstic.h"
#include "./arch35/scatter_nd_add_struct.h"

using namespace ScatterNdAdd;

#define TILING_KEY_FP32_NOT_EXCEED_UINT32 100
#define TILING_KEY_FP16_NOT_EXCEED_UINT32 101
#define TILING_KEY_INT32_NOT_EXCEED_UINT32 103
#define TILING_KEY_INT64_NOT_EXCEED_UINT32 109
#define TILING_KEY_BOOL_NOT_EXCEED_UINT32 112
#define TILING_KEY_FP32_EXCEED_UINT32 200
#define TILING_KEY_FP16_EXCEED_UINT32 201
#define TILING_KEY_INT32_EXCEED_UINT32 203
#define TILING_KEY_INT64_EXCEED_UINT32 209
#define TILING_KEY_BOOL_EXCEED_UINT32 212

__aicore__ inline void fp32VarUnderUint32ScatterNdAddTemplate(GM_ADDR x, GM_ADDR indices, GM_ADDR updates, GM_ADDR y, GM_ADDR workspace, 
                                                                 const ScatterNdAddRegBaseTilingData& tilingData, TPipe &pipe) 
{
    if (tilingData.isDeterminstic == 1) {
        if (tilingData.indiceCastMode == CAST_0) {
            ScatterNdAddDeterministic<float, DTYPE_INDICES, DTYPE_INDICES, CAST_0> op(tilingData, pipe);
            op.Init(x, indices, updates, y, workspace);
            op.Process();
        } else if (tilingData.indiceCastMode == CAST_1) {
            ScatterNdAddDeterministic<float, DTYPE_INDICES, int16_t, CAST_1> op(tilingData, pipe);
            op.Init(x, indices, updates, y, workspace);
            op.Process();
        } else if (tilingData.indiceCastMode == CAST_2) {
            ScatterNdAddDeterministic<float, DTYPE_INDICES, int32_t, CAST_2> op(tilingData, pipe);
            op.Init(x, indices, updates, y, workspace);
            op.Process();
        } else if (tilingData.indiceCastMode == CAST_3) {
            ScatterNdAddDeterministic<float, DTYPE_INDICES, int16_t, CAST_3> op(tilingData, pipe);
            op.Init(x, indices, updates, y, workspace);
            op.Process();
        } else if (tilingData.indiceCastMode == CAST_4) {
            ScatterNdAddDeterministic<float, DTYPE_INDICES, uint8_t, CAST_4> op(tilingData, pipe);
            op.Init(x, indices, updates, y, workspace);
            op.Process();
        } else if (tilingData.indiceCastMode == CAST_5) {
            ScatterNdAddDeterministic<float, DTYPE_INDICES, uint8_t, CAST_5> op(tilingData, pipe);
            op.Init(x, indices, updates, y, workspace);
            op.Process();
        }
    } else if (tilingData.isSimdNonDeterminstic == 1) {
        if (tilingData.indiceCastMode == CAST_0) {
            ScatterNdAddSimd<float, DTYPE_INDICES, DTYPE_INDICES, CAST_0> op(tilingData, pipe);
            op.Init(x, indices, updates, y, workspace);
            op.Process();
        } else if (tilingData.indiceCastMode == CAST_1) {
            ScatterNdAddSimd<float, DTYPE_INDICES, int16_t, CAST_1> op(tilingData, pipe);
            op.Init(x, indices, updates, y, workspace);
            op.Process();
        } else if (tilingData.indiceCastMode == CAST_2) {
            ScatterNdAddSimd<float, DTYPE_INDICES, int32_t, CAST_2> op(tilingData, pipe);
            op.Init(x, indices, updates, y, workspace);
            op.Process();
        } else if (tilingData.indiceCastMode == CAST_3) {
            ScatterNdAddSimd<float, DTYPE_INDICES, int16_t, CAST_3> op(tilingData, pipe);
            op.Init(x, indices, updates, y, workspace);
            op.Process();
        } else if (tilingData.indiceCastMode == CAST_4) {
            ScatterNdAddSimd<float, DTYPE_INDICES, uint8_t, CAST_4> op(tilingData, pipe);
            op.Init(x, indices, updates, y, workspace);
            op.Process();
        } else if (tilingData.indiceCastMode == CAST_5) {
            ScatterNdAddSimd<float, DTYPE_INDICES, uint8_t, CAST_5> op(tilingData, pipe);
            op.Init(x, indices, updates, y, workspace);
            op.Process();
        }
    } else {
        if (tilingData.isSort == 1) {
            if (tilingData.indiceCastMode == CAST_0) {
                ScatterNdAddSimtSort<float, DTYPE_INDICES, uint32_t, DTYPE_INDICES, CAST_0> op(tilingData, pipe);
                op.Init(x, indices, updates, y, workspace);
                op.Process();
            } else if (tilingData.indiceCastMode == CAST_1) {
                ScatterNdAddSimtSort<float, DTYPE_INDICES, uint32_t, int16_t, CAST_1> op(tilingData, pipe);
                op.Init(x, indices, updates, y, workspace);
                op.Process();
            } else if (tilingData.indiceCastMode == CAST_2) {
                ScatterNdAddSimtSort<float, DTYPE_INDICES, uint32_t, int32_t, CAST_2> op(tilingData, pipe);
                op.Init(x, indices, updates, y, workspace);
                op.Process();
            } else if (tilingData.indiceCastMode == CAST_3) {
                ScatterNdAddSimtSort<float, DTYPE_INDICES, uint32_t, int16_t, CAST_3> op(tilingData, pipe);
                op.Init(x, indices, updates, y, workspace);
                op.Process();
            } else if (tilingData.indiceCastMode == CAST_4) {
                ScatterNdAddSimtSort<float, DTYPE_INDICES, uint32_t, uint8_t, CAST_4> op(tilingData, pipe);
                op.Init(x, indices, updates, y, workspace);
                op.Process();
            } else if (tilingData.indiceCastMode == CAST_5) {
                ScatterNdAddSimtSort<float, DTYPE_INDICES, uint32_t, uint8_t, CAST_5> op(tilingData, pipe);
                op.Init(x, indices, updates, y, workspace);
                op.Process();
            }
        } else {
            ScatterNdAddSimt<float, DTYPE_INDICES, uint32_t> op(tilingData, pipe);
            op.Init(x, indices, updates, y, workspace);
            op.Process();
        }
    }
}

__aicore__ inline void fp32VarBeyondUint32ScatterNdAddTemplate(GM_ADDR x, GM_ADDR indices, GM_ADDR updates, GM_ADDR y, GM_ADDR workspace, 
                                                                 const ScatterNdAddRegBaseTilingData& tilingData, TPipe &pipe) 
{
    if (tilingData.isDeterminstic == 1) {
        if (tilingData.indiceCastMode == CAST_0) {
            ScatterNdAddDeterministic<float, DTYPE_INDICES, DTYPE_INDICES, CAST_0> op(tilingData, pipe);
            op.Init(x, indices, updates, y, workspace);
            op.Process();
        } else if (tilingData.indiceCastMode == CAST_1) {
            ScatterNdAddDeterministic<float, DTYPE_INDICES, int16_t, CAST_1> op(tilingData, pipe);
            op.Init(x, indices, updates, y, workspace);
            op.Process();
        } else if (tilingData.indiceCastMode == CAST_2) {
            ScatterNdAddDeterministic<float, DTYPE_INDICES, int32_t, CAST_2> op(tilingData, pipe);
            op.Init(x, indices, updates, y, workspace);
            op.Process();
        } else if (tilingData.indiceCastMode == CAST_3) {
            ScatterNdAddDeterministic<float, DTYPE_INDICES, int16_t, CAST_3> op(tilingData, pipe);
            op.Init(x, indices, updates, y, workspace);
            op.Process();
        } else if (tilingData.indiceCastMode == CAST_4) {
            ScatterNdAddDeterministic<float, DTYPE_INDICES, uint8_t, CAST_4> op(tilingData, pipe);
            op.Init(x, indices, updates, y, workspace);
            op.Process();
        } else if (tilingData.indiceCastMode == CAST_5) {
            ScatterNdAddDeterministic<float, DTYPE_INDICES, uint8_t, CAST_5> op(tilingData, pipe);
            op.Init(x, indices, updates, y, workspace);
            op.Process();
        }
    } else if (tilingData.isSimdNonDeterminstic == 1) {
        if (tilingData.indiceCastMode == CAST_0) {
            ScatterNdAddSimd<float, DTYPE_INDICES, DTYPE_INDICES, CAST_0> op(tilingData, pipe);
            op.Init(x, indices, updates, y, workspace);
            op.Process();
        } else if (tilingData.indiceCastMode == CAST_1) {
            ScatterNdAddSimd<float, DTYPE_INDICES, int16_t, CAST_1> op(tilingData, pipe);
            op.Init(x, indices, updates, y, workspace);
            op.Process();
        } else if (tilingData.indiceCastMode == CAST_2) {
            ScatterNdAddSimd<float, DTYPE_INDICES, int32_t, CAST_2> op(tilingData, pipe);
            op.Init(x, indices, updates, y, workspace);
            op.Process();
        } else if (tilingData.indiceCastMode == CAST_3) {
            ScatterNdAddSimd<float, DTYPE_INDICES, int16_t, CAST_3> op(tilingData, pipe);
            op.Init(x, indices, updates, y, workspace);
            op.Process();
        } else if (tilingData.indiceCastMode == CAST_4) {
            ScatterNdAddSimd<float, DTYPE_INDICES, uint8_t, CAST_4> op(tilingData, pipe);
            op.Init(x, indices, updates, y, workspace);
            op.Process();
        } else if (tilingData.indiceCastMode == CAST_5) {
            ScatterNdAddSimd<float, DTYPE_INDICES, uint8_t, CAST_5> op(tilingData, pipe);
            op.Init(x, indices, updates, y, workspace);
            op.Process();
        }
    } else {
        if (tilingData.isSort == 1) {
            if (tilingData.indiceCastMode == CAST_0) {
                ScatterNdAddSimtSort<float, DTYPE_INDICES, uint64_t, DTYPE_INDICES, CAST_0> op(tilingData, pipe);
                op.Init(x, indices, updates, y, workspace);
                op.Process();
            } else if (tilingData.indiceCastMode == CAST_1) {
                ScatterNdAddSimtSort<float, DTYPE_INDICES, uint64_t, int16_t, CAST_1> op(tilingData, pipe);
                op.Init(x, indices, updates, y, workspace);
                op.Process();
            } else if (tilingData.indiceCastMode == CAST_2) {
                ScatterNdAddSimtSort<float, DTYPE_INDICES, uint64_t, int32_t, CAST_2> op(tilingData, pipe);
                op.Init(x, indices, updates, y, workspace);
                op.Process();
            } else if (tilingData.indiceCastMode == CAST_3) {
                ScatterNdAddSimtSort<float, DTYPE_INDICES, uint64_t, int16_t, CAST_3> op(tilingData, pipe);
                op.Init(x, indices, updates, y, workspace);
                op.Process();
            } else if (tilingData.indiceCastMode == CAST_4) {
                ScatterNdAddSimtSort<float, DTYPE_INDICES, uint64_t, uint8_t, CAST_4> op(tilingData, pipe);
                op.Init(x, indices, updates, y, workspace);
                op.Process();
            } else if (tilingData.indiceCastMode == CAST_5) {
                ScatterNdAddSimtSort<float, DTYPE_INDICES, uint64_t, uint8_t, CAST_5> op(tilingData, pipe);
                op.Init(x, indices, updates, y, workspace);
                op.Process();
            }
        } else {
            ScatterNdAddSimt<float, DTYPE_INDICES, uint64_t> op(tilingData, pipe);
            op.Init(x, indices, updates, y, workspace);
            op.Process();
        }
    }
}

__aicore__ inline void fp16VarUnderUint32ScatterNdAddTemplate(GM_ADDR x, GM_ADDR indices, GM_ADDR updates, GM_ADDR y, GM_ADDR workspace, 
                                                                 const ScatterNdAddRegBaseTilingData& tilingData, TPipe &pipe) 
{
    if (tilingData.isDeterminstic == 1) {
        if (tilingData.indiceCastMode == CAST_0) {
            ScatterNdAddDeterministic<half, DTYPE_INDICES, DTYPE_INDICES, CAST_0> op(tilingData, pipe);
            op.Init(x, indices, updates, y, workspace);
            op.Process();
        } else if (tilingData.indiceCastMode == CAST_1) {
            ScatterNdAddDeterministic<half, DTYPE_INDICES, int16_t, CAST_1> op(tilingData, pipe);
            op.Init(x, indices, updates, y, workspace);
            op.Process();
        } else if (tilingData.indiceCastMode == CAST_2) {
            ScatterNdAddDeterministic<half, DTYPE_INDICES, int32_t, CAST_2> op(tilingData, pipe);
            op.Init(x, indices, updates, y, workspace);
            op.Process();
        } else if (tilingData.indiceCastMode == CAST_3) {
            ScatterNdAddDeterministic<half, DTYPE_INDICES, int16_t, CAST_3> op(tilingData, pipe);
            op.Init(x, indices, updates, y, workspace);
            op.Process();
        } else if (tilingData.indiceCastMode == CAST_4) {
            ScatterNdAddDeterministic<half, DTYPE_INDICES, uint8_t, CAST_4> op(tilingData, pipe);
            op.Init(x, indices, updates, y, workspace);
            op.Process();
        } else if (tilingData.indiceCastMode == CAST_5) {
            ScatterNdAddDeterministic<half, DTYPE_INDICES, uint8_t, CAST_5> op(tilingData, pipe);
            op.Init(x, indices, updates, y, workspace);
            op.Process();
        }
    } else if (tilingData.isSimdNonDeterminstic == 1) {
        if (tilingData.indiceCastMode == CAST_0) {
            ScatterNdAddSimd<half, DTYPE_INDICES, DTYPE_INDICES, CAST_0> op(tilingData, pipe);
            op.Init(x, indices, updates, y, workspace);
            op.Process();
        } else if (tilingData.indiceCastMode == CAST_1) {
            ScatterNdAddSimd<half, DTYPE_INDICES, int16_t, CAST_1> op(tilingData, pipe);
            op.Init(x, indices, updates, y, workspace);
            op.Process();
        } else if (tilingData.indiceCastMode == CAST_2) {
            ScatterNdAddSimd<half, DTYPE_INDICES, int32_t, CAST_2> op(tilingData, pipe);
            op.Init(x, indices, updates, y, workspace);
            op.Process();
        } else if (tilingData.indiceCastMode == CAST_3) {
            ScatterNdAddSimd<half, DTYPE_INDICES, int16_t, CAST_3> op(tilingData, pipe);
            op.Init(x, indices, updates, y, workspace);
            op.Process();
        } else if (tilingData.indiceCastMode == CAST_4) {
            ScatterNdAddSimd<half, DTYPE_INDICES, uint8_t, CAST_4> op(tilingData, pipe);
            op.Init(x, indices, updates, y, workspace);
            op.Process();
        } else if (tilingData.indiceCastMode == CAST_5) {
            ScatterNdAddSimd<half, DTYPE_INDICES, uint8_t, CAST_5> op(tilingData, pipe);
            op.Init(x, indices, updates, y, workspace);
            op.Process();
        }
    } else {
        if (tilingData.isSort == 1) {
            if (tilingData.indiceCastMode == CAST_0) {
                ScatterNdAddSimtSort<half, DTYPE_INDICES, uint32_t, DTYPE_INDICES, CAST_0> op(tilingData, pipe);
                op.Init(x, indices, updates, y, workspace);
                op.Process();
            } else if (tilingData.indiceCastMode == CAST_1) {
                ScatterNdAddSimtSort<half, DTYPE_INDICES, uint32_t, int16_t, CAST_1> op(tilingData, pipe);
                op.Init(x, indices, updates, y, workspace);
                op.Process();
            } else if (tilingData.indiceCastMode == CAST_2) {
                ScatterNdAddSimtSort<half, DTYPE_INDICES, uint32_t, int32_t, CAST_2> op(tilingData, pipe);
                op.Init(x, indices, updates, y, workspace);
                op.Process();
            } else if (tilingData.indiceCastMode == CAST_3) {
                ScatterNdAddSimtSort<half, DTYPE_INDICES, uint32_t, int16_t, CAST_3> op(tilingData, pipe);
                op.Init(x, indices, updates, y, workspace);
                op.Process();
            } else if (tilingData.indiceCastMode == CAST_4) {
                ScatterNdAddSimtSort<half, DTYPE_INDICES, uint32_t, uint8_t, CAST_4> op(tilingData, pipe);
                op.Init(x, indices, updates, y, workspace);
                op.Process();
            } else if (tilingData.indiceCastMode == CAST_5) {
                ScatterNdAddSimtSort<half, DTYPE_INDICES, uint32_t, uint8_t, CAST_5> op(tilingData, pipe);
                op.Init(x, indices, updates, y, workspace);
                op.Process();
            }
        } else {
            ScatterNdAddSimt<half, DTYPE_INDICES, uint32_t> op(tilingData, pipe);
            op.Init(x, indices, updates, y, workspace);
            op.Process();
        }
    }
}

__aicore__ inline void fp16VarBeyondUint32ScatterNdAddTemplate(GM_ADDR x, GM_ADDR indices, GM_ADDR updates, GM_ADDR y, GM_ADDR workspace, 
                                                                 const ScatterNdAddRegBaseTilingData& tilingData, TPipe &pipe) 
{
    if (tilingData.isDeterminstic == 1) {
        if (tilingData.indiceCastMode == CAST_0) {
            ScatterNdAddDeterministic<half, DTYPE_INDICES, DTYPE_INDICES, CAST_0> op(tilingData, pipe);
            op.Init(x, indices, updates, y, workspace);
            op.Process();
        } else if (tilingData.indiceCastMode == CAST_1) {
            ScatterNdAddDeterministic<half, DTYPE_INDICES, int16_t, CAST_1> op(tilingData, pipe);
            op.Init(x, indices, updates, y, workspace);
            op.Process();
        } else if (tilingData.indiceCastMode == CAST_2) {
            ScatterNdAddDeterministic<half, DTYPE_INDICES, int32_t, CAST_2> op(tilingData, pipe);
            op.Init(x, indices, updates, y, workspace);
            op.Process();
        } else if (tilingData.indiceCastMode == CAST_3) {
            ScatterNdAddDeterministic<half, DTYPE_INDICES, int16_t, CAST_3> op(tilingData, pipe);
            op.Init(x, indices, updates, y, workspace);
            op.Process();
        } else if (tilingData.indiceCastMode == CAST_4) {
            ScatterNdAddDeterministic<half, DTYPE_INDICES, uint8_t, CAST_4> op(tilingData, pipe);
            op.Init(x, indices, updates, y, workspace);
            op.Process();
        } else if (tilingData.indiceCastMode == CAST_5) {
            ScatterNdAddDeterministic<half, DTYPE_INDICES, uint8_t, CAST_5> op(tilingData, pipe);
            op.Init(x, indices, updates, y, workspace);
            op.Process();
        }
    } else if (tilingData.isSimdNonDeterminstic == 1) {
        if (tilingData.indiceCastMode == CAST_0) {
            ScatterNdAddSimd<half, DTYPE_INDICES, DTYPE_INDICES, CAST_0> op(tilingData, pipe);
            op.Init(x, indices, updates, y, workspace);
            op.Process();
        } else if (tilingData.indiceCastMode == CAST_1) {
            ScatterNdAddSimd<half, DTYPE_INDICES, int16_t, CAST_1> op(tilingData, pipe);
            op.Init(x, indices, updates, y, workspace);
            op.Process();
        } else if (tilingData.indiceCastMode == CAST_2) {
            ScatterNdAddSimd<half, DTYPE_INDICES, int32_t, CAST_2> op(tilingData, pipe);
            op.Init(x, indices, updates, y, workspace);
            op.Process();
        } else if (tilingData.indiceCastMode == CAST_3) {
            ScatterNdAddSimd<half, DTYPE_INDICES, int16_t, CAST_3> op(tilingData, pipe);
            op.Init(x, indices, updates, y, workspace);
            op.Process();
        } else if (tilingData.indiceCastMode == CAST_4) {
            ScatterNdAddSimd<half, DTYPE_INDICES, uint8_t, CAST_4> op(tilingData, pipe);
            op.Init(x, indices, updates, y, workspace);
            op.Process();
        } else if (tilingData.indiceCastMode == CAST_5) {
            ScatterNdAddSimd<half, DTYPE_INDICES, uint8_t, CAST_5> op(tilingData, pipe);
            op.Init(x, indices, updates, y, workspace);
            op.Process();
        }
    } else {
        if (tilingData.isSort == 1) { 
            if (tilingData.indiceCastMode == CAST_0) {
                ScatterNdAddSimtSort<half, DTYPE_INDICES, uint64_t, DTYPE_INDICES, CAST_0> op(tilingData, pipe);
                op.Init(x, indices, updates, y, workspace);
                op.Process();
            } else if (tilingData.indiceCastMode == CAST_1) {
                ScatterNdAddSimtSort<half, DTYPE_INDICES, uint64_t, int16_t, CAST_1> op(tilingData, pipe);
                op.Init(x, indices, updates, y, workspace);
                op.Process();
            } else if (tilingData.indiceCastMode == CAST_2) {
                ScatterNdAddSimtSort<half, DTYPE_INDICES, uint64_t, int32_t, CAST_2> op(tilingData, pipe);
                op.Init(x, indices, updates, y, workspace);
                op.Process();
            } else if (tilingData.indiceCastMode == CAST_3) {
                ScatterNdAddSimtSort<half, DTYPE_INDICES, uint64_t, int16_t, CAST_3> op(tilingData, pipe);
                op.Init(x, indices, updates, y, workspace);
                op.Process();
            } else if (tilingData.indiceCastMode == CAST_4) {
                ScatterNdAddSimtSort<half, DTYPE_INDICES, uint64_t, uint8_t, CAST_4> op(tilingData, pipe);
                op.Init(x, indices, updates, y, workspace);
                op.Process();
            } else if (tilingData.indiceCastMode == CAST_5) {
                ScatterNdAddSimtSort<half, DTYPE_INDICES, uint64_t, uint8_t, CAST_5> op(tilingData, pipe);
                op.Init(x, indices, updates, y, workspace);
                op.Process();
            }
        } else {
            ScatterNdAddSimt<half, DTYPE_INDICES, uint64_t> op(tilingData, pipe);
            op.Init(x, indices, updates, y, workspace);
            op.Process();
        }
    }
}

__aicore__ inline void int32VarUnderUint32ScatterNdAddTemplate(GM_ADDR x, GM_ADDR indices, GM_ADDR updates, GM_ADDR y, GM_ADDR workspace, 
                                                                 const ScatterNdAddRegBaseTilingData& tilingData, TPipe &pipe) 
{
    if (tilingData.isSimdNonDeterminstic == 1) {
        if (tilingData.indiceCastMode == CAST_0) {
            ScatterNdAddSimd<int32_t, DTYPE_INDICES, DTYPE_INDICES, CAST_0> op(tilingData, pipe);
            op.Init(x, indices, updates, y, workspace);
            op.Process();
        } else if (tilingData.indiceCastMode == CAST_1) {
            ScatterNdAddSimd<int32_t, DTYPE_INDICES, int16_t, CAST_1> op(tilingData, pipe);
            op.Init(x, indices, updates, y, workspace);
            op.Process();
        } else if (tilingData.indiceCastMode == CAST_2) {
            ScatterNdAddSimd<int32_t, DTYPE_INDICES, int32_t, CAST_2> op(tilingData, pipe);
            op.Init(x, indices, updates, y, workspace);
            op.Process();
        } else if (tilingData.indiceCastMode == CAST_3) {
            ScatterNdAddSimd<int32_t, DTYPE_INDICES, int16_t, CAST_3> op(tilingData, pipe);
            op.Init(x, indices, updates, y, workspace);
            op.Process();
        } else if (tilingData.indiceCastMode == CAST_4) {
            ScatterNdAddSimd<int32_t, DTYPE_INDICES, uint8_t, CAST_4> op(tilingData, pipe);
            op.Init(x, indices, updates, y, workspace);
            op.Process();
        } else if (tilingData.indiceCastMode == CAST_5) {
            ScatterNdAddSimd<int32_t, DTYPE_INDICES, uint8_t, CAST_5> op(tilingData, pipe);
            op.Init(x, indices, updates, y, workspace);
            op.Process();
        }
    } else {
        if (tilingData.isSort == 1) {
            if (tilingData.indiceCastMode == CAST_0) {
                ScatterNdAddSimtSort<int32_t, DTYPE_INDICES, uint32_t, DTYPE_INDICES, CAST_1> op(tilingData, pipe);
                op.Init(x, indices, updates, y, workspace);
                op.Process();
            } else if (tilingData.indiceCastMode == CAST_1) {
                ScatterNdAddSimtSort<int32_t, DTYPE_INDICES, uint32_t, int16_t, CAST_1> op(tilingData, pipe);
                op.Init(x, indices, updates, y, workspace);
                op.Process();
            } else if (tilingData.indiceCastMode == CAST_2) {
                ScatterNdAddSimtSort<int32_t, DTYPE_INDICES, uint32_t, int32_t, CAST_2> op(tilingData, pipe);
                op.Init(x, indices, updates, y, workspace);
                op.Process();
            } else if (tilingData.indiceCastMode == CAST_3) {
                ScatterNdAddSimtSort<int32_t, DTYPE_INDICES, uint32_t, int16_t, CAST_3> op(tilingData, pipe);
                op.Init(x, indices, updates, y, workspace);
                op.Process();
            } else if (tilingData.indiceCastMode == CAST_4) {
                ScatterNdAddSimtSort<int32_t, DTYPE_INDICES, uint32_t, uint8_t, CAST_4> op(tilingData, pipe);
                op.Init(x, indices, updates, y, workspace);
                op.Process();
            } else if (tilingData.indiceCastMode == CAST_5) {
                ScatterNdAddSimtSort<int32_t, DTYPE_INDICES, uint32_t, uint8_t, CAST_5> op(tilingData, pipe);
                op.Init(x, indices, updates, y, workspace);
                op.Process();
            }
        } else {
            ScatterNdAddSimt<int32_t, DTYPE_INDICES, uint32_t> op(tilingData, pipe);
            op.Init(x, indices, updates, y, workspace);
            op.Process();
        }
    }
}

__aicore__ inline void int32VarBeyondUint32ScatterNdAddTemplate(GM_ADDR x, GM_ADDR indices, GM_ADDR updates, GM_ADDR y, GM_ADDR workspace, 
                                                                 const ScatterNdAddRegBaseTilingData& tilingData, TPipe &pipe) 
{
    if (tilingData.isSimdNonDeterminstic == 1) {
        if (tilingData.indiceCastMode == CAST_0) {
            ScatterNdAddSimd<int32_t, DTYPE_INDICES, DTYPE_INDICES, CAST_0> op(tilingData, pipe);
            op.Init(x, indices, updates, y, workspace);
            op.Process();
        } else if (tilingData.indiceCastMode == CAST_1) {
            ScatterNdAddSimd<int32_t, DTYPE_INDICES, int16_t, CAST_1> op(tilingData, pipe);
            op.Init(x, indices, updates, y, workspace);
            op.Process();
        } else if (tilingData.indiceCastMode == CAST_2) {
            ScatterNdAddSimd<int32_t, DTYPE_INDICES, int32_t, CAST_2> op(tilingData, pipe);
            op.Init(x, indices, updates, y, workspace);
            op.Process();
        } else if (tilingData.indiceCastMode == CAST_3) {
            ScatterNdAddSimd<int32_t, DTYPE_INDICES, int16_t, CAST_3> op(tilingData, pipe);
            op.Init(x, indices, updates, y, workspace);
            op.Process();
        } else if (tilingData.indiceCastMode == CAST_4) {
            ScatterNdAddSimd<int32_t, DTYPE_INDICES, uint8_t, CAST_4> op(tilingData, pipe);
            op.Init(x, indices, updates, y, workspace);
            op.Process();
        } else if (tilingData.indiceCastMode == CAST_5) {
            ScatterNdAddSimd<int32_t, DTYPE_INDICES, uint8_t, CAST_5> op(tilingData, pipe);
            op.Init(x, indices, updates, y, workspace);
            op.Process();
        }
    } else {
        if (tilingData.isSort == 1) {
            if (tilingData.indiceCastMode == CAST_0) {
                ScatterNdAddSimtSort<int32_t, DTYPE_INDICES, uint64_t, DTYPE_INDICES, CAST_0> op(tilingData, pipe);
                op.Init(x, indices, updates, y, workspace);
                op.Process();
            } else if (tilingData.indiceCastMode == CAST_1) {
                ScatterNdAddSimtSort<int32_t, DTYPE_INDICES, uint64_t, int16_t, CAST_1> op(tilingData, pipe);
                op.Init(x, indices, updates, y, workspace);
                op.Process();
            } else if (tilingData.indiceCastMode == CAST_2) {
                ScatterNdAddSimtSort<int32_t, DTYPE_INDICES, uint64_t, int32_t, CAST_2> op(tilingData, pipe);
                op.Init(x, indices, updates, y, workspace);
                op.Process();
            } else if (tilingData.indiceCastMode == CAST_3) {
                ScatterNdAddSimtSort<int32_t, DTYPE_INDICES, uint64_t, int16_t, CAST_3> op(tilingData, pipe);
                op.Init(x, indices, updates, y, workspace);
                op.Process();
            } else if (tilingData.indiceCastMode == CAST_4) {
                ScatterNdAddSimtSort<int32_t, DTYPE_INDICES, uint64_t, uint8_t, CAST_4> op(tilingData, pipe);
                op.Init(x, indices, updates, y, workspace);
                op.Process();
            } else if (tilingData.indiceCastMode == CAST_5) {
                ScatterNdAddSimtSort<int32_t, DTYPE_INDICES, uint64_t, uint8_t, CAST_5> op(tilingData, pipe);
                op.Init(x, indices, updates, y, workspace);
                op.Process();
            }
        } else {
            ScatterNdAddSimt<int32_t, DTYPE_INDICES, uint64_t> op(tilingData, pipe);
            op.Init(x, indices, updates, y, workspace);
            op.Process();
        }
    }
}

__aicore__ inline void boolVarUnderUint32ScatterNdAddTemplate(GM_ADDR x, GM_ADDR indices, GM_ADDR updates, GM_ADDR y, GM_ADDR workspace, 
                                                                 const ScatterNdAddRegBaseTilingData& tilingData, TPipe &pipe) 
{
    if (tilingData.isSimdNonDeterminstic == 1) {
        // cast添加
        if (tilingData.indiceCastMode == CAST_0) {
            ScatterNdAddSimd<bool, DTYPE_INDICES, DTYPE_INDICES, CAST_0> op(tilingData, pipe);
            op.Init(x, indices, updates, y, workspace);
            op.Process();
        } else if (tilingData.indiceCastMode == CAST_1) {
            ScatterNdAddSimd<bool, DTYPE_INDICES, int16_t, CAST_1> op(tilingData, pipe);
            op.Init(x, indices, updates, y, workspace);
            op.Process();
        } else if (tilingData.indiceCastMode == CAST_2) {
            ScatterNdAddSimd<bool, DTYPE_INDICES, int32_t, CAST_2> op(tilingData, pipe);
            op.Init(x, indices, updates, y, workspace);
            op.Process();
        } else if (tilingData.indiceCastMode == CAST_3) {
            ScatterNdAddSimd<bool, DTYPE_INDICES, int16_t, CAST_3> op(tilingData, pipe);
            op.Init(x, indices, updates, y, workspace);
            op.Process();
        } else if (tilingData.indiceCastMode == CAST_4) {
            ScatterNdAddSimd<bool, DTYPE_INDICES, uint8_t, CAST_4> op(tilingData, pipe);
            op.Init(x, indices, updates, y, workspace);
            op.Process();
        } else if (tilingData.indiceCastMode == CAST_5) {
            ScatterNdAddSimd<bool, DTYPE_INDICES, uint8_t, CAST_5> op(tilingData, pipe);
            op.Init(x, indices, updates, y, workspace);
            op.Process();
        }
    } else {
        if (tilingData.isSort == 1) {
            if (tilingData.indiceCastMode == CAST_0) {
                ScatterNdAddSimtSort<bool, DTYPE_INDICES, uint32_t, DTYPE_INDICES, CAST_0> op(tilingData, pipe);
                op.Init(x, indices, updates, y, workspace);
                op.Process();
            } else if (tilingData.indiceCastMode == CAST_1) {
                ScatterNdAddSimtSort<bool, DTYPE_INDICES, uint32_t, int16_t, CAST_1> op(tilingData, pipe);
                op.Init(x, indices, updates, y, workspace);
                op.Process();
            } else if (tilingData.indiceCastMode == CAST_2) {
                ScatterNdAddSimtSort<bool, DTYPE_INDICES, uint32_t, int32_t, CAST_2> op(tilingData, pipe);
                op.Init(x, indices, updates, y, workspace);
                op.Process();
            } else if (tilingData.indiceCastMode == CAST_3) {
                ScatterNdAddSimtSort<bool, DTYPE_INDICES, uint32_t, int16_t, CAST_3> op(tilingData, pipe);
                op.Init(x, indices, updates, y, workspace);
                op.Process();
            } else if (tilingData.indiceCastMode == CAST_4) {
                ScatterNdAddSimtSort<bool, DTYPE_INDICES, uint32_t, uint8_t, CAST_4> op(tilingData, pipe);
                op.Init(x, indices, updates, y, workspace);
                op.Process();
            } else if (tilingData.indiceCastMode == CAST_5) {
                ScatterNdAddSimtSort<bool, DTYPE_INDICES, uint32_t, uint8_t, CAST_5> op(tilingData, pipe);
                op.Init(x, indices, updates, y, workspace);
                op.Process();
            }
        } else {
            ScatterNdAddSimt<bool, DTYPE_INDICES, uint32_t> op(tilingData, pipe);
            op.Init(x, indices, updates, y, workspace);
            op.Process();
        }
    }
}

__aicore__ inline void boolVarBeyondUint32ScatterNdAddTemplate(GM_ADDR x, GM_ADDR indices, GM_ADDR updates, GM_ADDR y, GM_ADDR workspace, 
                                                                 const ScatterNdAddRegBaseTilingData& tilingData, TPipe &pipe) 
{
    if (tilingData.isSimdNonDeterminstic == 1) {
        if (tilingData.indiceCastMode == CAST_0) {
            ScatterNdAddSimd<bool, DTYPE_INDICES, DTYPE_INDICES, CAST_0> op(tilingData, pipe);
            op.Init(x, indices, updates, y, workspace);
            op.Process();
        } else if (tilingData.indiceCastMode == CAST_1) {
            ScatterNdAddSimd<bool, DTYPE_INDICES, int16_t, CAST_1> op(tilingData, pipe);
            op.Init(x, indices, updates, y, workspace);
            op.Process();
        } else if (tilingData.indiceCastMode == CAST_2) {
            ScatterNdAddSimd<bool, DTYPE_INDICES, int32_t, CAST_2> op(tilingData, pipe);
            op.Init(x, indices, updates, y, workspace);
            op.Process();
        } else if (tilingData.indiceCastMode == CAST_3) {
            ScatterNdAddSimd<bool, DTYPE_INDICES, int16_t, CAST_3> op(tilingData, pipe);
            op.Init(x, indices, updates, y, workspace);
            op.Process();
        } else if (tilingData.indiceCastMode == CAST_4) {
            ScatterNdAddSimd<bool, DTYPE_INDICES, uint8_t, CAST_4> op(tilingData, pipe);
            op.Init(x, indices, updates, y, workspace);
            op.Process();
        } else if (tilingData.indiceCastMode == CAST_5) {
            ScatterNdAddSimd<bool, DTYPE_INDICES, uint8_t, CAST_5> op(tilingData, pipe);
            op.Init(x, indices, updates, y, workspace);
            op.Process();
        }
    } else {
        if (tilingData.isSort == 1) {
            if (tilingData.indiceCastMode == CAST_0) {
                ScatterNdAddSimtSort<bool, DTYPE_INDICES, uint64_t, DTYPE_INDICES, CAST_0> op(tilingData, pipe);
                op.Init(x, indices, updates, y, workspace);
                op.Process();
            } else if (tilingData.indiceCastMode == CAST_1) {
                ScatterNdAddSimtSort<bool, DTYPE_INDICES, uint64_t, int16_t, CAST_1> op(tilingData, pipe);
                op.Init(x, indices, updates, y, workspace);
                op.Process();
            } else if (tilingData.indiceCastMode == CAST_2) {
                ScatterNdAddSimtSort<bool, DTYPE_INDICES, uint64_t, int32_t, CAST_2> op(tilingData, pipe);
                op.Init(x, indices, updates, y, workspace);
                op.Process();
            } else if (tilingData.indiceCastMode == CAST_3) {
                ScatterNdAddSimtSort<bool, DTYPE_INDICES, uint64_t, int16_t, CAST_3> op(tilingData, pipe);
                op.Init(x, indices, updates, y, workspace);
                op.Process();
            } else if (tilingData.indiceCastMode == CAST_4) {
                ScatterNdAddSimtSort<bool, DTYPE_INDICES, uint64_t, uint8_t, CAST_4> op(tilingData, pipe);
                op.Init(x, indices, updates, y, workspace);
                op.Process();
            } else if (tilingData.indiceCastMode == CAST_5) {
                ScatterNdAddSimtSort<bool, DTYPE_INDICES, uint64_t, uint8_t, CAST_5> op(tilingData, pipe);
                op.Init(x, indices, updates, y, workspace);
                op.Process();
            }
        } else {
            ScatterNdAddSimt<bool, DTYPE_INDICES, uint64_t> op(tilingData, pipe);
            op.Init(x, indices, updates, y, workspace);
            op.Process();
        }
    }
}

extern "C" __global__ __aicore__ void scatter_nd_add(
    GM_ADDR x, GM_ADDR indices, GM_ADDR updates, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }
    SetSysWorkspace(workspace);
    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }
    REGISTER_TILING_DEFAULT(ScatterNdAddRegBaseTilingData);
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);

    if (TILING_KEY_IS(TILING_KEY_FP32_NOT_EXCEED_UINT32)) {
        fp32VarUnderUint32ScatterNdAddTemplate(x, indices, updates, y, workspace, tilingData, pipe);
    } else if (TILING_KEY_IS(TILING_KEY_FP16_NOT_EXCEED_UINT32)) {
        fp16VarUnderUint32ScatterNdAddTemplate(x, indices, updates, y, workspace, tilingData, pipe);
    } else if (TILING_KEY_IS(TILING_KEY_INT32_NOT_EXCEED_UINT32)) {
        int32VarUnderUint32ScatterNdAddTemplate(x, indices, updates, y, workspace, tilingData, pipe);
    } else if (TILING_KEY_IS(TILING_KEY_INT64_NOT_EXCEED_UINT32)) {
        ScatterNdAddSimt<int64_t, DTYPE_INDICES, uint32_t> op(tilingData, pipe);
        op.Init(x, indices, updates, y, workspace);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_BOOL_NOT_EXCEED_UINT32)) {
        boolVarUnderUint32ScatterNdAddTemplate(x, indices, updates, y, workspace, tilingData, pipe);
    } else if (TILING_KEY_IS(TILING_KEY_FP32_EXCEED_UINT32)) {
        fp32VarBeyondUint32ScatterNdAddTemplate(x, indices, updates, y, workspace, tilingData, pipe);
    } else if (TILING_KEY_IS(TILING_KEY_FP16_EXCEED_UINT32)) {
        fp16VarBeyondUint32ScatterNdAddTemplate(x, indices, updates, y, workspace, tilingData, pipe);
    } else if (TILING_KEY_IS(TILING_KEY_INT32_EXCEED_UINT32)) {
        int32VarBeyondUint32ScatterNdAddTemplate(x, indices, updates, y, workspace, tilingData, pipe);
    } else if (TILING_KEY_IS(TILING_KEY_INT64_EXCEED_UINT32)) {
        ScatterNdAddSimt<int64_t, DTYPE_INDICES, uint64_t> op(tilingData, pipe);
        op.Init(x, indices, updates, y, workspace);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_BOOL_EXCEED_UINT32)) {
        boolVarBeyondUint32ScatterNdAddTemplate(x, indices, updates, y, workspace, tilingData, pipe);
    }
}