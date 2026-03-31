/* *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file reverse_v2.cpp
 * \brief reverse_v2
 */

 #include "../inc/platform.h"
 #include "arch35/reverse_v2_simt.h"
 #include "arch35/reverse_v2_simd.h"
 #include "arch35/reverse_v2_tensor_move.h"

 using namespace ReverseV2;
 
 #define REVERSE_V2_DIM8_Y_N_UINT32 8101
 #define REVERSE_V2_DIM8_N_Y_UINT32 8011
 #define REVERSE_V2_DIM8_Y_N_UINT64 8100
 #define REVERSE_V2_DIM8_N_Y_UINT64 8010
 #define REVERSE_V2_DIM7_Y_N_UINT32 7101
 #define REVERSE_V2_DIM7_N_Y_UINT32 7011
 #define REVERSE_V2_DIM7_Y_N_UINT64 7100
 #define REVERSE_V2_DIM7_N_Y_UINT64 7010
 #define REVERSE_V2_DIM6_Y_N_UINT32 6101
 #define REVERSE_V2_DIM6_N_Y_UINT32 6011
 #define REVERSE_V2_DIM6_Y_N_UINT64 6100
 #define REVERSE_V2_DIM6_N_Y_UINT64 6010
 #define REVERSE_V2_DIM5_Y_N_UINT32 5101
 #define REVERSE_V2_DIM5_N_Y_UINT32 5011
 #define REVERSE_V2_DIM5_Y_N_UINT64 5100
 #define REVERSE_V2_DIM5_N_Y_UINT64 5010
 #define REVERSE_V2_DIM4_Y_N_UINT32 4101
 #define REVERSE_V2_DIM4_N_Y_UINT32 4011
 #define REVERSE_V2_DIM4_Y_N_UINT64 4100
 #define REVERSE_V2_DIM4_N_Y_UINT64 4010
 #define REVERSE_V2_DIM3_Y_N_UINT32 3101
 #define REVERSE_V2_DIM3_N_Y_UINT32 3011
 #define REVERSE_V2_DIM3_Y_N_UINT64 3100
 #define REVERSE_V2_DIM3_N_Y_UINT64 3010
 #define REVERSE_V2_DIM2_Y_N_UINT32 2101
 #define REVERSE_V2_DIM2_N_Y_UINT32 2011
 #define REVERSE_V2_DIM2_Y_N_UINT64 2100
 #define REVERSE_V2_DIM2_N_Y_UINT64 2010
 #define REVERSE_V2_DIM1_Y_N_UINT32 1101
 #define REVERSE_V2_DIM1_Y_N_UINT64 1100
 #define TENSOR_MOVE_ONE_BYTE 1000
 #define TENSOR_MOVE_TWO_BYTE 2000
 #define TENSOR_MOVE_FOUR_BYTE 4000
 #define TENSOR_MOVE_EIGHT_BYTE 8000

 #define REVERSE_V2_SIMD_TILING_KEY 10001

 constexpr uint32_t DIM_NUM_8 = 8;
 constexpr uint32_t DIM_NUM_7 = 7;
 constexpr uint32_t DIM_NUM_6 = 6;
 constexpr uint32_t DIM_NUM_5 = 5;
 constexpr uint32_t DIM_NUM_4 = 4;
 constexpr uint32_t DIM_NUM_3 = 3;
 constexpr uint32_t DIM_NUM_2 = 2;
 constexpr uint32_t DIM_NUM_1 = 1;
 
 KERNEL_API void reverse_v2(GM_ADDR x, GM_ADDR axis, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
 {
     TPipe pipe;
     KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
     if (TILING_KEY_IS(TENSOR_MOVE_ONE_BYTE)) {
         GET_TILING_DATA_WITH_STRUCT(TensorMoveTilingData, tilingData, tiling);
         ReverseV2::ReverseV2TensorMoveKernel<int8_t> op;
         op.Init(x, y, nullptr, tilingData);
         op.Process();
     } else if (TILING_KEY_IS(TENSOR_MOVE_TWO_BYTE)) {
         GET_TILING_DATA_WITH_STRUCT(TensorMoveTilingData, tilingData, tiling);
         ReverseV2::ReverseV2TensorMoveKernel<int16_t> op;
         op.Init(x, y, nullptr, tilingData);
         op.Process();
     } else if (TILING_KEY_IS(TENSOR_MOVE_FOUR_BYTE)) {
         GET_TILING_DATA_WITH_STRUCT(TensorMoveTilingData, tilingData, tiling);
         ReverseV2::ReverseV2TensorMoveKernel<int32_t> op;
         op.Init(x, y, nullptr, tilingData);
         op.Process();
     } else if (TILING_KEY_IS(TENSOR_MOVE_EIGHT_BYTE)) {
         GET_TILING_DATA_WITH_STRUCT(TensorMoveTilingData, tilingData, tiling);
         ReverseV2::ReverseV2TensorMoveKernel<int64_t> op;
         op.Init(x, y, nullptr, tilingData);
         op.Process();
     } else {
        // Non-tensormove Scenario
        GET_TILING_DATA_WITH_STRUCT(ReverseV2TilingData4AscendC, tilingData, tiling);
         if (TILING_KEY_IS(REVERSE_V2_DIM8_Y_N_UINT32)) {
            if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
                ReverseV2Simt<uint32_t, int8_t, true, DIM_NUM_8> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int16_t)) {
                ReverseV2Simt<uint32_t, int16_t, true, DIM_NUM_8> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
                ReverseV2Simt<uint32_t, int32_t, true, DIM_NUM_8> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
                ReverseV2Simt<uint32_t, int64_t, true, DIM_NUM_8> op;
                op.Init(x, y, &tilingData);
                op.Process();
            }
         } else if (TILING_KEY_IS(REVERSE_V2_DIM8_N_Y_UINT32)) {
            if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
                ReverseV2Simt<uint32_t, int8_t, false, DIM_NUM_8> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int16_t)) {
                ReverseV2Simt<uint32_t, int16_t, false, DIM_NUM_8> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
                ReverseV2Simt<uint32_t, int32_t, false, DIM_NUM_8> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
                ReverseV2Simt<uint32_t, int64_t, false, DIM_NUM_8> op;
                op.Init(x, y, &tilingData);
                op.Process();
            }
        } else if (TILING_KEY_IS(REVERSE_V2_DIM8_Y_N_UINT64)) {
            if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
                ReverseV2Simt<uint64_t, int8_t, true, DIM_NUM_8> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int16_t)) {
                ReverseV2Simt<uint64_t, int16_t, true, DIM_NUM_8> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
                ReverseV2Simt<uint64_t, int32_t, true, DIM_NUM_8> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
                ReverseV2Simt<uint64_t, int64_t, true, DIM_NUM_8> op;
                op.Init(x, y, &tilingData);
                op.Process();
            }
        } else if (TILING_KEY_IS(REVERSE_V2_DIM8_N_Y_UINT64)) {
            if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
                ReverseV2Simt<uint64_t, int8_t, false, DIM_NUM_8> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int16_t)) {
                ReverseV2Simt<uint64_t, int16_t, false, DIM_NUM_8> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
                ReverseV2Simt<uint64_t, int32_t, false, DIM_NUM_8> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
                ReverseV2Simt<uint64_t, int64_t, false, DIM_NUM_8> op;
                op.Init(x, y, &tilingData);
                op.Process();
            }
        } else if (TILING_KEY_IS(REVERSE_V2_DIM7_Y_N_UINT32)) {
            if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
                ReverseV2Simt<uint32_t, int8_t, true, DIM_NUM_7> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int16_t)) {
                ReverseV2Simt<uint32_t, int16_t, true, DIM_NUM_7> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
                ReverseV2Simt<uint32_t, int32_t, true, DIM_NUM_7> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
                ReverseV2Simt<uint32_t, int64_t, true, DIM_NUM_7> op;
                op.Init(x, y, &tilingData);
                op.Process();
            }
        } else if (TILING_KEY_IS(REVERSE_V2_DIM7_N_Y_UINT32)) {
            if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
                ReverseV2Simt<uint32_t, int8_t, false, DIM_NUM_7> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int16_t)) {
                ReverseV2Simt<uint32_t, int16_t, false, DIM_NUM_7> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
                ReverseV2Simt<uint32_t, int32_t, false, DIM_NUM_7> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
                ReverseV2Simt<uint32_t, int64_t, false, DIM_NUM_7> op;
                op.Init(x, y, &tilingData);
                op.Process();
            }
        } else if (TILING_KEY_IS(REVERSE_V2_DIM7_Y_N_UINT64)) {
            if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
                ReverseV2Simt<uint64_t, int8_t, true, DIM_NUM_7> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int16_t)) {
                ReverseV2Simt<uint64_t, int16_t, true, DIM_NUM_7> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
                ReverseV2Simt<uint64_t, int32_t, true, DIM_NUM_7> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
                ReverseV2Simt<uint64_t, int64_t, true, DIM_NUM_7> op;
                op.Init(x, y, &tilingData);
                op.Process();
            }
        } else if (TILING_KEY_IS(REVERSE_V2_DIM7_N_Y_UINT64)) {
            if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
                ReverseV2Simt<uint64_t, int8_t, false, DIM_NUM_7> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int16_t)) {
                ReverseV2Simt<uint64_t, int16_t, false, DIM_NUM_7> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
                ReverseV2Simt<uint64_t, int32_t, false, DIM_NUM_7> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
                ReverseV2Simt<uint64_t, int64_t, false, DIM_NUM_7> op;
                op.Init(x, y, &tilingData);
                op.Process();
            }
        } else if (TILING_KEY_IS(REVERSE_V2_DIM6_Y_N_UINT32)) {
            if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
                ReverseV2Simt<uint32_t, int8_t, true, DIM_NUM_6> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int16_t)) {
                ReverseV2Simt<uint32_t, int16_t, true, DIM_NUM_6> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
                ReverseV2Simt<uint32_t, int32_t, true, DIM_NUM_6> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
                ReverseV2Simt<uint32_t, int64_t, true, DIM_NUM_6> op;
                op.Init(x, y, &tilingData);
                op.Process();
            }
        } else if (TILING_KEY_IS(REVERSE_V2_DIM6_N_Y_UINT32)) {
            if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
                ReverseV2Simt<uint32_t, int8_t, false, DIM_NUM_6> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int16_t)) {
                ReverseV2Simt<uint32_t, int16_t, false, DIM_NUM_6> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
                ReverseV2Simt<uint32_t, int32_t, false, DIM_NUM_6> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
                ReverseV2Simt<uint32_t, int64_t, false, DIM_NUM_6> op;
                op.Init(x, y, &tilingData);
                op.Process();
            }
        } else if (TILING_KEY_IS(REVERSE_V2_DIM6_Y_N_UINT64)) {
            if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
                ReverseV2Simt<uint64_t, int8_t, true, DIM_NUM_6> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int16_t)) {
                ReverseV2Simt<uint64_t, int16_t, true, DIM_NUM_6> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
                ReverseV2Simt<uint64_t, int32_t, true, DIM_NUM_6> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
                ReverseV2Simt<uint64_t, int64_t, true, DIM_NUM_6> op;
                op.Init(x, y, &tilingData);
                op.Process();
            }
        } else if (TILING_KEY_IS(REVERSE_V2_DIM6_N_Y_UINT64)) {
            if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
                ReverseV2Simt<uint64_t, int8_t, false, DIM_NUM_6> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int16_t)) {
                ReverseV2Simt<uint64_t, int16_t, false, DIM_NUM_6> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
                ReverseV2Simt<uint64_t, int32_t, false, DIM_NUM_6> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
                ReverseV2Simt<uint64_t, int64_t, false, DIM_NUM_6> op;
                op.Init(x, y, &tilingData);
                op.Process();
            }
        } else if (TILING_KEY_IS(REVERSE_V2_DIM5_Y_N_UINT32)) {
            if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
                ReverseV2Simt<uint32_t, int8_t, true, DIM_NUM_5> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int16_t)) {
                ReverseV2Simt<uint32_t, int16_t, true, DIM_NUM_5> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
                ReverseV2Simt<uint32_t, int32_t, true, DIM_NUM_5> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
                ReverseV2Simt<uint32_t, int64_t, true, DIM_NUM_5> op;
                op.Init(x, y, &tilingData);
                op.Process();
            }
        } else if (TILING_KEY_IS(REVERSE_V2_DIM5_N_Y_UINT32)) {
            if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
                ReverseV2Simt<uint32_t, int8_t, false, DIM_NUM_5> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int16_t)) {
                ReverseV2Simt<uint32_t, int16_t, false, DIM_NUM_5> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
                ReverseV2Simt<uint32_t, int32_t, false, DIM_NUM_5> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
                ReverseV2Simt<uint32_t, int64_t, false, DIM_NUM_5> op;
                op.Init(x, y, &tilingData);
                op.Process();
            }
        } else if (TILING_KEY_IS(REVERSE_V2_DIM5_Y_N_UINT64)) {
            if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
                ReverseV2Simt<uint64_t, int8_t, true, DIM_NUM_5> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int16_t)) {
                ReverseV2Simt<uint64_t, int16_t, true, DIM_NUM_5> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
                ReverseV2Simt<uint64_t, int32_t, true, DIM_NUM_5> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
                ReverseV2Simt<uint64_t, int64_t, true, DIM_NUM_5> op;
                op.Init(x, y, &tilingData);
                op.Process();
            }
        } else if (TILING_KEY_IS(REVERSE_V2_DIM5_N_Y_UINT64)) {
            if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
                ReverseV2Simt<uint64_t, int8_t, false, DIM_NUM_5> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int16_t)) {
                ReverseV2Simt<uint64_t, int16_t, false, DIM_NUM_5> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
                ReverseV2Simt<uint64_t, int32_t, false, DIM_NUM_5> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
                ReverseV2Simt<uint64_t, int64_t, false, DIM_NUM_5> op;
                op.Init(x, y, &tilingData);
                op.Process();
            }
        } else if (TILING_KEY_IS(REVERSE_V2_DIM4_Y_N_UINT32)) {
            if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
                ReverseV2Simt<uint32_t, int8_t, true, DIM_NUM_4> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int16_t)) {
                ReverseV2Simt<uint32_t, int16_t, true, DIM_NUM_4> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
                ReverseV2Simt<uint32_t, int32_t, true, DIM_NUM_4> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
                ReverseV2Simt<uint32_t, int64_t, true, DIM_NUM_4> op;
                op.Init(x, y, &tilingData);
                op.Process();
            }
        } else if (TILING_KEY_IS(REVERSE_V2_DIM4_N_Y_UINT32)) {
            if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
                ReverseV2Simt<uint32_t, int8_t, false, DIM_NUM_4> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int16_t)) {
                ReverseV2Simt<uint32_t, int16_t, false, DIM_NUM_4> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
                ReverseV2Simt<uint32_t, int32_t, false, DIM_NUM_4> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
                ReverseV2Simt<uint32_t, int64_t, false, DIM_NUM_4> op;
                op.Init(x, y, &tilingData);
                op.Process();
            }
        } else if (TILING_KEY_IS(REVERSE_V2_DIM4_Y_N_UINT64)) {
            if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
                ReverseV2Simt<uint64_t, int8_t, true, DIM_NUM_4> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int16_t)) {
                ReverseV2Simt<uint64_t, int16_t, true, DIM_NUM_4> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
                ReverseV2Simt<uint64_t, int32_t, true, DIM_NUM_4> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
                ReverseV2Simt<uint64_t, int64_t, true, DIM_NUM_4> op;
                op.Init(x, y, &tilingData);
                op.Process();
            }
        } else if (TILING_KEY_IS(REVERSE_V2_DIM4_N_Y_UINT64)) {
            if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
                ReverseV2Simt<uint64_t, int8_t, false, DIM_NUM_4> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int16_t)) {
                ReverseV2Simt<uint64_t, int16_t, false, DIM_NUM_4> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
                ReverseV2Simt<uint64_t, int32_t, false, DIM_NUM_4> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
                ReverseV2Simt<uint64_t, int64_t, false, DIM_NUM_4> op;
                op.Init(x, y, &tilingData);
                op.Process();
            }
        } else if (TILING_KEY_IS(REVERSE_V2_DIM3_Y_N_UINT32)) {
            if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
                ReverseV2Simt<uint32_t, int8_t, true, DIM_NUM_3> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int16_t)) {
                ReverseV2Simt<uint32_t, int16_t, true, DIM_NUM_3> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
                ReverseV2Simt<uint32_t, int32_t, true, DIM_NUM_3> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
                ReverseV2Simt<uint32_t, int64_t, true, DIM_NUM_3> op;
                op.Init(x, y, &tilingData);
                op.Process();
            }
        } else if (TILING_KEY_IS(REVERSE_V2_DIM3_N_Y_UINT32)) {
            if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
                ReverseV2Simt<uint32_t, int8_t, false, DIM_NUM_3> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int16_t)) {
                ReverseV2Simt<uint32_t, int16_t, false, DIM_NUM_3> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
                ReverseV2Simt<uint32_t, int32_t, false, DIM_NUM_3> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
                ReverseV2Simt<uint32_t, int64_t, false, DIM_NUM_3> op;
                op.Init(x, y, &tilingData);
                op.Process();
            }
        } else if (TILING_KEY_IS(REVERSE_V2_DIM3_Y_N_UINT64)) {
            if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
                ReverseV2Simt<uint64_t, int8_t, true, DIM_NUM_3> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int16_t)) {
                ReverseV2Simt<uint64_t, int16_t, true, DIM_NUM_3> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
                ReverseV2Simt<uint64_t, int32_t, true, DIM_NUM_3> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
                ReverseV2Simt<uint64_t, int64_t, true, DIM_NUM_3> op;
                op.Init(x, y, &tilingData);
                op.Process();
            }
        } else if (TILING_KEY_IS(REVERSE_V2_DIM3_N_Y_UINT64)) {
            if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
                ReverseV2Simt<uint64_t, int8_t, false, DIM_NUM_3> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int16_t)) {
                ReverseV2Simt<uint64_t, int16_t, false, DIM_NUM_3> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
                ReverseV2Simt<uint64_t, int32_t, false, DIM_NUM_3> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
                ReverseV2Simt<uint64_t, int64_t, false, DIM_NUM_3> op;
                op.Init(x, y, &tilingData);
                op.Process();
            }
        } else if (TILING_KEY_IS(REVERSE_V2_DIM2_Y_N_UINT32)) {
            if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
                ReverseV2Simt<uint32_t, int8_t, true, DIM_NUM_2> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int16_t)) {
                ReverseV2Simt<uint32_t, int16_t, true, DIM_NUM_2> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
                ReverseV2Simt<uint32_t, int32_t, true, DIM_NUM_2> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
                ReverseV2Simt<uint32_t, int64_t, true, DIM_NUM_2> op;
                op.Init(x, y, &tilingData);
                op.Process();
            }
        } else if (TILING_KEY_IS(REVERSE_V2_DIM2_N_Y_UINT32)) {
            if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
                ReverseV2Simt<uint32_t, int8_t, false, DIM_NUM_2> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int16_t)) {
                ReverseV2Simt<uint32_t, int16_t, false, DIM_NUM_2> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
                ReverseV2Simt<uint32_t, int32_t, false, DIM_NUM_2> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
                ReverseV2Simt<uint32_t, int64_t, false, DIM_NUM_2> op;
                op.Init(x, y, &tilingData);
                op.Process();
            }
        } else if (TILING_KEY_IS(REVERSE_V2_DIM2_Y_N_UINT64)) {
            if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
                ReverseV2Simt<uint64_t, int8_t, true, DIM_NUM_2> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int16_t)) {
                ReverseV2Simt<uint64_t, int16_t, true, DIM_NUM_2> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
                ReverseV2Simt<uint64_t, int32_t, true, DIM_NUM_2> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
                ReverseV2Simt<uint64_t, int64_t, true, DIM_NUM_2> op;
                op.Init(x, y, &tilingData);
                op.Process();
            }
        } else if (TILING_KEY_IS(REVERSE_V2_DIM2_N_Y_UINT64)) {
            if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
                ReverseV2Simt<uint64_t, int8_t, false, DIM_NUM_2> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int16_t)) {
                ReverseV2Simt<uint64_t, int16_t, false, DIM_NUM_2> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
                ReverseV2Simt<uint64_t, int32_t, false, DIM_NUM_2> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
                ReverseV2Simt<uint64_t, int64_t, false, DIM_NUM_2> op;
                op.Init(x, y, &tilingData);
                op.Process();
            }
        } else if (TILING_KEY_IS(REVERSE_V2_DIM1_Y_N_UINT32)) {
            if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
                ReverseV2Simt<uint32_t, int8_t, true, DIM_NUM_1> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int16_t)) {
                ReverseV2Simt<uint32_t, int16_t, true, DIM_NUM_1> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
                ReverseV2Simt<uint32_t, int32_t, true, DIM_NUM_1> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
                ReverseV2Simt<uint32_t, int64_t, true, DIM_NUM_1> op;
                op.Init(x, y, &tilingData);
                op.Process();
            }
        } else if (TILING_KEY_IS(REVERSE_V2_DIM1_Y_N_UINT64)) {
            if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
                ReverseV2Simt<uint64_t, int8_t, true, DIM_NUM_1> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int16_t)) {
                ReverseV2Simt<uint64_t, int16_t, true, DIM_NUM_1> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
                ReverseV2Simt<uint64_t, int32_t, true, DIM_NUM_1> op;
                op.Init(x, y, &tilingData);
                op.Process();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
                ReverseV2Simt<uint64_t, int64_t, true, DIM_NUM_1> op;
                op.Init(x, y, &tilingData);
                op.Process();
            }
        } else if (TILING_KEY_IS(REVERSE_V2_SIMD_TILING_KEY)) {
            ReverseV2Simd op;
            op.Init(x, y, &tilingData, &pipe);
            op.Process();
        }
     }
 }
 
