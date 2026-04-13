/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file unsorted_segment_sum_tiling_arch35.cpp
 * \brief unsorted segment sum tiling
 */
 #include "unsorted_segment_sum_tiling_arch35.h"
 #include "unsorted_segment_sum_tiling.h"
 #include "util/platform_util.h"


 namespace optiling {
 using namespace ge;

 const size_t INPUT_DATA_IDX = 0;
 const size_t INPUT_SEGMENT_IDS_IDX = 1;
 const size_t INPUT_NUM_SEGMENTS_IDX = 2;

 const std::string UNSORTED_SEGMENT_SUM_OP_TYPE = "UnsortedSegmentSum";
 const int32_t BYTE_BLOCK = 32;
 const int32_t MIN_ELE_SIZE_USING_ALL_CORE = 1024;
 const int32_t MASK_FP32 = 64;
 const int32_t MASK_INT32 = 64;
 const int32_t MASK_FP16 = 128;
 const int32_t MAX_REPEAT_TIME = 255;
 const int32_t FP32_ELE_NUM_ALIGN_32B = 8;
 const int32_t BYTE_FULL_MASK = 256;
 const int32_t MULTI = 4;
 const int32_t FP16_BLOCK_NUM = 16;
 const int32_t INT32_BLOCK_NUM = 8;
 const int32_t INT32_MIN_NUM = -2147483648;
 const int32_t EACH_CORE_NUM_UNIT = 64;
 const int32_t MASK_NUM_UNIT = 255;

 // dtype
 const std::string DTYPE_FP32 = "float32";
 const std::string DTYPE_FP16 = "float16";
 const std::string DTYPE_INT32 = "int32";
 const std::string DTYPE_INT8 = "int8";
 const std::string DTYPE_UINT8 = "uint8";

 // ub tensor num
 const int32_t UB_TENSOR_NUM_FP32_INPUT_LAST_AXIS_ALIGN = 2;
 const int32_t UB_TENSOR_NUM_FP32_INPUT_LAST_AXIS_ONE = 6;
 const int32_t UB_TENSOR_NUM_FP32_INPUT_ONE_DIM = 3;
 const int32_t UB_TENSOR_NUM_FP32_INPUT_LAST_AXIS_NOT_ALIGN = 3;
 const int32_t UB_TENSOR_NUM_FP32_PAD = 6;
 const int32_t PAD_ONCE = 128;
 const int32_t PAD_LASE_AXIS = 168;
 const int32_t UB_TENSOR_NUM_FP32_INPUT_LAST_AXIS_NOT_ALIGN_HP = 3;

 // fp32 select key
 const int32_t SELECT_KEY_MODE_FP32_INPUT_LAST_AXIS_ALIGN_SMALL_E = 1;
 const int32_t SELECT_KEY_MODE_FP32_INPUT_LAST_AXIS_ONE = 2;
 const int32_t SELECT_KEY_MODE_FP32_INPUT_ONE_DIM = 2;
 const int32_t SELECT_KEY_MODE_FP32_INPUT_LAST_AXIS_NOT_ALIGN_SMALL_E = 4;
 const int32_t SELECT_KEY_MODE_FP32_INPUT_LAST_AXIS_ALIGN_BIG_E = 5;
 const int32_t SELECT_KEY_MODE_FP32_INPUT_LAST_AXIS_NOT_ALIGN_BIG_E = 6;
 const int32_t SELECT_KEY_MODE_FP32_INPUT_LAST_AXIS_ONE_MODIFY = 7;
 const int32_t SELECT_KEY_MODE_FP32_INPUT_LAST_AXIS_ONE_MULTI = 8;
 const int32_t SELECT_KEY_MODE_FP32_INPUT_NUM_SEGMENT_ONE = 17;
 const int32_t SELECT_KEY_MODE_FP32_INPUT_LAST_AXIS_ALIGN_SMALL_E_HP = 18;
 const int32_t SELECT_KEY_MODE_FP32_INPUT_LAST_AXIS_NOT_ALIGN_SMALL_E_HP = 19;
 const int32_t SELECT_KEY_MODE_FP32_INPUT_LAST_AXIS_NOT_ALIGN_SMALL_E_HP_PAD = 20;

 // int32 select key
 const int32_t SELECT_KEY_MODE_NO_ATOMIC_SMALL_E_SMALL_ID = 9;
 const int32_t SELECT_KEY_MODE_NO_ATOMIC_SMALL_E_BIG_ID = 10;
 const int32_t SELECT_KEY_MODE_NO_ATOMIC_BIG_E_SMALL_ID = 11;
 const int32_t SELECT_KEY_MODE_NO_ATOMIC_BIG_E_BIG_ID = 12;
 const int32_t SELECT_KEY_MODE_NO_ATOMIC_SMALL_E_SMALL_ID_SMALLBLOCK = 13;
 const int32_t SELECT_KEY_MODE_NO_ATOMIC_SMALL_E_BIG_ID_SMALLBLOCK = 14;
 const int32_t SELECT_KEY_MODE_NO_ATOMIC_NUM_SEGMENT_ONE = 15;
 const int32_t SELECT_KEY_MODE_NO_ATOMIC_ALL_IN_ALIGN = 16;

 const int32_t PARA_CORE_NUM_0 = 0;
 const int32_t PARA_UB_SIZE_1 = 1;
 const int32_t PARA_UB_TENSOR_NUM_2 = 2;
 const int32_t PARA_IMPL_MODE_3 = 3;
 const int32_t UB_TENSOR_NUM_2 = 2;
 const int32_t UB_TENSOR_MAX_X_NUM = 16000;
 const int32_t PART_TWO = 2;
 const int32_t PART_THREE = 3;
 const int32_t PART_FOUR = 4;
 const int32_t PART_FIVE = 5;
 const int32_t PART_SIX = 6;
 const int32_t PART_SERVERN = 7;


 enum EleByte { FP16_BYTE = 2, FP32_BYTE = 4, INT32_BYTE = 4, INT64_BYTE = 8, INT8_BYTE = 1, UINT8_BYTE = 1 };
 enum ParamsMode { FRONT_CORE = 0, LAST_CORE = 1 };
 enum InputDataParams {
   FRONT_PART_FRONT_CORE = 0,
   LAST_PART_FRONT_CORE = 1,
   FRONT_PART_LAST_CORE = 2,
   LAST_PART_LAST_CORE = 3
 };

 struct CommParas {
   int32_t e_num;
   int32_t x_e_num;
   int32_t num_segments;
   int32_t ub_size;
   int32_t impl_mode;
   int32_t ub_tensor_ele_num;
   int32_t ub_tensor_ele_num_pad;
   int32_t e_size_align;
   bool is_pad;
 };

 struct CommParas4Int32 {
   ge::DataType input_dytpe;
   EleByte ele_byte;
   int32_t e_once_num;
   int32_t all_size;
   int32_t num_segments;
 };

 struct CommParas4UbSizeNoAtomic {
   ge::DataType input_dtype;
   int32_t ub_size;
   int32_t e_size;
   int32_t output_ub_ele_num_one_row;
   int32_t need_core_num;
   int32_t mask;
   int32_t num_segments;
   int32_t ub_tensor_size_id;
   int32_t ub_tensor_size_input;
   int32_t ub_tensor_size_outpu;
 };

 struct CommParas4TilingModeNoAtomic {
   ge::DataType input_dtype;
   int32_t e_size;
   int32_t ids_size;
   int32_t ub_tensor_size;
   int32_t ub_tensor_size_input;
   int32_t need_core;
   int32_t output_ub_ele_num_one_row;
   int32_t all_size;
   int32_t num_segments;
 };

 struct InputParamsGm2UbOut {
   int32_t mode;
   int32_t input_mov_times_gm2ub;
   int32_t input_front_burst_len;
   int32_t input_last_burst_len;
   int32_t input_ele_num_ub_front_part;
   int32_t input_ele_num_ub_last_part;
   int32_t input_rows_front_part;
   int32_t input_rows_last_part;
 };

 /******************COMMON_FUNCTION******************/
 static int32_t ComputeDivRemainders(const int32_t& num, const int32_t& factor, const int32_t& times) {
   int32_t res = num - factor * times;
   return res;
 }

 static int32_t UssCeil(const int32_t& num, const int32_t& factor) {
   if (factor == 0) {
     return num;
   }

   int32_t res = (num % factor == 0) ? num : factor * (num / factor + 1);
   return res;
 }

 static int32_t UssCeilDiv(const int32_t& num, const int32_t& factor) {
   if (factor == 0) {
     return num;
   }

   int32_t res = (num % factor == 0) ? num / factor : num / factor + 1;
   return res;
 }

 static int32_t UssCeilDivNoAtomic(const int32_t& num, const int32_t& factor, const int32_t& e_size,
                                   const int32_t& output_ub_ele_num_one_row) {
   int32_t res = 0;

   if (factor == 0) {
     res = num;
   } else {
     res = num / factor;
   }

   if (factor > 1 && e_size < output_ub_ele_num_one_row) {
     if (e_size * res % output_ub_ele_num_one_row != 0) {
       res = res / output_ub_ele_num_one_row * output_ub_ele_num_one_row;
     }
     int32_t last = num - (factor - 1) * res;
     if (e_size * last < output_ub_ele_num_one_row) {
       last = output_ub_ele_num_one_row / e_size + 1;
       res = (num - last) / (factor - 1);
       if (res < 1) {
         res = 1;
       }
       res = e_size * res / output_ub_ele_num_one_row * output_ub_ele_num_one_row / e_size;
     }
   }
   return res;
 }

 void ComputeDiv(const int32_t& num, const int32_t& factor, int32_t& front_part_num, int32_t& last_part) {
   front_part_num = num / factor;
   last_part = num - front_part_num * factor;
   return;
 }

 static bool CheckHighPerf(const int32_t& ub_size, const int32_t& e_size, const int32_t& impl_mode) {
   int32_t e_size_align8 = UssCeil(e_size, FP32_ELE_NUM_ALIGN_32B);
   int32_t x_ub_size = ub_size / UB_TENSOR_NUM_2 / BYTE_BLOCK * BYTE_BLOCK;
   if ((impl_mode == 1) && (((e_size_align8 * UB_TENSOR_NUM_2 + FP32_ELE_NUM_ALIGN_32B) * FP32_BYTE) < x_ub_size)) {
     return true;
   }
   return false;
 }
 static bool IsPad(const int32_t& ub_size, const int32_t& e_size, const int32_t& impl_mode) {
   if (e_size % FP32_ELE_NUM_ALIGN_32B == 0) {
     return false;
   }
   int32_t e_size_align8 = UssCeil(e_size, FP32_ELE_NUM_ALIGN_32B);
   int32_t x_ub_size = ub_size / UB_TENSOR_NUM_FP32_PAD / BYTE_BLOCK * BYTE_BLOCK;
   int32_t num = x_ub_size / MULTI;
   if ((impl_mode == 1) && (e_size < PAD_LASE_AXIS && PAD_ONCE * e_size_align8 < num)) {
     return true;
   }
   return false;
 }

 static bool GetTilingMode(const CommParas& commParas, const int32_t& x_ub_size, int32_t& selectKey) {
   int32_t e_num = commParas.e_num;
   int32_t x_e_num = commParas.x_e_num;
   int32_t num_segments = commParas.num_segments;
   int32_t ub_size = commParas.ub_size;
   int32_t x_ub_ele_num = x_ub_size / FP32_BYTE;
   int32_t impl_mode = commParas.impl_mode;
   bool is_pad = commParas.is_pad;

   bool isHighPerf = CheckHighPerf(ub_size, e_num, impl_mode);

   if (num_segments == 1) {
     selectKey = SELECT_KEY_MODE_FP32_INPUT_NUM_SEGMENT_ONE;
     return true;
   }

   if (num_segments > 1 && x_e_num >= 1) {
     if (e_num % FP32_ELE_NUM_ALIGN_32B == 0) {
       if (e_num < x_ub_ele_num) {
         selectKey = isHighPerf ? SELECT_KEY_MODE_FP32_INPUT_LAST_AXIS_ALIGN_SMALL_E_HP
                                : SELECT_KEY_MODE_FP32_INPUT_LAST_AXIS_ALIGN_SMALL_E;
         return true;
       }
       selectKey = SELECT_KEY_MODE_FP32_INPUT_LAST_AXIS_ALIGN_BIG_E;
       return true;
     }
     if (e_num < x_ub_ele_num) {
       if (isHighPerf && is_pad) {
         selectKey = SELECT_KEY_MODE_FP32_INPUT_LAST_AXIS_NOT_ALIGN_SMALL_E_HP_PAD;
         return true;
       }
       selectKey = isHighPerf ? SELECT_KEY_MODE_FP32_INPUT_LAST_AXIS_NOT_ALIGN_SMALL_E_HP
                              : SELECT_KEY_MODE_FP32_INPUT_LAST_AXIS_NOT_ALIGN_SMALL_E;
       return true;
     }
     selectKey = SELECT_KEY_MODE_FP32_INPUT_LAST_AXIS_NOT_ALIGN_BIG_E;
     return true;
   }

   return false;
 }

 static bool GetTilingModeNoAtomic(const CommParas4TilingModeNoAtomic comm_params, int32_t& select_key,
                                   int32_t& e_once_num, int32_t& id_once_num, int32_t& num_segment_max) {
   int input_byte = 0;
   input_byte = (comm_params.input_dtype == ge::DT_FLOAT16) ? FP16_BYTE : FP32_BYTE;

   int32_t e_once_ubsize = (comm_params.ub_tensor_size_input / BYTE_FULL_MASK) * BYTE_FULL_MASK;
   e_once_num = e_once_ubsize / input_byte;
   id_once_num = ((comm_params.ub_tensor_size / BYTE_BLOCK) * BYTE_BLOCK) / INT32_BYTE;
   num_segment_max = ((comm_params.ub_tensor_size / BYTE_BLOCK) * BYTE_BLOCK) / input_byte;
   num_segment_max = num_segment_max / comm_params.e_size;

   if (comm_params.num_segments == 1) {
     select_key = SELECT_KEY_MODE_NO_ATOMIC_NUM_SEGMENT_ONE;

     return true;
   } else if (comm_params.e_size > e_once_num && comm_params.ids_size > id_once_num) {
     // e big id big
     select_key = SELECT_KEY_MODE_NO_ATOMIC_BIG_E_BIG_ID;

     return true;
   } else if (comm_params.e_size > e_once_num && comm_params.ids_size <= id_once_num) {
     // e nig id small
     select_key = SELECT_KEY_MODE_NO_ATOMIC_BIG_E_SMALL_ID;

     return true;
   } else if (comm_params.e_size <= e_once_num && comm_params.ids_size <= id_once_num) {
     // e small id small
     select_key = SELECT_KEY_MODE_NO_ATOMIC_SMALL_E_SMALL_ID;
     if (comm_params.need_core > 1 && comm_params.e_size < comm_params.output_ub_ele_num_one_row) {
       select_key = SELECT_KEY_MODE_NO_ATOMIC_SMALL_E_SMALL_ID_SMALLBLOCK;
     } else if (comm_params.e_size % comm_params.output_ub_ele_num_one_row == 0 && comm_params.all_size < e_once_num) {
       select_key = SELECT_KEY_MODE_NO_ATOMIC_ALL_IN_ALIGN;
     }

     return true;
   } else if (comm_params.ids_size > id_once_num && comm_params.e_size <= e_once_num) {
     // e small id big
     select_key = SELECT_KEY_MODE_NO_ATOMIC_SMALL_E_BIG_ID;
     if (comm_params.need_core > 1 && comm_params.e_size < comm_params.output_ub_ele_num_one_row) {
       select_key = SELECT_KEY_MODE_NO_ATOMIC_SMALL_E_BIG_ID_SMALLBLOCK;
     }

     return true;
   }

   return false;
 }

 static bool GetEleDtype(const ge::DataType& dtype, EleByte& elebyte) {
   if (dtype == ge::DT_FLOAT) {
     elebyte = FP32_BYTE;
     return true;
   } else if (dtype == ge::DT_FLOAT16) {
     elebyte = FP16_BYTE;
     return true;
   } else if (dtype == ge::DT_INT32) {
     elebyte = INT32_BYTE;
     return true;
   } else if (dtype == ge::DT_INT64) {
     elebyte = INT64_BYTE;
     return true;
   } else if (dtype == ge::DT_INT8) {
     elebyte = INT8_BYTE;
     return true;
   } else if (dtype == ge::DT_UINT8) {
     elebyte = UINT8_BYTE;
     return true;
   }
   return false;
 }

 static ge::graphStatus CalcNeededCoreNum(const gert::TilingContext* context, const CommParas& commParas,
                                          const int32_t& ids_size, const int32_t& core_num, int32_t& need_core_num) {
   OP_CHECK_IF(
       core_num == 0,
       OP_LOGE(context->GetNodeName(), "CalcNeededCoreNum check param failed, core_num is 0"),
       return ge::GRAPH_FAILED);
   int32_t ele_num = ids_size / core_num;

   if (commParas.num_segments > 1) {
     if (commParas.e_num > 1) {
       need_core_num = (ele_num >= 1) ? core_num : ids_size;
     } else {
       if (ids_size <= EACH_CORE_NUM_UNIT || ele_num <= 0) {
         need_core_num = 1;
       } else {
         need_core_num = (ele_num >= EACH_CORE_NUM_UNIT) ? core_num : ids_size / EACH_CORE_NUM_UNIT;
       }
     }

     return ge::GRAPH_SUCCESS;
   } else {
     need_core_num = (ele_num >= 1) ? core_num : ids_size;

     return ge::GRAPH_SUCCESS;
   }

   need_core_num = 1;

   return ge::GRAPH_SUCCESS;
 }

 static ge::graphStatus CalcNeededCoreByNumSegments(const gert::TilingContext* context,
                                                    const CommParas4UbSizeNoAtomic comm_ub_size_params,
                                                    const int32_t& core_num, int32_t& need_core_num) {
   OP_CHECK_IF(core_num == 0,
                   OP_LOGE(context->GetNodeName(),
                                                   "CalcNeededCoreByNumSegments check param failed, core_num is 0"),
                   return ge::GRAPH_FAILED);
   int32_t ele_num = comm_ub_size_params.num_segments / core_num;

   if (comm_ub_size_params.num_segments == 1) {
     if (comm_ub_size_params.e_size <= comm_ub_size_params.output_ub_ele_num_one_row) {
       need_core_num = 1;
     } else {
       OP_CHECK_IF(
           comm_ub_size_params.output_ub_ele_num_one_row == 0,
           OP_LOGE(
               context->GetNodeName(), "CalcNeededCoreByNumSegments check param failed, output_ub_ele_num_one_row is 0"),
           return ge::GRAPH_FAILED);
       int32_t core_one = comm_ub_size_params.e_size / comm_ub_size_params.output_ub_ele_num_one_row;
       if (core_one >= core_num) {
         need_core_num = core_num;
       } else {
         need_core_num =
             (comm_ub_size_params.e_size % comm_ub_size_params.output_ub_ele_num_one_row == 0) ? core_one : core_one + 1;
       }
     }

     return ge::GRAPH_SUCCESS;
   }
   if (comm_ub_size_params.e_size < comm_ub_size_params.output_ub_ele_num_one_row &&
       ele_num < comm_ub_size_params.output_ub_ele_num_one_row) {
     need_core_num = 1;

     return ge::GRAPH_SUCCESS;
   }
   if (comm_ub_size_params.e_size >= comm_ub_size_params.output_ub_ele_num_one_row) {
     need_core_num = (ele_num >= 1) ? core_num : comm_ub_size_params.num_segments;

     return ge::GRAPH_SUCCESS;
   }
   if (comm_ub_size_params.e_size < comm_ub_size_params.output_ub_ele_num_one_row &&
       ele_num >= comm_ub_size_params.output_ub_ele_num_one_row) {
     need_core_num = core_num;

     return ge::GRAPH_SUCCESS;
   }

   need_core_num = 1;
   return ge::GRAPH_SUCCESS;
 }

 static void ComputeUbTensorSizeNoAtomic(const CommParas4UbSizeNoAtomic comm_params, int32_t& ub_tensor_size_id,
                                         int32_t& ub_tensor_size_input, int32_t& ub_tensor_size_output) {
   int32_t input_ele_byte = (comm_params.input_dtype == ge::DT_FLOAT16) ? FP16_BYTE : INT32_BYTE;
   if (comm_params.num_segments == 1) {
     ub_tensor_size_id = comm_params.ub_size / UB_TENSOR_NUM_2;
     ub_tensor_size_input = ub_tensor_size_id;
     ub_tensor_size_output = ub_tensor_size_id;
   } else if (comm_params.need_core_num > 1 && comm_params.e_size < comm_params.output_ub_ele_num_one_row) {
     int32_t ub_tensor_num = UB_TENSOR_NUM_2;
     ub_tensor_size_input = comm_params.mask * input_ele_byte;
     ub_tensor_size_id = (comm_params.ub_size - ub_tensor_size_input) / ub_tensor_num;
     ub_tensor_size_output = ub_tensor_size_id;
   } else {
     ub_tensor_size_input = UB_TENSOR_MAX_X_NUM * input_ele_byte;
     ub_tensor_size_output = ub_tensor_size_input;
     ub_tensor_size_id = comm_params.ub_size - UB_TENSOR_NUM_2 * ub_tensor_size_input;
   }

   return;
 }

 static void NumSegmentOne(const int32_t& max_ele_num_one_ub_tensor, const int32_t& mask, const EleByte& ele_byte,
                           const int32_t& mode, UnsortedSegmentSumTilingParamsInt32* tiling_params) {
   int32_t e_mov_times_gm2ub_input_scalar = 0;
   int32_t e_ub2gm_front_burst_len_input_scalar = 0;
   int32_t e_ub2gm_last_burst_len_input_scalar = 0;
   int32_t repeat_times_last_part = 0;
   int32_t num_segments_front_core_input_scalar = 0;
   int32_t repeat_time_last_part_input_scalar = 0;

   if (mode == FRONT_CORE) {
     e_mov_times_gm2ub_input_scalar = tiling_params->e_mov_times_gm2ub_input_scalar;
     num_segments_front_core_input_scalar = tiling_params->num_segments_front_core_input_scalar;
   } else {
     e_mov_times_gm2ub_input_scalar = tiling_params->e_mov_times_gm2ub_input_scalar_lastcore;
     num_segments_front_core_input_scalar = tiling_params->num_segments_last_core_input_scalar;
   }

   if (e_mov_times_gm2ub_input_scalar > 1) {
     tiling_params->e_num_front_part_input_scalar = max_ele_num_one_ub_tensor;
     e_ub2gm_front_burst_len_input_scalar =
         UssCeilDiv(tiling_params->e_num_front_part_input_scalar * ele_byte, BYTE_BLOCK);
     tiling_params->repeat_times = UssCeilDiv(tiling_params->e_num_front_part_input_scalar, mask * MASK_NUM_UNIT);

     tiling_params->repeat_time_front_part_input_scalar = UssCeilDiv(
         tiling_params->e_num_front_part_input_scalar - (tiling_params->repeat_times - 1) * mask * MASK_NUM_UNIT, mask);

     tiling_params->e_num_last_part_input_scalar =
         ComputeDivRemainders(num_segments_front_core_input_scalar, tiling_params->e_num_front_part_input_scalar,
                              e_mov_times_gm2ub_input_scalar - 1);
     e_ub2gm_last_burst_len_input_scalar =
         UssCeilDiv(tiling_params->e_num_last_part_input_scalar * ele_byte, BYTE_BLOCK);

     repeat_times_last_part = UssCeilDiv(tiling_params->e_num_last_part_input_scalar, mask * MASK_NUM_UNIT);
     if (repeat_times_last_part > 1) {
       repeat_time_last_part_input_scalar = UssCeilDiv(
           tiling_params->e_num_last_part_input_scalar - (repeat_times_last_part - 1) * mask * MASK_NUM_UNIT, mask);
     } else {
       repeat_time_last_part_input_scalar = UssCeilDiv(tiling_params->e_num_last_part_input_scalar, mask);
     }
   } else {
     tiling_params->e_num_front_part_input_scalar = num_segments_front_core_input_scalar;

     e_ub2gm_front_burst_len_input_scalar =
         UssCeilDiv(tiling_params->e_num_front_part_input_scalar * ele_byte, BYTE_BLOCK);
     tiling_params->repeat_times = UssCeilDiv(tiling_params->e_num_front_part_input_scalar, mask * MASK_NUM_UNIT);
     if (tiling_params->repeat_times > 1) {
       tiling_params->repeat_time_front_part_input_scalar = UssCeilDiv(
           tiling_params->e_num_front_part_input_scalar - (tiling_params->repeat_times - 1) * mask * MASK_NUM_UNIT,
           mask);
     } else {
       tiling_params->repeat_time_front_part_input_scalar =
           UssCeilDiv(tiling_params->e_num_front_part_input_scalar, mask);
     }
     repeat_time_last_part_input_scalar = tiling_params->repeat_time_front_part_input_scalar;
     tiling_params->e_num_last_part_input_scalar = tiling_params->e_num_front_part_input_scalar;
     e_ub2gm_last_burst_len_input_scalar = e_ub2gm_front_burst_len_input_scalar;
     repeat_times_last_part = tiling_params->repeat_times;
   }

   if (mode == FRONT_CORE) {
     tiling_params->e_ub2gm_front_burst_len_input_scalar = e_ub2gm_front_burst_len_input_scalar;
     tiling_params->e_ub2gm_last_burst_len_input_scalar = e_ub2gm_last_burst_len_input_scalar;
     tiling_params->repeat_times_last_part = repeat_times_last_part;
     tiling_params->repeat_time_last_part_input_scalar = repeat_time_last_part_input_scalar;
   } else {
     tiling_params->e_ub2gm_front_burst_len_input_scalar_lastcore = e_ub2gm_front_burst_len_input_scalar;
     tiling_params->e_ub2gm_last_burst_len_input_scalar_lastcore = e_ub2gm_last_burst_len_input_scalar;
     tiling_params->repeat_times_last_part_lastcore = repeat_times_last_part;
     tiling_params->repeat_time_front_part_input_scalar_lastcore = repeat_time_last_part_input_scalar;
   }

   return;
 }

 static void ComputeUbTensorSize(CommParas& commParas, int32_t& x_ub_size, int32_t& ids_ub_size,
                                 int32_t& x_ub_size_pad) {
   int32_t e_num = commParas.e_num;
   int32_t num_segments = commParas.num_segments;
   int32_t ub_size = commParas.ub_size;
   int32_t impl_mode = commParas.impl_mode;

   bool isHighPerf = CheckHighPerf(ub_size, e_num, impl_mode);
   commParas.is_pad = IsPad(ub_size, e_num, impl_mode);
   if (num_segments == 1) {
     ids_ub_size = ub_size / BYTE_BLOCK * BYTE_BLOCK;
     x_ub_size = ids_ub_size;
     return;
   }

   int32_t ub_tensor_num = 0;
   int32_t ids_ub_size_perf = 0;
   // align
   if (e_num % FP32_ELE_NUM_ALIGN_32B == 0) {
     ub_tensor_num = UB_TENSOR_NUM_FP32_INPUT_LAST_AXIS_ALIGN;
     ids_ub_size = ((ub_size / ub_tensor_num) / BYTE_BLOCK) * BYTE_BLOCK;
     x_ub_size = ids_ub_size;
     ids_ub_size_perf = ((ub_size / UB_TENSOR_NUM_FP32_INPUT_LAST_AXIS_ONE) / BYTE_BLOCK) * BYTE_BLOCK;
     if (isHighPerf) {
       x_ub_size = ids_ub_size_perf;
       ids_ub_size = ids_ub_size_perf;
     }
     return;
   }

   // not align
   ub_tensor_num = UB_TENSOR_NUM_FP32_INPUT_LAST_AXIS_NOT_ALIGN;
   ids_ub_size = ((ub_size / ub_tensor_num) / BYTE_BLOCK) * BYTE_BLOCK;
   x_ub_size = ids_ub_size;
   if (isHighPerf && commParas.is_pad) {
     ids_ub_size_perf = ((ub_size / UB_TENSOR_NUM_FP32_PAD) / BYTE_BLOCK) * BYTE_BLOCK;
     ids_ub_size = ids_ub_size_perf;
     x_ub_size = ids_ub_size_perf;
     x_ub_size_pad = ids_ub_size_perf;
   }
   if (isHighPerf && (not commParas.is_pad)) {
     ids_ub_size_perf = ((ub_size / UB_TENSOR_NUM_FP32_INPUT_LAST_AXIS_NOT_ALIGN_HP) / BYTE_BLOCK) * BYTE_BLOCK;
     ids_ub_size = ids_ub_size_perf;
     x_ub_size = ids_ub_size_perf;
     x_ub_size_pad = 0;
   }
   return;
 }

 /******************MODE_FP32_INPUT_LAST_AXIS_ALIGN******************/
 static void ComputeEleNumOneCore(const int32_t& ids_num, const int32_t& e_size,
                                  const int32_t& num_segments, UnsortedSegmentSumTilingParamsFp32* tiling_params) {
   int32_t core_need = tiling_params->need_core_num_input_scalar;
   if (num_segments > 1) {
     tiling_params->ids_ele_num_front_core_input_scalar = (ids_num + core_need - 1) / core_need;

     tiling_params->ids_ele_num_last_core_input_scalar = ComputeDivRemainders(
         ids_num, tiling_params->ids_ele_num_front_core_input_scalar, core_need - 1);
     if (tiling_params->ids_ele_num_last_core_input_scalar <= 0) {
       tiling_params->ids_ele_num_front_core_input_scalar = ids_num / core_need;
       tiling_params->ids_ele_num_last_core_input_scalar = ComputeDivRemainders(
           ids_num, tiling_params->ids_ele_num_front_core_input_scalar, core_need - 1);
     }

     tiling_params->input_ele_num_front_core_input_scalar = tiling_params->ids_ele_num_front_core_input_scalar * e_size;
     tiling_params->input_ele_num_last_core_input_scalar = tiling_params->ids_ele_num_last_core_input_scalar * e_size;
   } else {
     tiling_params->ids_ele_num_front_core_input_scalar = (ids_num + core_need - 1) / core_need;
     tiling_params->ids_ele_num_last_core_input_scalar = ComputeDivRemainders(
         ids_num, tiling_params->ids_ele_num_front_core_input_scalar, core_need - 1);
     if (tiling_params->ids_ele_num_last_core_input_scalar <= 0) {
       tiling_params->ids_ele_num_front_core_input_scalar = ids_num / core_need;
       tiling_params->ids_ele_num_last_core_input_scalar = ComputeDivRemainders(
           ids_num, tiling_params->ids_ele_num_front_core_input_scalar, core_need - 1);
     }
   }
 }

 static void SetInputParamsMovGm2ub(InputParamsGm2UbOut& gm_ub_out_params,
                                    UnsortedSegmentSumTilingParamsFp32* tiling_params) {
   if (gm_ub_out_params.mode == FRONT_PART_FRONT_CORE) {
     tiling_params->input_mov_times_gm2ub_front_part_front_core_input_scalar = gm_ub_out_params.input_mov_times_gm2ub;
     tiling_params->input_front_burst_len_front_part_front_core_input_scalar = gm_ub_out_params.input_front_burst_len;
     tiling_params->input_last_burst_len_front_part_front_core_input_scalar = gm_ub_out_params.input_last_burst_len;
     tiling_params->input_front_ele_num_ub_front_part_front_core_input_scalar =
         gm_ub_out_params.input_ele_num_ub_front_part;
     tiling_params->input_last_ele_num_ub_front_part_front_core_input_scalar =
         gm_ub_out_params.input_ele_num_ub_last_part;
     tiling_params->input_front_rows_front_part_front_core_input_scalar = gm_ub_out_params.input_rows_front_part;
     tiling_params->input_last_rows_front_part_front_core_input_scalar = gm_ub_out_params.input_rows_last_part;
   } else if (gm_ub_out_params.mode == LAST_PART_FRONT_CORE) {
     tiling_params->input_mov_times_gm2ub_last_part_front_core_input_scalar = gm_ub_out_params.input_mov_times_gm2ub;
     tiling_params->input_front_burst_len_last_part_front_core_input_scalar = gm_ub_out_params.input_front_burst_len;
     tiling_params->input_last_burst_len_last_part_front_core_input_scalar = gm_ub_out_params.input_last_burst_len;
     tiling_params->input_front_ele_num_ub_last_part_front_core_input_scalar =
         gm_ub_out_params.input_ele_num_ub_front_part;
     tiling_params->input_last_ele_num_ub_last_part_front_core_input_scalar =
         gm_ub_out_params.input_ele_num_ub_last_part;
     tiling_params->input_front_rows_last_part_front_core_input_scalar = gm_ub_out_params.input_rows_front_part;
     tiling_params->input_last_rows_last_part_front_core_input_scalar = gm_ub_out_params.input_rows_last_part;
   } else if (gm_ub_out_params.mode == FRONT_PART_LAST_CORE) {
     tiling_params->input_mov_times_gm2ub_front_part_last_core_input_scalar = gm_ub_out_params.input_mov_times_gm2ub;
     tiling_params->input_front_burst_len_front_part_last_core_input_scalar = gm_ub_out_params.input_front_burst_len;
     tiling_params->input_last_burst_len_front_part_last_core_input_scalar = gm_ub_out_params.input_last_burst_len;
     tiling_params->input_front_ele_num_ub_front_part_last_core_input_scalar =
         gm_ub_out_params.input_ele_num_ub_front_part;
     tiling_params->input_last_ele_num_ub_front_part_last_core_input_scalar =
         gm_ub_out_params.input_ele_num_ub_last_part;
     tiling_params->input_front_rows_front_part_last_core_input_scalar = gm_ub_out_params.input_rows_front_part;
     tiling_params->input_last_rows_front_part_last_core_input_scalar = gm_ub_out_params.input_rows_last_part;
   } else if (gm_ub_out_params.mode == LAST_PART_LAST_CORE) {
     tiling_params->input_mov_times_gm2ub_last_part_last_core_input_scalar = gm_ub_out_params.input_mov_times_gm2ub;
     tiling_params->input_front_burst_len_last_part_last_core_input_scalar = gm_ub_out_params.input_front_burst_len;
     tiling_params->input_last_burst_len_last_part_last_core_input_scalar = gm_ub_out_params.input_last_burst_len;
     tiling_params->input_front_ele_num_ub_last_part_last_core_input_scalar =
         gm_ub_out_params.input_ele_num_ub_front_part;
     tiling_params->input_last_ele_num_ub_last_part_last_core_input_scalar = gm_ub_out_params.input_ele_num_ub_last_part;
     tiling_params->input_front_rows_last_part_last_core_input_scalar = gm_ub_out_params.input_rows_front_part;
     tiling_params->input_last_rows_last_part_last_core_input_scalar = gm_ub_out_params.input_rows_last_part;
   }

   return;
 }

 static void ComputeInputParamsMovGm2ub(const int32_t& ub_tensor_size, const EleByte& input_ele_byte,
                                        const int32_t& e_size, UnsortedSegmentSumTilingParamsFp32* tiling_params,
                                        InputParamsGm2UbOut& gm_ub_out_params) {
   int32_t e_mov_times_gm2ub = 0;
   int32_t ids_ele_num_ub = 0;
   int32_t input_ele_byte_int32 = static_cast<int32_t>(input_ele_byte);

   if (gm_ub_out_params.mode == FRONT_PART_FRONT_CORE) {
     e_mov_times_gm2ub = tiling_params->e_mov_times_gm2ub_input_scalar;
     ids_ele_num_ub = tiling_params->ids_ele_num_ub_front_part_front_core_input_scalar;
   } else if (gm_ub_out_params.mode == LAST_PART_FRONT_CORE) {
     e_mov_times_gm2ub = tiling_params->e_mov_times_gm2ub_input_scalar;
     ids_ele_num_ub = tiling_params->ids_ele_num_ub_last_part_front_core_input_scalar;
   } else if (gm_ub_out_params.mode == FRONT_PART_LAST_CORE) {
     e_mov_times_gm2ub = tiling_params->e_mov_times_gm2ub_input_scalar;
     ids_ele_num_ub = tiling_params->ids_ele_num_ub_front_part_last_core_input_scalar;
   } else if (gm_ub_out_params.mode == LAST_PART_LAST_CORE) {
     e_mov_times_gm2ub = tiling_params->e_mov_times_gm2ub_input_scalar;
     ids_ele_num_ub = tiling_params->ids_ele_num_ub_last_part_last_core_input_scalar;
   }

   if (e_mov_times_gm2ub == 1) {
     // e_size is small, ub_tensor_size is enough for e_num
     int32_t e_byte_align_32B = UssCeil(e_size, FP32_ELE_NUM_ALIGN_32B) * input_ele_byte_int32;
     gm_ub_out_params.input_rows_front_part = ub_tensor_size / e_byte_align_32B;
     gm_ub_out_params.input_mov_times_gm2ub = UssCeilDiv(ids_ele_num_ub, gm_ub_out_params.input_rows_front_part);
     if (gm_ub_out_params.input_mov_times_gm2ub > 1) {
       gm_ub_out_params.input_front_burst_len = e_byte_align_32B * gm_ub_out_params.input_rows_front_part / BYTE_BLOCK;
       gm_ub_out_params.input_ele_num_ub_front_part = e_size * gm_ub_out_params.input_rows_front_part;
       gm_ub_out_params.input_rows_last_part = ComputeDivRemainders(
           ids_ele_num_ub, gm_ub_out_params.input_rows_front_part, gm_ub_out_params.input_mov_times_gm2ub - 1);
       gm_ub_out_params.input_last_burst_len = e_byte_align_32B * gm_ub_out_params.input_rows_last_part / BYTE_BLOCK;
       gm_ub_out_params.input_ele_num_ub_last_part = gm_ub_out_params.input_rows_last_part * e_size;
     } else if (gm_ub_out_params.input_mov_times_gm2ub == 1) {
       gm_ub_out_params.input_rows_front_part = ids_ele_num_ub;
       gm_ub_out_params.input_rows_last_part = gm_ub_out_params.input_rows_front_part;
       gm_ub_out_params.input_ele_num_ub_front_part = e_size * gm_ub_out_params.input_rows_front_part;
       gm_ub_out_params.input_ele_num_ub_last_part = gm_ub_out_params.input_ele_num_ub_front_part;
       gm_ub_out_params.input_front_burst_len =
           UssCeil(gm_ub_out_params.input_ele_num_ub_front_part * input_ele_byte_int32, BYTE_BLOCK) / BYTE_BLOCK;
       gm_ub_out_params.input_last_burst_len = gm_ub_out_params.input_front_burst_len;
     }
   } else if (e_mov_times_gm2ub > 1) {
     // e_size is big, ub_tensor_size is not enough for e_num, use e params to move data
     gm_ub_out_params.input_mov_times_gm2ub = ids_ele_num_ub;
   }
   return;
 }

 static void ComputeIdsParamsMovGm2ub(const int32_t& ub_tensor_size, const EleByte& ids_ele_byte, const int32_t& mode,
                                      UnsortedSegmentSumTilingParamsFp32* tiling_params) {
   int32_t ids_mov_times_gm2ub = 0;
   int32_t ids_front_burst_len = 0;
   int32_t ids_last_burst_len = 0;
   int32_t ids_ele_num_ub_front_part = 0;
   int32_t ids_ele_num_ub_last_part = 0;
   int32_t ids_ele_byte_int32 = static_cast<int32_t>(ids_ele_byte);
   int32_t max_ids_ele_num_one_ub_tensor = ub_tensor_size / ids_ele_byte_int32;

   int32_t ids_ele_num_one_core = (mode == FRONT_CORE) ? tiling_params->ids_ele_num_front_core_input_scalar
                                                       : tiling_params->ids_ele_num_last_core_input_scalar;
   if (ids_ele_num_one_core <= max_ids_ele_num_one_ub_tensor) {
     // mov_times = 1, ub tensor is enough for ele one core
     ids_mov_times_gm2ub = 1;
     ids_ele_num_ub_front_part = ids_ele_num_one_core;
     ids_ele_num_ub_last_part = ids_ele_num_ub_front_part;
   } else {
     // mov_times > 1
     if (ids_ele_num_one_core % max_ids_ele_num_one_ub_tensor == 0) {
       // no last part
       ids_mov_times_gm2ub = ids_ele_num_one_core / max_ids_ele_num_one_ub_tensor;
       ids_ele_num_ub_front_part = max_ids_ele_num_one_ub_tensor;
       ids_ele_num_ub_last_part = ids_ele_num_ub_front_part;
     } else {
       // exist last part
       ids_mov_times_gm2ub = ids_ele_num_one_core / max_ids_ele_num_one_ub_tensor + 1;
       ids_ele_num_ub_front_part = max_ids_ele_num_one_ub_tensor;
       ids_ele_num_ub_last_part =
           ComputeDivRemainders(ids_ele_num_one_core, max_ids_ele_num_one_ub_tensor, ids_mov_times_gm2ub - 1);
     }
   }

   ids_front_burst_len = UssCeilDiv(ids_ele_num_ub_front_part * ids_ele_byte_int32, BYTE_BLOCK);
   ids_last_burst_len = UssCeilDiv(ids_ele_num_ub_last_part * ids_ele_byte_int32, BYTE_BLOCK);

   if (mode == FRONT_CORE) {
     tiling_params->ids_mov_times_gm2ub_front_core_input_scalar = ids_mov_times_gm2ub;
     tiling_params->ids_front_burst_len_front_core_input_scalar = ids_front_burst_len;
     tiling_params->ids_last_burst_len_front_core_input_scalar = ids_last_burst_len;
     tiling_params->ids_ele_num_ub_front_part_front_core_input_scalar = ids_ele_num_ub_front_part;
     tiling_params->ids_ele_num_ub_last_part_front_core_input_scalar = ids_ele_num_ub_last_part;
   } else {
     tiling_params->ids_mov_times_gm2ub_last_core_input_scalar = ids_mov_times_gm2ub;
     tiling_params->ids_front_burst_len_last_core_input_scalar = ids_front_burst_len;
     tiling_params->ids_last_burst_len_last_core_input_scalar = ids_last_burst_len;
     tiling_params->ids_ele_num_ub_front_part_last_core_input_scalar = ids_ele_num_ub_front_part;
     tiling_params->ids_ele_num_ub_last_part_last_core_input_scalar = ids_ele_num_ub_last_part;
   }

   return;
 }

 static void ComputeIdsParamsMovGm2ubNoAtomic(const int32_t& ids_ele_num_one_core, const int32_t& id_once_num,
                                              const EleByte& ids_ele_byte,
                                              UnsortedSegmentSumTilingParamsInt32* tiling_params) {
   int32_t max_ids_ele_num_one_ub_tensor = id_once_num;
   if (ids_ele_num_one_core <= max_ids_ele_num_one_ub_tensor) {
     // mov_times = 1, ub tensor is enough for ele one core
     tiling_params->ids_mov_times_gm2ub_input_scalar = 1;
     tiling_params->ids_ele_num_ub_front_part_input_scalar = ids_ele_num_one_core;
     tiling_params->ids_ele_num_ub_last_part_input_scalar = tiling_params->ids_ele_num_ub_front_part_input_scalar;
   } else {
     // mov_times > 1
     if (ids_ele_num_one_core % max_ids_ele_num_one_ub_tensor == 0) {
       // no last part
       tiling_params->ids_mov_times_gm2ub_input_scalar = ids_ele_num_one_core / max_ids_ele_num_one_ub_tensor;
       tiling_params->ids_ele_num_ub_front_part_input_scalar = max_ids_ele_num_one_ub_tensor;
       tiling_params->ids_ele_num_ub_last_part_input_scalar = tiling_params->ids_ele_num_ub_front_part_input_scalar;
     } else {
       // exist last part
       tiling_params->ids_mov_times_gm2ub_input_scalar = ids_ele_num_one_core / max_ids_ele_num_one_ub_tensor + 1;
       tiling_params->ids_ele_num_ub_front_part_input_scalar = max_ids_ele_num_one_ub_tensor;
       tiling_params->ids_ele_num_ub_last_part_input_scalar = ComputeDivRemainders(
           ids_ele_num_one_core, max_ids_ele_num_one_ub_tensor, tiling_params->ids_mov_times_gm2ub_input_scalar - 1);
     }
   }
   tiling_params->ids_front_burst_len_input_scalar =
       UssCeilDiv(tiling_params->ids_ele_num_ub_front_part_input_scalar * ids_ele_byte, BYTE_BLOCK);
   tiling_params->ids_last_burst_len_input_scalar =
       UssCeilDiv(tiling_params->ids_ele_num_ub_last_part_input_scalar * ids_ele_byte, BYTE_BLOCK);
 }
 static void ComputeInitOutputUbParams(const int32_t& ids_ele_num, const int32_t& output_ub_ele_num_one_row,
                                       UnsortedSegmentSumTilingParamsFp32* tiling_params, int32_t mode) {
   int32_t e_size_align = tiling_params->input_last_axis_align_floor_ele_num_input_scalar;
   int32_t repeat_times = UssCeilDiv(ids_ele_num * output_ub_ele_num_one_row, MASK_FP32);
   int32_t repeat_pad = ids_ele_num / 128;
   int32_t sheng = ComputeDivRemainders(ids_ele_num, 128, repeat_pad);
   int32_t col_sub_block = UssCeilDiv(sheng * e_size_align, 8);

   int32_t output_ub_init_times = 0;
   int32_t output_ub_init_last_repeat_time = 0;
   OP_LOGD("UnsorteSegmentSum",
           "#e_size_align is %d, ids_ele_num is %d, repeat_pad is %d, sheng is %d, col_sub_block is %d, mode is %d",
           e_size_align, ids_ele_num, repeat_pad, sheng, col_sub_block, mode);
   if (repeat_times % MAX_REPEAT_TIME == 0) {
     output_ub_init_times = repeat_times / MAX_REPEAT_TIME;
     output_ub_init_last_repeat_time = MAX_REPEAT_TIME;
   } else {
     output_ub_init_times = repeat_times / MAX_REPEAT_TIME + 1;
     output_ub_init_last_repeat_time =
         ComputeDivRemainders(repeat_times, repeat_times / MAX_REPEAT_TIME, MAX_REPEAT_TIME);
   }
   if (mode == 0) {
     tiling_params->output_ub_init_last_repeat_time_front_part_front_core_input_scalar = output_ub_init_last_repeat_time;
     tiling_params->output_ub_init_times_front_part_front_core_input_scalar = output_ub_init_times;
     tiling_params->repeat_front_front_part_front_core = repeat_pad;
     tiling_params->col_sub_block_front_front_part_front_core = col_sub_block;
   } else if (mode == 1) {
     tiling_params->output_ub_init_last_row_last_repeat_time_front_part_front_core_input_scalar =
         output_ub_init_last_repeat_time;
     tiling_params->output_ub_init_last_row_times_front_part_front_core_input_scalar = output_ub_init_times;
     tiling_params->repeat_last_front_part_front_core = repeat_pad;
     tiling_params->col_sub_block_last_front_part_front_core = col_sub_block;
   } else if (mode == PART_TWO) {
     tiling_params->output_ub_init_last_repeat_time_last_part_front_core_input_scalar = output_ub_init_last_repeat_time;
     tiling_params->output_ub_init_times_last_part_front_core_input_scalar = output_ub_init_times;
     tiling_params->repeat_front_last_part_front_core = repeat_pad;
     tiling_params->col_sub_block_front_last_part_front_core = col_sub_block;
   } else if (mode == PART_THREE) {
     tiling_params->output_ub_init_last_row_last_repeat_time_last_part_front_core_input_scalar =
         output_ub_init_last_repeat_time;
     tiling_params->output_ub_init_last_row_times_last_part_front_core_input_scalar = output_ub_init_times;
     tiling_params->repeat_last_last_part_front_core = repeat_pad;
     tiling_params->col_sub_block_last_last_part_front_core = col_sub_block;
   } else if (mode == PART_FOUR) {
     tiling_params->output_ub_init_last_repeat_time_front_part_last_core_input_scalar = output_ub_init_last_repeat_time;
     tiling_params->output_ub_init_times_front_part_last_core_input_scalar = output_ub_init_times;
     tiling_params->repeat_front_front_part_last_core = repeat_pad;
     tiling_params->col_sub_block_front_front_part_last_core = col_sub_block;
   } else if (mode == PART_FIVE) {
     tiling_params->output_ub_init_last_row_last_repeat_time_front_part_last_core_input_scalar =
         output_ub_init_last_repeat_time;
     tiling_params->output_ub_init_last_row_times_front_part_last_core_input_scalar = output_ub_init_times;
     tiling_params->repeat_last_front_part_last_core = repeat_pad;
     tiling_params->col_sub_block_last_front_part_last_core = col_sub_block;
   } else if (mode == PART_SIX) {
     tiling_params->output_ub_init_last_repeat_time_last_part_last_core_input_scalar = output_ub_init_last_repeat_time;
     tiling_params->output_ub_init_times_last_part_last_core_input_scalar = output_ub_init_times;
     tiling_params->repeat_front_last_part_last_core = repeat_pad;
     tiling_params->col_sub_block_front_last_part_last_core = col_sub_block;
   } else if (mode == PART_SERVERN) {
     tiling_params->output_ub_init_last_row_last_repeat_time_last_part_last_core_input_scalar =
         output_ub_init_last_repeat_time;
     tiling_params->output_ub_init_last_row_times_last_part_last_core_input_scalar = output_ub_init_times;
     tiling_params->repeat_last_last_part_last_core = repeat_pad;
     tiling_params->col_sub_block_last_last_part_last_core = col_sub_block;
   }
 }

 static void ComputeNumSegmentsParams(const int32_t& num_segmens, const int32_t& e_size,
                                      const int32_t& output_ub_ele_num_one_row,
                                      UnsortedSegmentSumTilingParamsInt32* tiling_params) {
   if (tiling_params->need_core_num_input_scalar == 1 && num_segmens > 1) {
     tiling_params->num_segments_front_core_input_scalar = num_segmens;
     tiling_params->num_segments_last_core_input_scalar = tiling_params->num_segments_front_core_input_scalar;
   } else if (tiling_params->need_core_num_input_scalar > 1 && num_segmens > 1) {
     tiling_params->num_segments_front_core_input_scalar =
         UssCeilDivNoAtomic(num_segmens, tiling_params->need_core_num_input_scalar, e_size, output_ub_ele_num_one_row);
     tiling_params->num_segments_last_core_input_scalar =
         ComputeDivRemainders(num_segmens, tiling_params->num_segments_front_core_input_scalar,
                              tiling_params->need_core_num_input_scalar - 1);
   } else if (num_segmens == 1) {
     if (e_size < output_ub_ele_num_one_row) {
       tiling_params->num_segments_front_core_input_scalar = e_size;
       tiling_params->num_segments_last_core_input_scalar = e_size;
     } else {
       if (e_size / tiling_params->need_core_num_input_scalar > output_ub_ele_num_one_row) {
         tiling_params->num_segments_front_core_input_scalar =
             e_size / tiling_params->need_core_num_input_scalar / output_ub_ele_num_one_row * output_ub_ele_num_one_row;
       } else {
         tiling_params->num_segments_front_core_input_scalar = output_ub_ele_num_one_row;
       }
       tiling_params->num_segments_last_core_input_scalar =
           e_size -
           (tiling_params->num_segments_front_core_input_scalar * (tiling_params->need_core_num_input_scalar - 1));
     }
   }

   return;
 }

 static void ComputeENumParamsGEMaxEleNum(const CommParas4Int32& comm_paras,
                                          UnsortedSegmentSumTilingParamsInt32* tiling_params) {
   int32_t mask = (comm_paras.input_dytpe == ge::DT_INT32) ? MASK_INT32 : MASK_FP16;
   int32_t byte = (comm_paras.input_dytpe == ge::DT_INT32) ? INT32_BLOCK_NUM : FP16_BLOCK_NUM;
   int32_t comm_paras_ele_byte = static_cast<int32_t>(comm_paras.ele_byte);

   tiling_params->e_mov_times_gm2ub_input_scalar = UssCeilDiv(tiling_params->e_num_input_scalar, comm_paras.e_once_num);
   int32_t last = ComputeDivRemainders(tiling_params->e_num_input_scalar, comm_paras.e_once_num,
                                       tiling_params->e_mov_times_gm2ub_input_scalar - 1);
   if (last < byte) {
     tiling_params->e_mov_times_gm2ub_input_scalar =
         UssCeilDiv(tiling_params->e_num_input_scalar, comm_paras.e_once_num - mask);
   }
   // front part
   tiling_params->e_gm2ub_front_burst_len_input_scalar =
       UssCeilDiv(comm_paras.e_once_num * comm_paras_ele_byte, BYTE_BLOCK);
   tiling_params->e_ub2gm_front_burst_len_input_scalar = comm_paras.e_once_num * comm_paras_ele_byte / BYTE_BLOCK;
   tiling_params->e_num_front_part_input_scalar = comm_paras.e_once_num;
   tiling_params->repeat_time_front_part_input_scalar = UssCeilDiv(tiling_params->e_num_front_part_input_scalar, mask);
   // last part
   tiling_params->e_num_last_part_input_scalar =
       ComputeDivRemainders(tiling_params->e_num_input_scalar, tiling_params->e_num_front_part_input_scalar,
                            tiling_params->e_mov_times_gm2ub_input_scalar - 1);
   tiling_params->e_gm2ub_last_burst_len_input_scalar =
       UssCeilDiv(tiling_params->e_num_last_part_input_scalar * comm_paras_ele_byte, BYTE_BLOCK);
   tiling_params->repeat_time_last_part_input_scalar = UssCeilDiv(tiling_params->e_num_last_part_input_scalar, mask);
   tiling_params->e_ub2gm_last_burst_len_input_scalar =
       tiling_params->e_num_last_part_input_scalar * comm_paras_ele_byte / BYTE_BLOCK;

   return;
 }

 static void ComputeENumParamsInputScalarGEOneBlockNum(const CommParas4Int32& comm_paras,
                                                       UnsortedSegmentSumTilingParamsInt32* tiling_params) {
   int32_t mask = (comm_paras.input_dytpe == ge::DT_INT32) ? MASK_INT32 : MASK_FP16;
   int32_t byte = (comm_paras.input_dytpe == ge::DT_INT32) ? INT32_BLOCK_NUM : FP16_BLOCK_NUM;
   int32_t comm_paras_ele_byte = static_cast<int32_t>(comm_paras.ele_byte);

   tiling_params->e_ub2gm_front_burst_len_input_scalar =
       tiling_params->e_num_input_scalar * comm_paras_ele_byte / BYTE_BLOCK;
   if (tiling_params->need_core_num_input_scalar == 1 &&
       tiling_params->e_num_input_scalar * comm_paras_ele_byte % BYTE_BLOCK != 0) {
     tiling_params->e_ub2gm_front_burst_len_input_scalar = tiling_params->e_ub2gm_front_burst_len_input_scalar + 1;
   }
   tiling_params->e_gm2ub_front_burst_len_input_scalar =
       UssCeilDiv(tiling_params->e_num_input_scalar * comm_paras_ele_byte, BYTE_BLOCK);
   if (tiling_params->e_ub2gm_front_burst_len_input_scalar < 1) {
     tiling_params->e_ub2gm_front_burst_len_input_scalar = 1;
   }
   tiling_params->e_num_front_part_input_scalar = tiling_params->e_num_input_scalar;
   tiling_params->repeat_time_front_part_input_scalar = UssCeilDiv(tiling_params->e_num_front_part_input_scalar, mask);
   // last part
   tiling_params->e_num_last_part_input_scalar = tiling_params->e_num_input_scalar;
   tiling_params->e_ub2gm_last_burst_len_input_scalar = tiling_params->e_ub2gm_front_burst_len_input_scalar;
   tiling_params->e_gm2ub_last_burst_len_input_scalar = tiling_params->e_gm2ub_front_burst_len_input_scalar;
   if (tiling_params->e_num_input_scalar % byte == 0 && comm_paras.all_size <= comm_paras.e_once_num) {
     tiling_params->e_gm2ub_front_burst_len_input_scalar =
         UssCeilDiv(comm_paras.all_size * comm_paras_ele_byte, BYTE_BLOCK);
     tiling_params->e_gm2ub_last_burst_len_input_scalar = tiling_params->e_gm2ub_front_burst_len_input_scalar;
   }
   tiling_params->repeat_time_last_part_input_scalar = tiling_params->repeat_time_front_part_input_scalar;

   return;
 }

 static void ComputeENumParamsFrontInputScalarLESegmentMax(const CommParas4Int32& comm_paras,
                                                           UnsortedSegmentSumTilingParamsInt32* tiling_params) {
   int32_t byte = (comm_paras.input_dytpe == ge::DT_INT32) ? INT32_BLOCK_NUM : FP16_BLOCK_NUM;
   int32_t count = tiling_params->e_num_input_scalar * tiling_params->num_segments_front_core_input_scalar;
   int32_t comm_paras_ele_byte = static_cast<int32_t>(comm_paras.ele_byte);

   tiling_params->num_segment_max_time = 1;
   tiling_params->e_ub2gm_front_burst_len_input_scalar = count * comm_paras_ele_byte / BYTE_BLOCK;
   if (tiling_params->e_ub2gm_front_burst_len_input_scalar < 1) {
     tiling_params->e_ub2gm_front_burst_len_input_scalar = 1;
   }
   tiling_params->front_num_segment = tiling_params->num_segments_front_core_input_scalar;
   tiling_params->front_num_segment_last = tiling_params->num_segments_front_core_input_scalar;
   tiling_params->e_gm2ub_front_burst_len_input_scalar = 1;
   tiling_params->repeat_time_front_part_input_scalar = 1;
   tiling_params->align_scalar = (count % byte == 0) ? 0 : byte - (count - (count / byte) * byte);
   tiling_params->e_num_front_part_input_scalar = tiling_params->e_num_input_scalar;
   tiling_params->e_num_last_part_input_scalar = tiling_params->e_num_input_scalar;
   tiling_params->e_ub2gm_last_burst_len_input_scalar = tiling_params->e_ub2gm_front_burst_len_input_scalar;
   tiling_params->e_gm2ub_last_burst_len_input_scalar = tiling_params->e_gm2ub_front_burst_len_input_scalar;
   tiling_params->repeat_time_last_part_input_scalar = tiling_params->repeat_time_front_part_input_scalar;

   return;
 }

 static void ComputeENumParamsFrontInputScalarGTSegmentMax(const CommParas4Int32& comm_paras,
                                                           UnsortedSegmentSumTilingParamsInt32* tiling_params) {
   int32_t byte = (comm_paras.input_dytpe == ge::DT_INT32) ? INT32_BLOCK_NUM : FP16_BLOCK_NUM;

   tiling_params->num_segment_max_time =
       tiling_params->num_segments_front_core_input_scalar / tiling_params->num_segment_max;
   tiling_params->num_segment_max_time =
       (tiling_params->num_segments_front_core_input_scalar % tiling_params->num_segment_max == 0)
           ? tiling_params->num_segment_max_time
           : tiling_params->num_segment_max_time + 1;
   tiling_params->front_num_segment = tiling_params->num_segment_max;
   tiling_params->front_num_segment_last = tiling_params->num_segments_front_core_input_scalar -
                                           tiling_params->num_segment_max * (tiling_params->num_segment_max_time - 1);
   if (tiling_params->front_num_segment_last * tiling_params->e_num_input_scalar < byte) {
     tiling_params->front_num_segment_last = byte;
     int32_t front = tiling_params->num_segments_front_core_input_scalar - tiling_params->front_num_segment_last;
     tiling_params->num_segment_max = (front / (tiling_params->num_segment_max_time - 1) / byte) * byte;
     tiling_params->front_num_segment = tiling_params->num_segment_max;
     tiling_params->front_num_segment_last = tiling_params->num_segments_front_core_input_scalar -
                                             tiling_params->num_segment_max * (tiling_params->num_segment_max_time - 1);
   }
   tiling_params->align_scalar =
       (tiling_params->front_num_segment_last * tiling_params->e_num_input_scalar % byte == 0)
           ? 0
           : byte - (tiling_params->front_num_segment_last * tiling_params->e_num_input_scalar -
                     (tiling_params->front_num_segment_last * tiling_params->e_num_input_scalar / byte) * byte);
   tiling_params->e_ub2gm_front_burst_len_input_scalar =
       tiling_params->num_segment_max * tiling_params->e_num_input_scalar * comm_paras.ele_byte / BYTE_BLOCK;
   tiling_params->e_gm2ub_front_burst_len_input_scalar = 1;
   tiling_params->repeat_time_front_part_input_scalar = 1;

   tiling_params->e_num_front_part_input_scalar = tiling_params->e_num_input_scalar;
   tiling_params->e_num_last_part_input_scalar = tiling_params->e_num_input_scalar;
   tiling_params->e_ub2gm_last_burst_len_input_scalar =
       tiling_params->front_num_segment_last * tiling_params->e_num_input_scalar * comm_paras.ele_byte / BYTE_BLOCK;
   tiling_params->e_gm2ub_last_burst_len_input_scalar = tiling_params->e_gm2ub_front_burst_len_input_scalar;
   tiling_params->repeat_time_last_part_input_scalar = tiling_params->repeat_time_front_part_input_scalar;

   return;
 }

 static void ComputeENumParamsLastInputScalarLTSegmentMax(const CommParas4Int32& comm_paras,
                                                          UnsortedSegmentSumTilingParamsInt32* tiling_params) {
   int32_t byte = (comm_paras.input_dytpe == ge::DT_INT32) ? INT32_BLOCK_NUM : FP16_BLOCK_NUM;
   int32_t lastcore_count = tiling_params->e_num_input_scalar * tiling_params->num_segments_last_core_input_scalar;
   int32_t comm_paras_ele_byte = static_cast<int32_t>(comm_paras.ele_byte);

   tiling_params->num_segment_max_time_lastcore = 1;
   tiling_params->align_scalar_lastcore =
       (lastcore_count % byte == 0) ? 0 : byte - (lastcore_count - (lastcore_count / byte) * byte);
   tiling_params->e_num_front_part_input_scalar = tiling_params->e_num_input_scalar;
   tiling_params->e_num_last_part_input_scalar = tiling_params->e_num_input_scalar;
   tiling_params->e_ub2gm_front_burst_len_input_scalar_lastcore = lastcore_count * comm_paras_ele_byte / BYTE_BLOCK;
   tiling_params->e_ub2gm_last_burst_len_input_scalar_lastcore =
       tiling_params->e_ub2gm_front_burst_len_input_scalar_lastcore;
   if (tiling_params->e_ub2gm_front_burst_len_input_scalar_lastcore < 1) {
     tiling_params->e_ub2gm_front_burst_len_input_scalar_lastcore = 1;
   }
   tiling_params->front_num_segment_lastcore = tiling_params->num_segments_last_core_input_scalar;
   tiling_params->front_num_segment_last_lastcore = tiling_params->num_segments_last_core_input_scalar;
   tiling_params->e_gm2ub_last_burst_len_input_scalar = 1;
   tiling_params->repeat_time_last_part_input_scalar = 1;
   tiling_params->e_gm2ub_front_burst_len_input_scalar = 1;
   tiling_params->repeat_time_front_part_input_scalar = 1;

   return;
 }

 static void ComputeENumParamsLastInputScalarGTSegmentMax(const CommParas4Int32& comm_paras,
                                                          UnsortedSegmentSumTilingParamsInt32* tiling_params) {
   int32_t byte = (comm_paras.input_dytpe == ge::DT_INT32) ? INT32_BLOCK_NUM : FP16_BLOCK_NUM;

   tiling_params->num_segment_max_time_lastcore =
       tiling_params->num_segments_last_core_input_scalar / tiling_params->num_segment_max;
   tiling_params->num_segment_max_time_lastcore =
       (tiling_params->num_segments_last_core_input_scalar % tiling_params->num_segment_max == 0)
           ? tiling_params->num_segment_max_time_lastcore
           : tiling_params->num_segment_max_time_lastcore + 1;
   tiling_params->front_num_segment_lastcore = tiling_params->num_segment_max;
   tiling_params->front_num_segment_last_lastcore =
       tiling_params->num_segments_last_core_input_scalar -
       tiling_params->num_segment_max * (tiling_params->num_segment_max_time_lastcore - 1);
   if (tiling_params->front_num_segment_last_lastcore * tiling_params->e_num_input_scalar < byte) {
     tiling_params->front_num_segment_last_lastcore = byte;
     int32_t front = tiling_params->num_segments_last_core_input_scalar - tiling_params->front_num_segment_last_lastcore;
     int32_t num_segment_max_lastcore_front = (front / (tiling_params->num_segment_max_time_lastcore - 1) / byte) * byte;
     tiling_params->front_num_segment_lastcore = num_segment_max_lastcore_front;
     tiling_params->front_num_segment_last_lastcore =
         tiling_params->num_segments_last_core_input_scalar -
         tiling_params->front_num_segment_lastcore * (tiling_params->num_segment_max_time_lastcore - 1);
   }

   tiling_params->align_scalar_lastcore =
       (tiling_params->front_num_segment_last_lastcore * tiling_params->e_num_input_scalar % byte == 0)
           ? 0
           : byte - (tiling_params->front_num_segment_last_lastcore * tiling_params->e_num_input_scalar -
                     (tiling_params->front_num_segment_last_lastcore * tiling_params->e_num_input_scalar / byte) * byte);
   tiling_params->e_ub2gm_front_burst_len_input_scalar_lastcore =
       tiling_params->front_num_segment_lastcore * tiling_params->e_num_input_scalar * comm_paras.ele_byte / BYTE_BLOCK;
   tiling_params->e_gm2ub_front_burst_len_input_scalar = 1;
   tiling_params->repeat_time_front_part_input_scalar = 1;

   tiling_params->e_num_front_part_input_scalar = tiling_params->e_num_input_scalar;
   tiling_params->e_num_last_part_input_scalar = tiling_params->e_num_input_scalar;
   tiling_params->e_ub2gm_last_burst_len_input_scalar_lastcore = tiling_params->front_num_segment_last_lastcore *
                                                                 tiling_params->e_num_input_scalar *
                                                                 comm_paras.ele_byte / BYTE_BLOCK;
   tiling_params->e_gm2ub_last_burst_len_input_scalar = 1;
   tiling_params->repeat_time_last_part_input_scalar = 1;

   return;
 }

 static void ComputeENumParamsInputScalarLTOneBlockNum(const CommParas4Int32& comm_paras,
                                                       UnsortedSegmentSumTilingParamsInt32* tiling_params) {
   if (tiling_params->num_segments_front_core_input_scalar <= tiling_params->num_segment_max) {
     ComputeENumParamsFrontInputScalarLESegmentMax(comm_paras, tiling_params);
   } else {
     ComputeENumParamsFrontInputScalarGTSegmentMax(comm_paras, tiling_params);
   }

   if (tiling_params->num_segments_last_core_input_scalar < tiling_params->num_segment_max) {
     ComputeENumParamsLastInputScalarLTSegmentMax(comm_paras, tiling_params);
   } else if (tiling_params->num_segments_last_core_input_scalar > tiling_params->num_segment_max) {
     ComputeENumParamsLastInputScalarGTSegmentMax(comm_paras, tiling_params);
   }
 }
 static void ComputeENumParamsLTMaxEleNum(const CommParas4Int32& comm_paras,
                                          UnsortedSegmentSumTilingParamsInt32* tiling_params) {
   int32_t byte = (comm_paras.input_dytpe == ge::DT_INT32) ? INT32_BLOCK_NUM : FP16_BLOCK_NUM;

   tiling_params->e_mov_times_gm2ub_input_scalar = 1;
   if (tiling_params->e_num_input_scalar >= byte || tiling_params->need_core_num_input_scalar == 1) {
     ComputeENumParamsInputScalarGEOneBlockNum(comm_paras, tiling_params);
   } else {
     ComputeENumParamsInputScalarLTOneBlockNum(comm_paras, tiling_params);
   }

   return;
 }

 static void ComputeENumParams(const CommParas4Int32& comm_paras, UnsortedSegmentSumTilingParamsInt32* tiling_params) {
   int32_t max_ele_num_one_ub_tensor = comm_paras.e_once_num;
   int32_t mask = (comm_paras.input_dytpe == ge::DT_INT32) ? MASK_INT32 : MASK_FP16;
   int32_t byte = (comm_paras.input_dytpe == ge::DT_INT32) ? INT32_BLOCK_NUM : FP16_BLOCK_NUM;
   if (tiling_params->e_num_input_scalar % byte == 0 && tiling_params->e_num_input_scalar > byte) {
     tiling_params->align_scalar = 0;
   } else if (tiling_params->e_num_input_scalar % byte != 0 && tiling_params->e_num_input_scalar > byte) {
     tiling_params->align_scalar =
         byte - (tiling_params->e_num_input_scalar - (tiling_params->e_num_input_scalar / byte) * byte);
   }

   if (tiling_params->e_num_input_scalar >= max_ele_num_one_ub_tensor && comm_paras.num_segments > 1) {
     ComputeENumParamsGEMaxEleNum(comm_paras, tiling_params);
   } else if (comm_paras.num_segments > 1 && tiling_params->e_num_input_scalar < max_ele_num_one_ub_tensor) {
     ComputeENumParamsLTMaxEleNum(comm_paras, tiling_params);
   } else if (comm_paras.num_segments == 1) {
     tiling_params->e_mov_times_gm2ub_input_scalar =
         UssCeilDiv(tiling_params->num_segments_front_core_input_scalar, max_ele_num_one_ub_tensor);
     tiling_params->e_mov_times_gm2ub_input_scalar_lastcore =
         UssCeilDiv(tiling_params->num_segments_last_core_input_scalar, max_ele_num_one_ub_tensor);
     // front core
     uint32_t mode = FRONT_CORE;
     NumSegmentOne(max_ele_num_one_ub_tensor, mask, comm_paras.ele_byte, mode, tiling_params);
     // last core
     mode = LAST_CORE;
     NumSegmentOne(max_ele_num_one_ub_tensor, mask, comm_paras.ele_byte, mode, tiling_params);
   }

   return;
 }

 static void InitTilingParams(UnsortedSegmentSumTilingParamsFp32* tiling_params, const int32_t num_segments) {
   // common params
   tiling_params->select_key_input_scalar = 0;
   tiling_params->need_core_num_input_scalar = 0;

   // input data params
   // front core
   tiling_params->input_ele_num_front_core_input_scalar = 0;
   // front part front core
   tiling_params->input_mov_times_gm2ub_front_part_front_core_input_scalar = 0;
   tiling_params->input_front_burst_len_front_part_front_core_input_scalar = 0;
   tiling_params->input_last_burst_len_front_part_front_core_input_scalar = 0;
   tiling_params->input_front_ele_num_ub_front_part_front_core_input_scalar = 0;
   tiling_params->input_last_ele_num_ub_front_part_front_core_input_scalar = 0;
   tiling_params->input_front_rows_front_part_front_core_input_scalar = 0;
   tiling_params->input_last_rows_front_part_front_core_input_scalar = 0;
   // last part front core
   tiling_params->input_mov_times_gm2ub_last_part_front_core_input_scalar = 0;
   tiling_params->input_front_burst_len_last_part_front_core_input_scalar = 0;
   tiling_params->input_last_burst_len_last_part_front_core_input_scalar = 0;
   tiling_params->input_front_ele_num_ub_last_part_front_core_input_scalar = 0;
   tiling_params->input_last_ele_num_ub_last_part_front_core_input_scalar = 0;
   tiling_params->input_front_rows_last_part_front_core_input_scalar = 0;
   tiling_params->input_last_rows_last_part_front_core_input_scalar = 0;
   // last core
   tiling_params->input_ele_num_last_core_input_scalar = 0;
   // front part last core
   tiling_params->input_mov_times_gm2ub_front_part_last_core_input_scalar = 0;
   tiling_params->input_front_burst_len_front_part_last_core_input_scalar = 0;
   tiling_params->input_last_burst_len_front_part_last_core_input_scalar = 0;
   tiling_params->input_front_ele_num_ub_front_part_last_core_input_scalar = 0;
   tiling_params->input_last_ele_num_ub_front_part_last_core_input_scalar = 0;
   tiling_params->input_front_rows_front_part_last_core_input_scalar = 0;
   tiling_params->input_last_rows_front_part_last_core_input_scalar = 0;
   // last part last core
   tiling_params->input_mov_times_gm2ub_last_part_last_core_input_scalar = 0;
   tiling_params->input_front_burst_len_last_part_last_core_input_scalar = 0;
   tiling_params->input_last_burst_len_last_part_last_core_input_scalar = 0;
   tiling_params->input_front_ele_num_ub_last_part_last_core_input_scalar = 0;
   tiling_params->input_last_ele_num_ub_last_part_last_core_input_scalar = 0;
   tiling_params->input_front_rows_last_part_last_core_input_scalar = 0;
   tiling_params->input_last_rows_last_part_last_core_input_scalar = 0;

   // e num params
   tiling_params->e_num_input_scalar = 0;
   tiling_params->e_mov_times_gm2ub_input_scalar = 0;
   tiling_params->e_ub2gm_front_burst_len_input_scalar = 0;
   tiling_params->e_num_front_part_input_scalar = 0;
   tiling_params->e_ub2gm_last_burst_len_input_scalar = 0;
   tiling_params->e_gm2ub_last_burst_len_input_scalar = 0;
   tiling_params->e_num_last_part_input_scalar = 0;

   // ids params
   tiling_params->ids_size_input_scalar = 0;
   tiling_params->ids_ele_num_front_core_input_scalar = 0;
   tiling_params->ids_mov_times_gm2ub_front_core_input_scalar = 0;
   tiling_params->ids_front_burst_len_front_core_input_scalar = 0;
   tiling_params->ids_last_burst_len_front_core_input_scalar = 0;
   tiling_params->ids_ele_num_ub_front_part_front_core_input_scalar = 0;
   tiling_params->ids_ele_num_ub_last_part_front_core_input_scalar = 0;
   tiling_params->ids_ele_num_last_core_input_scalar = 0;
   tiling_params->ids_mov_times_gm2ub_last_core_input_scalar = 0;
   tiling_params->ids_front_burst_len_last_core_input_scalar = 0;
   tiling_params->ids_last_burst_len_last_core_input_scalar = 0;
   tiling_params->ids_ele_num_ub_front_part_last_core_input_scalar = 0;
   tiling_params->ids_ele_num_ub_last_part_last_core_input_scalar = 0;

   // output init params
   tiling_params->output_ub_init_last_repeat_time_front_part_front_core_input_scalar = 0;
   tiling_params->output_ub_init_times_front_part_front_core_input_scalar = 0;
   tiling_params->output_ub_init_last_repeat_time_last_part_front_core_input_scalar = 0;
   tiling_params->output_ub_init_times_last_part_front_core_input_scalar = 0;
   tiling_params->output_ub_init_last_repeat_time_front_part_last_core_input_scalar = 0;
   tiling_params->output_ub_init_times_front_part_last_core_input_scalar = 0;
   tiling_params->output_ub_init_last_repeat_time_last_part_last_core_input_scalar = 0;
   tiling_params->output_ub_init_times_last_part_last_core_input_scalar = 0;
   tiling_params->input_last_axis_align_front_part_ele_num_input_scalar = 0;
   tiling_params->input_last_axis_align_floor_ele_num_input_scalar = 0;
   tiling_params->last_part_vadd_mask_input_scalar = 0;
   tiling_params->output_ub_init_last_row_last_repeat_time_front_part_front_core_input_scalar = 0;
   tiling_params->output_ub_init_last_row_times_front_part_front_core_input_scalar = 0;
   tiling_params->output_ub_init_last_row_last_repeat_time_last_part_front_core_input_scalar = 0;
   tiling_params->output_ub_init_last_row_times_last_part_front_core_input_scalar = 0;
   tiling_params->output_ub_init_last_row_last_repeat_time_front_part_last_core_input_scalar = 0;
   tiling_params->output_ub_init_last_row_times_front_part_last_core_input_scalar = 0;
   tiling_params->output_ub_init_last_row_last_repeat_time_last_part_last_core_input_scalar = 0;
   tiling_params->output_ub_init_last_row_times_last_part_last_core_input_scalar = 0;

   // num_segments init params
   tiling_params->num_segments = num_segments;

   tiling_params->tiling_core_num = 0;

   tiling_params->repeat_front_front_part_front_core = 0;
   tiling_params->col_sub_block_front_front_part_front_core = 0;
   tiling_params->repeat_last_front_part_front_core = 0;
   tiling_params->col_sub_block_last_front_part_front_core = 0;
   tiling_params->repeat_front_last_part_front_core = 0;
   tiling_params->col_sub_block_front_last_part_front_core = 0;
   tiling_params->repeat_last_last_part_front_core = 0;
   tiling_params->col_sub_block_last_last_part_front_core = 0;
   tiling_params->repeat_front_front_part_last_core = 0;
   tiling_params->col_sub_block_front_front_part_last_core = 0;
   tiling_params->repeat_last_front_part_last_core = 0;
   tiling_params->col_sub_block_last_front_part_last_core = 0;
   tiling_params->repeat_front_last_part_last_core = 0;
   tiling_params->col_sub_block_front_last_part_last_core = 0;
   tiling_params->repeat_last_last_part_last_core = 0;
   tiling_params->col_sub_block_last_last_part_last_core = 0;
   tiling_params->e_num_sub = 0;
   tiling_params->vadd_repeat_255 = 0;
   tiling_params->vadd_repeat_64 = 0;
   tiling_params->vadd_repeat_last = 0;
   tiling_params->max_cache_n_num = 0;
   tiling_params->repeat_remove_pad = 0;
   tiling_params->col_block_remove_pad = 0;
   tiling_params->cache_num_block = 0;

   return;
 }

 static void InitTilingParams(UnsortedSegmentSumTilingParamsInt32* tiling_params, const int32_t num_segments) {
   // common params
   tiling_params->select_key_input_scalar = 0;
   tiling_params->need_core_num_input_scalar = 0;
   tiling_params->num_segments_front_core_input_scalar = 0;
   tiling_params->num_segments_last_core_input_scalar = 0;

   // ids params
   tiling_params->ids_size_input_scalar = 0;
   tiling_params->ids_mov_times_gm2ub_input_scalar = 0;
   tiling_params->ids_ele_num_ub_front_part_input_scalar = 0;
   tiling_params->ids_front_burst_len_input_scalar = 0;
   tiling_params->ids_ele_num_ub_last_part_input_scalar = 0;
   tiling_params->ids_last_burst_len_input_scalar = 0;

   // e num params
   tiling_params->e_num_input_scalar = 0;
   tiling_params->e_mov_times_gm2ub_input_scalar = 0;
   tiling_params->e_ub2gm_front_burst_len_input_scalar = 0;
   tiling_params->e_num_front_part_input_scalar = 0;
   tiling_params->repeat_time_front_part_input_scalar = 0;
   tiling_params->e_ub2gm_last_burst_len_input_scalar = 0;
   tiling_params->e_num_last_part_input_scalar = 0;
   tiling_params->repeat_time_last_part_input_scalar = 0;
   tiling_params->align_scalar = 0;
   tiling_params->align_scalar_lastcore = 0;
   tiling_params->e_gm2ub_front_burst_len_input_scalar = 0;
   tiling_params->e_gm2ub_last_burst_len_input_scalar = 0;
   tiling_params->num_segment_max = 0;
   tiling_params->num_segment_max_time = 0;
   tiling_params->num_segment_max_time_lastcore = 0;
   tiling_params->front_num_segment = 0;
   tiling_params->front_num_segment_last = 0;
   tiling_params->front_num_segment_lastcore = 0;
   tiling_params->front_num_segment_last_lastcore = 0;
   tiling_params->e_ub2gm_front_burst_len_input_scalar_lastcore = 0;
   tiling_params->e_ub2gm_last_burst_len_input_scalar_lastcore = 0;
   tiling_params->repeat_times = 0;
   tiling_params->repeat_times_last_part = 0;
   tiling_params->repeat_times_last_part_lastcore = 0;
   tiling_params->e_mov_times_gm2ub_input_scalar_lastcore = 0;
   tiling_params->repeat_time_front_part_input_scalar_lastcore = 0;

   // num_segments init params
   tiling_params->num_segments = num_segments;

   tiling_params->tiling_core_num = 0;

   return;
 }

 static void PrintTilingParamsPartOne(const gert::TilingContext* context,
                                      const UnsortedSegmentSumTilingParamsFp32* tiling_param) {
   OP_LOGD(context->GetNodeName(), "tiling_param->select_key_input_scalar=%d", tiling_param->select_key_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->need_core_num_input_scalar=%d",
           tiling_param->need_core_num_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->input_ele_num_front_core_input_scalar=%d",
           tiling_param->input_ele_num_front_core_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->input_mov_times_gm2ub_front_part_front_core_input_scalar=%d",
           tiling_param->input_mov_times_gm2ub_front_part_front_core_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->input_front_burst_len_front_part_front_core_input_scalar=%d",
           tiling_param->input_front_burst_len_front_part_front_core_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->input_last_burst_len_front_part_front_core_input_scalar=%d",
           tiling_param->input_last_burst_len_front_part_front_core_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->input_front_ele_num_ub_front_part_front_core_input_scalar=%d",
           tiling_param->input_front_ele_num_ub_front_part_front_core_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->input_last_ele_num_ub_front_part_front_core_input_scalar=%d",
           tiling_param->input_last_ele_num_ub_front_part_front_core_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->input_front_rows_front_part_front_core_input_scalar=%d",
           tiling_param->input_front_rows_front_part_front_core_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->input_last_rows_front_part_front_core_input_scalar=%d",
           tiling_param->input_last_rows_front_part_front_core_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->input_mov_times_gm2ub_last_part_front_core_input_scalar=%d",
           tiling_param->input_mov_times_gm2ub_last_part_front_core_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->input_front_burst_len_last_part_front_core_input_scalar=%d",
           tiling_param->input_front_burst_len_last_part_front_core_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->input_last_burst_len_last_part_front_core_input_scalar=%d",
           tiling_param->input_last_burst_len_last_part_front_core_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->input_front_ele_num_ub_last_part_front_core_input_scalar=%d",
           tiling_param->input_front_ele_num_ub_last_part_front_core_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->input_last_ele_num_ub_last_part_front_core_input_scalar=%d",
           tiling_param->input_last_ele_num_ub_last_part_front_core_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->input_front_rows_last_part_front_core_input_scalar=%d",
           tiling_param->input_front_rows_last_part_front_core_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->input_last_rows_last_part_front_core_input_scalar=%d",
           tiling_param->input_last_rows_last_part_front_core_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->input_ele_num_last_core_input_scalar=%d",
           tiling_param->input_ele_num_last_core_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->input_mov_times_gm2ub_front_part_last_core_input_scalar=%d",
           tiling_param->input_mov_times_gm2ub_front_part_last_core_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->input_front_burst_len_front_part_last_core_input_scalar=%d",
           tiling_param->input_front_burst_len_front_part_last_core_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->input_last_burst_len_front_part_last_core_input_scalar=%d",
           tiling_param->input_last_burst_len_front_part_last_core_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->input_front_ele_num_ub_front_part_last_core_input_scalar=%d",
           tiling_param->input_front_ele_num_ub_front_part_last_core_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->input_last_ele_num_ub_front_part_last_core_input_scalar=%d",
           tiling_param->input_last_ele_num_ub_front_part_last_core_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->input_front_rows_front_part_last_core_input_scalar=%d",
           tiling_param->input_front_rows_front_part_last_core_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->input_last_rows_front_part_last_core_input_scalar=%d",
           tiling_param->input_last_rows_front_part_last_core_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->input_mov_times_gm2ub_last_part_last_core_input_scalar=%d",
           tiling_param->input_mov_times_gm2ub_last_part_last_core_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->input_front_burst_len_last_part_last_core_input_scalar=%d",
           tiling_param->input_front_burst_len_last_part_last_core_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->input_last_burst_len_last_part_last_core_input_scalar=%d",
           tiling_param->input_last_burst_len_last_part_last_core_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->input_front_ele_num_ub_last_part_last_core_input_scalar=%d",
           tiling_param->input_front_ele_num_ub_last_part_last_core_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->input_last_ele_num_ub_last_part_last_core_input_scalar=%d",
           tiling_param->input_last_ele_num_ub_last_part_last_core_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->input_front_rows_last_part_last_core_input_scalar=%d",
           tiling_param->input_front_rows_last_part_last_core_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->input_last_rows_last_part_last_core_input_scalar=%d",
           tiling_param->input_last_rows_last_part_last_core_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->e_num_input_scalar=%d", tiling_param->e_num_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->e_mov_times_gm2ub_input_scalar=%d",
           tiling_param->e_mov_times_gm2ub_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->e_ub2gm_front_burst_len_input_scalar=%d",
           tiling_param->e_ub2gm_front_burst_len_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->e_num_front_part_input_scalar=%d",
           tiling_param->e_num_front_part_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->e_ub2gm_last_burst_len_input_scalar=%d",
           tiling_param->e_ub2gm_last_burst_len_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->e_num_last_part_input_scalar=%d",
           tiling_param->e_num_last_part_input_scalar);
   return;
 }

 static void PrintTilingParamsPartTwo(const gert::TilingContext* context,
                                      const UnsortedSegmentSumTilingParamsFp32* tiling_param) {
   OP_LOGD(context->GetNodeName(), "tiling_param->ids_size_input_scalar=%d", tiling_param->ids_size_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->ids_ele_num_front_core_input_scalar=%d",
           tiling_param->ids_ele_num_front_core_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->ids_mov_times_gm2ub_front_core_input_scalar=%d",
           tiling_param->ids_mov_times_gm2ub_front_core_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->ids_front_burst_len_front_core_input_scalar=%d",
           tiling_param->ids_front_burst_len_front_core_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->ids_last_burst_len_front_core_input_scalar=%d",
           tiling_param->ids_last_burst_len_front_core_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->ids_ele_num_ub_front_part_front_core_input_scalar=%d",
           tiling_param->ids_ele_num_ub_front_part_front_core_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->ids_ele_num_ub_last_part_front_core_input_scalar=%d",
           tiling_param->ids_ele_num_ub_last_part_front_core_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->ids_ele_num_last_core_input_scalar=%d",
           tiling_param->ids_ele_num_last_core_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->ids_mov_times_gm2ub_last_core_input_scalar=%d",
           tiling_param->ids_mov_times_gm2ub_last_core_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->ids_front_burst_len_last_core_input_scalar=%d",
           tiling_param->ids_front_burst_len_last_core_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->ids_last_burst_len_last_core_input_scalar=%d",
           tiling_param->ids_last_burst_len_last_core_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->ids_ele_num_ub_front_part_last_core_input_scalar=%d",
           tiling_param->ids_ele_num_ub_front_part_last_core_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->ids_ele_num_ub_last_part_last_core_input_scalar=%d",
           tiling_param->ids_ele_num_ub_last_part_last_core_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->output_ub_init_last_repeat_time_front_part_front_core_input_scalar=%d",
           tiling_param->output_ub_init_last_repeat_time_front_part_front_core_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->output_ub_init_times_front_part_front_core_input_scalar=%d",
           tiling_param->output_ub_init_times_front_part_front_core_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->output_ub_init_last_repeat_time_last_part_front_core_input_scalar=%d",
           tiling_param->output_ub_init_last_repeat_time_last_part_front_core_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->output_ub_init_times_last_part_front_core_input_scalar=%d",
           tiling_param->output_ub_init_times_last_part_front_core_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->output_ub_init_last_repeat_time_front_part_last_core_input_scalar=%d",
           tiling_param->output_ub_init_last_repeat_time_front_part_last_core_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->output_ub_init_times_front_part_last_core_input_scalar=%d",
           tiling_param->output_ub_init_times_front_part_last_core_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->output_ub_init_last_repeat_time_last_part_last_core_input_scalar=%d",
           tiling_param->output_ub_init_last_repeat_time_last_part_last_core_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->output_ub_init_times_last_part_last_core_input_scalar=%d",
           tiling_param->output_ub_init_times_last_part_last_core_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->input_last_axis_align_front_part_ele_num_input_scalar=%d",
           tiling_param->input_last_axis_align_front_part_ele_num_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->input_last_axis_align_floor_ele_num_input_scalar=%d",
           tiling_param->input_last_axis_align_floor_ele_num_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->last_part_vadd_mask_input_scalar=%d",
           tiling_param->last_part_vadd_mask_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->e_gm2ub_last_burst_len_input_scalar=%d",
           tiling_param->e_gm2ub_last_burst_len_input_scalar);
   OP_LOGD(context->GetNodeName(),
           "tiling_param->output_ub_init_last_row_last_repeat_time_front_part_front_core_input_scalar=%d",
           tiling_param->output_ub_init_last_row_last_repeat_time_front_part_front_core_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->output_ub_init_last_row_times_front_part_front_core_input_scalar=%d",
           tiling_param->output_ub_init_last_row_times_front_part_front_core_input_scalar);
   OP_LOGD(context->GetNodeName(),
           "tiling_param->output_ub_init_last_row_last_repeat_time_last_part_front_core_input_scalar=%d",
           tiling_param->output_ub_init_last_row_last_repeat_time_last_part_front_core_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->output_ub_init_last_row_times_last_part_front_core_input_scalar=%d",
           tiling_param->output_ub_init_last_row_times_last_part_front_core_input_scalar);
   OP_LOGD(context->GetNodeName(),
           "tiling_param->output_ub_init_last_row_last_repeat_time_front_part_last_core_input_scalar=%d",
           tiling_param->output_ub_init_last_row_last_repeat_time_front_part_last_core_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->output_ub_init_last_row_times_front_part_last_core_input_scalar=%d",
           tiling_param->output_ub_init_last_row_times_front_part_last_core_input_scalar);
   OP_LOGD(context->GetNodeName(),
           "tiling_param->output_ub_init_last_row_last_repeat_time_last_part_last_core_input_scalar=%d",
           tiling_param->output_ub_init_last_row_last_repeat_time_last_part_last_core_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->output_ub_init_last_row_times_last_part_last_core_input_scalar=%d",
           tiling_param->output_ub_init_last_row_times_last_part_last_core_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->num_segments=%d", tiling_param->num_segments);

   OP_LOGD(context->GetNodeName(), "tiling_param->tiling_core_num=%d", tiling_param->tiling_core_num);

   return;
 }

 static void PrintTilingParamsPartThree(const gert::TilingContext* context,
                                        const UnsortedSegmentSumTilingParamsFp32* tiling_param) {
   OP_LOGD(context->GetNodeName(), "tiling_param->repeat_front_front_part_front_core=%d",
           tiling_param->repeat_front_front_part_front_core);
   OP_LOGD(context->GetNodeName(), "tiling_param->col_sub_block_front_front_part_front_core=%d",
           tiling_param->col_sub_block_front_front_part_front_core);
   OP_LOGD(context->GetNodeName(), "tiling_param->repeat_last_front_part_front_core=%d",
           tiling_param->repeat_last_front_part_front_core);
   OP_LOGD(context->GetNodeName(), "tiling_param->col_sub_block_last_front_part_front_core=%d",
           tiling_param->col_sub_block_last_front_part_front_core);
   OP_LOGD(context->GetNodeName(), "tiling_param->repeat_front_last_part_front_core=%d",
           tiling_param->repeat_front_last_part_front_core);
   OP_LOGD(context->GetNodeName(), "tiling_param->col_sub_block_front_last_part_front_core=%d",
           tiling_param->col_sub_block_front_last_part_front_core);
   OP_LOGD(context->GetNodeName(), "tiling_param->repeat_last_last_part_front_core=%d",
           tiling_param->repeat_last_last_part_front_core);
   OP_LOGD(context->GetNodeName(), "tiling_param->col_sub_block_last_last_part_front_core=%d",
           tiling_param->col_sub_block_last_last_part_front_core);
   OP_LOGD(context->GetNodeName(), "tiling_param->repeat_front_front_part_last_core=%d",
           tiling_param->repeat_front_front_part_last_core);
   OP_LOGD(context->GetNodeName(), "tiling_param->col_sub_block_front_front_part_last_core=%d",
           tiling_param->col_sub_block_front_front_part_last_core);
   OP_LOGD(context->GetNodeName(), "tiling_param->repeat_last_front_part_last_core=%d",
           tiling_param->repeat_last_front_part_last_core);
   OP_LOGD(context->GetNodeName(), "tiling_param->col_sub_block_last_front_part_last_core=%d",
           tiling_param->col_sub_block_last_front_part_last_core);
   OP_LOGD(context->GetNodeName(), "tiling_param->repeat_front_last_part_last_core=%d",
           tiling_param->repeat_front_last_part_last_core);
   OP_LOGD(context->GetNodeName(), "tiling_param->col_sub_block_front_last_part_last_core=%d",
           tiling_param->col_sub_block_front_last_part_last_core);
   OP_LOGD(context->GetNodeName(), "tiling_param->repeat_last_last_part_last_core=%d",
           tiling_param->repeat_last_last_part_last_core);
   OP_LOGD(context->GetNodeName(), "tiling_param->col_sub_block_last_last_part_last_core=%d",
           tiling_param->col_sub_block_last_last_part_last_core);
   OP_LOGD(context->GetNodeName(), "tiling_param->e_num_sub=%d", tiling_param->e_num_sub);
   OP_LOGD(context->GetNodeName(), "tiling_param->vadd_repeat_255=%d", tiling_param->vadd_repeat_255);
   OP_LOGD(context->GetNodeName(), "tiling_param->vadd_repeat_64=%d", tiling_param->vadd_repeat_64);
   OP_LOGD(context->GetNodeName(), "tiling_param->vadd_repeat_last=%d", tiling_param->vadd_repeat_last);
   OP_LOGD(context->GetNodeName(), "tiling_param->move_pad=%d", tiling_param->move_pad);
   OP_LOGD(context->GetNodeName(), "tiling_param->max_cache_n_num=%d", tiling_param->max_cache_n_num);
   OP_LOGD(context->GetNodeName(), "tiling_param->repeat_remove_pad=%d", tiling_param->repeat_remove_pad);
   OP_LOGD(context->GetNodeName(), "tiling_param->col_block_remove_pad=%d", tiling_param->col_block_remove_pad);
   OP_LOGD(context->GetNodeName(), "tiling_param->cache_num_block=%d", tiling_param->cache_num_block);
   return;
 }

 static void PrintTilingParams(gert::TilingContext* context, const UnsortedSegmentSumTilingParamsFp32* tiling_param) {
   PrintTilingParamsPartOne(context, tiling_param);
   PrintTilingParamsPartTwo(context, tiling_param);
   PrintTilingParamsPartThree(context, tiling_param);
   return;
 }

 static void PrintTilingParams(const gert::TilingContext* context,
                               const UnsortedSegmentSumTilingParamsInt32* tiling_param) {
   OP_LOGD(context->GetNodeName(), "tiling_param->select_key_input_scalar=%d", tiling_param->select_key_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->need_core_num_input_scalar=%d",
           tiling_param->need_core_num_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->num_segments_front_core_input_scalar=%d",
           tiling_param->num_segments_front_core_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->num_segments_last_core_input_scalar=%d",
           tiling_param->num_segments_last_core_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->ids_size_input_scalar=%d", tiling_param->ids_size_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->ids_mov_times_gm2ub_input_scalar=%d",
           tiling_param->ids_mov_times_gm2ub_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->ids_ele_num_ub_front_part_input_scalar=%d",
           tiling_param->ids_ele_num_ub_front_part_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->ids_front_burst_len_input_scalar=%d",
           tiling_param->ids_front_burst_len_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->ids_ele_num_ub_last_part_input_scalar=%d",
           tiling_param->ids_ele_num_ub_last_part_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->ids_last_burst_len_input_scalar=%d",
           tiling_param->ids_last_burst_len_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->e_num_input_scalar=%d", tiling_param->e_num_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->e_mov_times_gm2ub_input_scalar=%d",
           tiling_param->e_mov_times_gm2ub_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->e_ub2gm_front_burst_len_input_scalar=%d",
           tiling_param->e_ub2gm_front_burst_len_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->e_num_front_part_input_scalar=%d",
           tiling_param->e_num_front_part_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->repeat_time_front_part_input_scalar=%d",
           tiling_param->repeat_time_front_part_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->e_ub2gm_last_burst_len_input_scalar=%d",
           tiling_param->e_ub2gm_last_burst_len_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->e_num_last_part_input_scalar=%d",
           tiling_param->e_num_last_part_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->repeat_time_last_part_input_scalar=%d",
           tiling_param->repeat_time_last_part_input_scalar);

   OP_LOGD(context->GetNodeName(), "tiling_param->align_scalar=%d", tiling_param->align_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->align_scalar_lastcore=%d", tiling_param->align_scalar_lastcore);
   OP_LOGD(context->GetNodeName(), "tiling_param->e_gm2ub_front_burst_len_input_scalar=%d",
           tiling_param->e_gm2ub_front_burst_len_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->e_gm2ub_last_burst_len_input_scalar=%d",
           tiling_param->e_gm2ub_last_burst_len_input_scalar);
   OP_LOGD(context->GetNodeName(), "tiling_param->num_segment_max=%d", tiling_param->num_segment_max);
   OP_LOGD(context->GetNodeName(), "tiling_param->num_segment_max_time=%d", tiling_param->num_segment_max_time);
   OP_LOGD(context->GetNodeName(), "tiling_param->num_segment_max_time_lastcore=%d",
           tiling_param->num_segment_max_time_lastcore);
   OP_LOGD(context->GetNodeName(), "tiling_param->front_num_segment=%d", tiling_param->front_num_segment);
   OP_LOGD(context->GetNodeName(), "tiling_param->front_num_segment_last=%d", tiling_param->front_num_segment_last);
   OP_LOGD(context->GetNodeName(), "tiling_param->front_num_segment_lastcore=%d",
           tiling_param->front_num_segment_lastcore);
   OP_LOGD(context->GetNodeName(), "tiling_param->front_num_segment_last_lastcore=%d",
           tiling_param->front_num_segment_last_lastcore);
   OP_LOGD(context->GetNodeName(), "tiling_param->e_ub2gm_front_burst_len_input_scalar_lastcore=%d",
           tiling_param->e_ub2gm_front_burst_len_input_scalar_lastcore);
   OP_LOGD(context->GetNodeName(), "tiling_param->e_ub2gm_last_burst_len_input_scalar_lastcore=%d",
           tiling_param->e_ub2gm_last_burst_len_input_scalar_lastcore);
   OP_LOGD(context->GetNodeName(), "tiling_param->repeat_times=%d", tiling_param->repeat_times);
   OP_LOGD(context->GetNodeName(), "tiling_param->repeat_times_last_part=%d", tiling_param->repeat_times_last_part);
   OP_LOGD(context->GetNodeName(), "tiling_param->repeat_times_last_part_lastcore=%d",
           tiling_param->repeat_times_last_part_lastcore);
   OP_LOGD(context->GetNodeName(), "tiling_param->e_mov_times_gm2ub_input_scalar_lastcore=%d",
           tiling_param->e_mov_times_gm2ub_input_scalar_lastcore);
   OP_LOGD(context->GetNodeName(), "tiling_param->repeat_time_front_part_input_scalar_lastcore=%d",
           tiling_param->repeat_time_front_part_input_scalar_lastcore);
   OP_LOGD(context->GetNodeName(), "tiling_param->num_segments=%d", tiling_param->num_segments);

   OP_LOGD(context->GetNodeName(), "tiling_param->tiling_core_num=%d", tiling_param->tiling_core_num);

   return;
 }

 ge::graphStatus CalcSpecialTiling(gert::TilingContext* context, const uint32_t num_segments) {
   auto tiling_params = context->GetTilingData<UnsortedSegmentSumTilingParamsFp32>();
   OP_CHECK_NULL_WITH_CONTEXT(context, tiling_params);

   auto compile_info = static_cast<const TilingPrepareForUnsortedSegmentSumCompileInfo*>(context->GetCompileInfo());
   OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);

   InitTilingParams(tiling_params, num_segments);
   tiling_params->select_key_input_scalar = 0;
   tiling_params->need_core_num_input_scalar = 1;
   tiling_params->tiling_core_num = compile_info->core_num;
   // cout tiling params
   PrintTilingParams(context, tiling_params);
   // BlockDim, core num used in tik op
   context->SetBlockDim(tiling_params->need_core_num_input_scalar);

   return ge::GRAPH_SUCCESS;
 }

 void CalcTilingFloat4LastAxisSmallE(UnsortedSegmentSumTilingParamsFp32* tiling_params, const CommParas& comm_paras,
                                     const int32_t x_ub_size, const EleByte input_ele_byte, const int32_t e_size) {
   // aign small e
   // e num params
   tiling_params->e_mov_times_gm2ub_input_scalar = 1;
   tiling_params->e_num_front_part_input_scalar = tiling_params->e_num_input_scalar;
   tiling_params->e_num_last_part_input_scalar = tiling_params->e_num_input_scalar;
   tiling_params->e_ub2gm_front_burst_len_input_scalar =
       tiling_params->e_num_front_part_input_scalar * input_ele_byte / BYTE_BLOCK;
   tiling_params->e_ub2gm_last_burst_len_input_scalar = tiling_params->e_ub2gm_front_burst_len_input_scalar;
   // input data params
   InputParamsGm2UbOut gm_ub_out_params = {0};

   // front part front core
   gm_ub_out_params.mode = FRONT_PART_FRONT_CORE;
   ComputeInputParamsMovGm2ub(x_ub_size, input_ele_byte, e_size, tiling_params, gm_ub_out_params);
   SetInputParamsMovGm2ub(gm_ub_out_params, tiling_params);

   // last part front core
   gm_ub_out_params.mode = LAST_PART_FRONT_CORE;
   ComputeInputParamsMovGm2ub(x_ub_size, input_ele_byte, e_size, tiling_params, gm_ub_out_params);
   SetInputParamsMovGm2ub(gm_ub_out_params, tiling_params);

   // front part last core
   gm_ub_out_params.mode = FRONT_PART_LAST_CORE;
   ComputeInputParamsMovGm2ub(x_ub_size, input_ele_byte, e_size, tiling_params, gm_ub_out_params);
   SetInputParamsMovGm2ub(gm_ub_out_params, tiling_params);

   // last part last core
   gm_ub_out_params.mode = LAST_PART_LAST_CORE;
   ComputeInputParamsMovGm2ub(x_ub_size, input_ele_byte, e_size, tiling_params, gm_ub_out_params);
   SetInputParamsMovGm2ub(gm_ub_out_params, tiling_params);
   bool perf = CheckHighPerf(comm_paras.ub_size, comm_paras.e_num, comm_paras.impl_mode);
   if (perf) {
     int32_t num = comm_paras.ub_tensor_ele_num / comm_paras.e_num;
     tiling_params->output_ub_init_last_repeat_time_front_part_front_core_input_scalar = num;
     if (num > comm_paras.num_segments) {
       tiling_params->output_ub_init_last_repeat_time_front_part_front_core_input_scalar = comm_paras.num_segments;
     }
     ComputeDiv(comm_paras.e_num, MASK_FP32 * MAX_REPEAT_TIME,
                tiling_params->output_ub_init_times_front_part_front_core_input_scalar,
                tiling_params->output_ub_init_last_repeat_time_last_part_front_core_input_scalar);
     ComputeDiv(tiling_params->output_ub_init_last_repeat_time_last_part_front_core_input_scalar, MASK_FP32,
                tiling_params->last_part_vadd_mask_input_scalar,
                tiling_params->output_ub_init_last_repeat_time_front_part_last_core_input_scalar);
     if (num <= comm_paras.num_segments) {
       tiling_params->output_ub_init_times_front_part_last_core_input_scalar =
           (comm_paras.e_num * num * MULTI) / BYTE_BLOCK;
     } else {
       tiling_params->output_ub_init_times_front_part_last_core_input_scalar =
           (comm_paras.e_num * comm_paras.num_segments * MULTI) / BYTE_BLOCK;
     }
   }
   return;
 }

 void CalcTilingFloat4LastAxisOne(UnsortedSegmentSumTilingParamsFp32* tiling_params,
                                  const int32_t output_ub_ele_num_one_row) {
   // last axis is one

   // e num params
   tiling_params->e_num_input_scalar = 1;
   tiling_params->e_mov_times_gm2ub_input_scalar = 1;
   tiling_params->e_ub2gm_front_burst_len_input_scalar = 1;

   // input data params
   // front part front core
   tiling_params->input_front_burst_len_front_part_front_core_input_scalar =
       tiling_params->ids_front_burst_len_front_core_input_scalar;
   tiling_params->input_front_ele_num_ub_front_part_front_core_input_scalar =
       tiling_params->ids_ele_num_ub_front_part_front_core_input_scalar;
   tiling_params->input_front_rows_front_part_front_core_input_scalar =
       tiling_params->input_front_ele_num_ub_front_part_front_core_input_scalar;
   // last part front core
   tiling_params->input_front_burst_len_last_part_front_core_input_scalar =
       tiling_params->ids_last_burst_len_front_core_input_scalar;
   tiling_params->input_front_ele_num_ub_last_part_front_core_input_scalar =
       tiling_params->ids_ele_num_ub_last_part_front_core_input_scalar;
   tiling_params->input_front_rows_last_part_front_core_input_scalar =
       tiling_params->input_front_ele_num_ub_last_part_front_core_input_scalar;
   // front part last core
   tiling_params->input_front_burst_len_front_part_last_core_input_scalar =
       tiling_params->ids_front_burst_len_last_core_input_scalar;
   tiling_params->input_front_ele_num_ub_front_part_last_core_input_scalar =
       tiling_params->ids_ele_num_ub_front_part_last_core_input_scalar;
   tiling_params->input_front_rows_front_part_last_core_input_scalar =
       tiling_params->input_front_ele_num_ub_front_part_last_core_input_scalar;
   // last part last core
   tiling_params->input_front_burst_len_last_part_last_core_input_scalar =
       tiling_params->ids_last_burst_len_last_core_input_scalar;
   tiling_params->input_front_ele_num_ub_last_part_last_core_input_scalar =
       tiling_params->ids_ele_num_ub_last_part_last_core_input_scalar;
   tiling_params->input_front_rows_last_part_last_core_input_scalar =
       tiling_params->input_front_ele_num_ub_last_part_last_core_input_scalar;

   // output init params
   // front part front core
   ComputeInitOutputUbParams(tiling_params->ids_ele_num_ub_front_part_front_core_input_scalar, output_ub_ele_num_one_row,
                             tiling_params, 0);
   // last part front core
   ComputeInitOutputUbParams(tiling_params->ids_ele_num_ub_last_part_front_core_input_scalar, output_ub_ele_num_one_row,
                             tiling_params, PART_TWO);
   // front part last core
   ComputeInitOutputUbParams(tiling_params->ids_ele_num_ub_front_part_last_core_input_scalar, output_ub_ele_num_one_row,
                             tiling_params, PART_FOUR);
   // last part last core
   ComputeInitOutputUbParams(tiling_params->ids_ele_num_ub_last_part_last_core_input_scalar, output_ub_ele_num_one_row,
                             tiling_params, PART_SIX);

   return;
 }

 void CalcTilingFloat4LastAxisNotAlignSmallE(UnsortedSegmentSumTilingParamsFp32* tiling_params, const int32_t x_ub_size,
                                             const EleByte input_ele_byte, const CommParas& comm_paras,
                                             const int32_t output_ub_ele_num_one_row) {
   // not align small e
   // e num params
   int32_t e_size = comm_paras.e_num;
   tiling_params->e_mov_times_gm2ub_input_scalar = 1;
   tiling_params->e_ub2gm_front_burst_len_input_scalar = e_size * input_ele_byte / BYTE_BLOCK;

   tiling_params->e_num_front_part_input_scalar = e_size / FP32_ELE_NUM_ALIGN_32B * FP32_ELE_NUM_ALIGN_32B;
   tiling_params->e_ub2gm_last_burst_len_input_scalar = 1;
   tiling_params->e_num_last_part_input_scalar = e_size - tiling_params->e_num_front_part_input_scalar;

   // input data params
   InputParamsGm2UbOut gm_ub_out_params = {0};

   // front part front core
   gm_ub_out_params.mode = FRONT_PART_FRONT_CORE;
   ComputeInputParamsMovGm2ub(x_ub_size, input_ele_byte, e_size, tiling_params, gm_ub_out_params);
   SetInputParamsMovGm2ub(gm_ub_out_params, tiling_params);

   // last part front core
   gm_ub_out_params.mode = LAST_PART_FRONT_CORE;
   ComputeInputParamsMovGm2ub(x_ub_size, input_ele_byte, e_size, tiling_params, gm_ub_out_params);
   SetInputParamsMovGm2ub(gm_ub_out_params, tiling_params);

   // front part last core
   gm_ub_out_params.mode = FRONT_PART_LAST_CORE;
   ComputeInputParamsMovGm2ub(x_ub_size, input_ele_byte, e_size, tiling_params, gm_ub_out_params);
   SetInputParamsMovGm2ub(gm_ub_out_params, tiling_params);

   // last part last core
   gm_ub_out_params.mode = LAST_PART_LAST_CORE;
   ComputeInputParamsMovGm2ub(x_ub_size, input_ele_byte, e_size, tiling_params, gm_ub_out_params);
   SetInputParamsMovGm2ub(gm_ub_out_params, tiling_params);

   // output init params
   // front row front part front core
   ComputeInitOutputUbParams(tiling_params->input_front_rows_front_part_front_core_input_scalar,
                             output_ub_ele_num_one_row, tiling_params, 0);
   // last row front part front core
   ComputeInitOutputUbParams(tiling_params->input_last_rows_front_part_front_core_input_scalar,
                             output_ub_ele_num_one_row, tiling_params, 1);
   // front row last part front core
   ComputeInitOutputUbParams(tiling_params->input_front_rows_last_part_front_core_input_scalar,
                             output_ub_ele_num_one_row, tiling_params, PART_TWO);
   // last row last part front core
   ComputeInitOutputUbParams(tiling_params->input_last_rows_last_part_front_core_input_scalar, output_ub_ele_num_one_row,
                             tiling_params, PART_THREE);
   // front row front part last core
   ComputeInitOutputUbParams(tiling_params->input_front_rows_front_part_last_core_input_scalar,
                             output_ub_ele_num_one_row, tiling_params, PART_FOUR);
   // last row front part last core
   ComputeInitOutputUbParams(tiling_params->input_last_rows_front_part_last_core_input_scalar, output_ub_ele_num_one_row,
                             tiling_params, PART_FIVE);
   // front row last part last core
   ComputeInitOutputUbParams(tiling_params->input_front_rows_last_part_last_core_input_scalar, output_ub_ele_num_one_row,
                             tiling_params, PART_SIX);
   // last row last part last core
   ComputeInitOutputUbParams(tiling_params->input_last_rows_last_part_last_core_input_scalar, output_ub_ele_num_one_row,
                             tiling_params, PART_SERVERN);
   tiling_params->input_last_axis_align_front_part_ele_num_input_scalar =
       e_size / FP32_ELE_NUM_ALIGN_32B * FP32_ELE_NUM_ALIGN_32B;

   tiling_params->last_part_vadd_mask_input_scalar =
       e_size - tiling_params->input_last_axis_align_front_part_ele_num_input_scalar;
   ComputeDiv(comm_paras.e_num, MASK_FP32 * MAX_REPEAT_TIME, tiling_params->vadd_repeat_255,
              tiling_params->vadd_repeat_last);
   ComputeDiv(tiling_params->vadd_repeat_last, MASK_FP32, tiling_params->vadd_repeat_64,
              tiling_params->vadd_repeat_last);
 }

 void CalcTilingFloat4LastAxisAlignBigE(const int32_t x_ub_size, const EleByte input_ele_byte, const int32_t e_size,
                                        const int32_t ub_tensor_ele_num,
                                        UnsortedSegmentSumTilingParamsFp32* tiling_params) {
   // align big e
   // e num params
   tiling_params->e_mov_times_gm2ub_input_scalar = UssCeilDiv(e_size, ub_tensor_ele_num);
   tiling_params->e_ub2gm_front_burst_len_input_scalar = x_ub_size / BYTE_BLOCK;
   tiling_params->e_num_front_part_input_scalar = ub_tensor_ele_num;
   tiling_params->e_num_last_part_input_scalar = ComputeDivRemainders(
       e_size, tiling_params->e_num_front_part_input_scalar, tiling_params->e_mov_times_gm2ub_input_scalar - 1);
   tiling_params->e_ub2gm_last_burst_len_input_scalar =
       tiling_params->e_num_last_part_input_scalar * input_ele_byte / BYTE_BLOCK;

   // input data params
   // front part front core
   tiling_params->input_mov_times_gm2ub_front_part_front_core_input_scalar =
       tiling_params->ids_ele_num_ub_front_part_front_core_input_scalar;
   tiling_params->input_front_ele_num_ub_front_part_front_core_input_scalar = ub_tensor_ele_num;
   tiling_params->input_last_ele_num_ub_front_part_front_core_input_scalar = tiling_params->e_num_last_part_input_scalar;
   // last part front core
   tiling_params->input_mov_times_gm2ub_last_part_front_core_input_scalar =
       tiling_params->ids_ele_num_ub_last_part_front_core_input_scalar;
   tiling_params->input_front_ele_num_ub_last_part_front_core_input_scalar = ub_tensor_ele_num;
   tiling_params->input_last_ele_num_ub_last_part_front_core_input_scalar = tiling_params->e_num_last_part_input_scalar;
   // front part last core
   tiling_params->input_mov_times_gm2ub_front_part_last_core_input_scalar =
       tiling_params->ids_ele_num_ub_front_part_last_core_input_scalar;
   tiling_params->input_front_ele_num_ub_front_part_last_core_input_scalar = ub_tensor_ele_num;
   tiling_params->input_last_ele_num_ub_front_part_last_core_input_scalar = tiling_params->e_num_last_part_input_scalar;
   // last part last core
   tiling_params->input_mov_times_gm2ub_last_part_last_core_input_scalar =
       tiling_params->ids_ele_num_ub_last_part_last_core_input_scalar;
   tiling_params->input_front_ele_num_ub_last_part_last_core_input_scalar = ub_tensor_ele_num;
   tiling_params->input_last_ele_num_ub_last_part_last_core_input_scalar = tiling_params->e_num_last_part_input_scalar;

   return;
 }

 void CalcTilingFloat4LastAxisNotAlignBigE(UnsortedSegmentSumTilingParamsFp32* tiling_params, const int32_t x_ub_size,
                                           const EleByte input_ele_byte, const int32_t e_size,
                                           const int32_t ub_tensor_ele_num) {
   // not align big e
   // e num params
   tiling_params->e_mov_times_gm2ub_input_scalar = UssCeilDiv(e_size, ub_tensor_ele_num);
   tiling_params->e_ub2gm_front_burst_len_input_scalar = x_ub_size / BYTE_BLOCK;
   tiling_params->e_num_front_part_input_scalar = ub_tensor_ele_num;
   tiling_params->e_num_last_part_input_scalar = ComputeDivRemainders(
       e_size, tiling_params->e_num_front_part_input_scalar, tiling_params->e_mov_times_gm2ub_input_scalar - 1);
   tiling_params->e_ub2gm_last_burst_len_input_scalar =
       tiling_params->e_num_last_part_input_scalar * input_ele_byte / BYTE_BLOCK;
   tiling_params->e_gm2ub_last_burst_len_input_scalar =
       UssCeilDiv(tiling_params->e_num_last_part_input_scalar * input_ele_byte, BYTE_BLOCK);

   // input data params
   // front part front core
   tiling_params->input_mov_times_gm2ub_front_part_front_core_input_scalar =
       tiling_params->ids_ele_num_ub_front_part_front_core_input_scalar;
   tiling_params->input_front_ele_num_ub_front_part_front_core_input_scalar = ub_tensor_ele_num;
   tiling_params->input_last_ele_num_ub_front_part_front_core_input_scalar = tiling_params->e_num_last_part_input_scalar;
   // last part front core
   tiling_params->input_mov_times_gm2ub_last_part_front_core_input_scalar =
       tiling_params->ids_ele_num_ub_last_part_front_core_input_scalar;
   tiling_params->input_front_ele_num_ub_last_part_front_core_input_scalar = ub_tensor_ele_num;
   tiling_params->input_last_ele_num_ub_last_part_front_core_input_scalar = tiling_params->e_num_last_part_input_scalar;
   // front part last core
   tiling_params->input_mov_times_gm2ub_front_part_last_core_input_scalar =
       tiling_params->ids_ele_num_ub_front_part_last_core_input_scalar;
   tiling_params->input_front_ele_num_ub_front_part_last_core_input_scalar = ub_tensor_ele_num;
   tiling_params->input_last_ele_num_ub_front_part_last_core_input_scalar = tiling_params->e_num_last_part_input_scalar;
   // last part last core
   tiling_params->input_mov_times_gm2ub_last_part_last_core_input_scalar =
       tiling_params->ids_ele_num_ub_last_part_last_core_input_scalar;
   tiling_params->input_front_ele_num_ub_last_part_last_core_input_scalar = ub_tensor_ele_num;
   tiling_params->input_last_ele_num_ub_last_part_last_core_input_scalar = tiling_params->e_num_last_part_input_scalar;

   // output init params
   tiling_params->input_last_axis_align_front_part_ele_num_input_scalar =
       tiling_params->e_num_last_part_input_scalar / FP32_ELE_NUM_ALIGN_32B * FP32_ELE_NUM_ALIGN_32B;
   tiling_params->input_last_axis_align_floor_ele_num_input_scalar =
       UssCeil(tiling_params->e_num_last_part_input_scalar, FP32_ELE_NUM_ALIGN_32B);
   tiling_params->last_part_vadd_mask_input_scalar =
       tiling_params->e_num_last_part_input_scalar -
       tiling_params->input_last_axis_align_front_part_ele_num_input_scalar;

   return;
 }

 void CalcTilingFloat4LastAxisModify(UnsortedSegmentSumTilingParamsFp32* tiling_params) {
   // last axis is one modify
   // e num params
   tiling_params->e_num_input_scalar = 1;
   tiling_params->e_mov_times_gm2ub_input_scalar = 1;
   tiling_params->e_ub2gm_front_burst_len_input_scalar = 1;

   // input data params
   // front part front core
   tiling_params->input_front_burst_len_front_part_front_core_input_scalar =
       tiling_params->ids_front_burst_len_front_core_input_scalar;
   tiling_params->input_front_ele_num_ub_front_part_front_core_input_scalar =
       tiling_params->ids_ele_num_ub_front_part_front_core_input_scalar;
   tiling_params->input_front_rows_front_part_front_core_input_scalar =
       tiling_params->input_front_ele_num_ub_front_part_front_core_input_scalar;
   // last part front core
   tiling_params->input_front_burst_len_last_part_front_core_input_scalar =
       tiling_params->ids_last_burst_len_front_core_input_scalar;
   tiling_params->input_front_ele_num_ub_last_part_front_core_input_scalar =
       tiling_params->ids_ele_num_ub_last_part_front_core_input_scalar;
   tiling_params->input_front_rows_last_part_front_core_input_scalar =
       tiling_params->input_front_ele_num_ub_last_part_front_core_input_scalar;
   // front part last core
   tiling_params->input_front_burst_len_front_part_last_core_input_scalar =
       tiling_params->ids_front_burst_len_last_core_input_scalar;
   tiling_params->input_front_ele_num_ub_front_part_last_core_input_scalar =
       tiling_params->ids_ele_num_ub_front_part_last_core_input_scalar;
   tiling_params->input_front_rows_front_part_last_core_input_scalar =
       tiling_params->input_front_ele_num_ub_front_part_last_core_input_scalar;
   // last part last core
   tiling_params->input_front_burst_len_last_part_last_core_input_scalar =
       tiling_params->ids_last_burst_len_last_core_input_scalar;
   tiling_params->input_front_ele_num_ub_last_part_last_core_input_scalar =
       tiling_params->ids_ele_num_ub_last_part_last_core_input_scalar;
   tiling_params->input_front_rows_last_part_last_core_input_scalar =
       tiling_params->input_front_ele_num_ub_last_part_last_core_input_scalar;

   // output init params
   tiling_params->output_ub_init_times_front_part_front_core_input_scalar =
       UssCeilDiv(tiling_params->ids_ele_num_ub_front_part_front_core_input_scalar, MASK_FP32);
   tiling_params->output_ub_init_times_last_part_front_core_input_scalar =
       UssCeilDiv(tiling_params->ids_ele_num_ub_last_part_front_core_input_scalar, MASK_FP32);
   tiling_params->output_ub_init_times_front_part_last_core_input_scalar =
       UssCeilDiv(tiling_params->ids_ele_num_ub_front_part_last_core_input_scalar, MASK_FP32);
   tiling_params->output_ub_init_times_last_part_last_core_input_scalar =
       UssCeilDiv(tiling_params->ids_ele_num_ub_last_part_last_core_input_scalar, MASK_FP32);
   tiling_params->last_part_vadd_mask_input_scalar =
       tiling_params->ids_ele_num_ub_last_part_last_core_input_scalar -
       (tiling_params->output_ub_init_times_last_part_last_core_input_scalar - 1) * MASK_FP32;

   return;
 }

 void CalcTilingFloat4LastAxisOneMulti(UnsortedSegmentSumTilingParamsFp32* tiling_params) {
   // last axis is one multi 64

   // e num params
   tiling_params->e_num_input_scalar = 1;
   tiling_params->e_mov_times_gm2ub_input_scalar = 1;
   tiling_params->e_ub2gm_front_burst_len_input_scalar = 1;

   // input data params
   // front part front core
   tiling_params->input_front_burst_len_front_part_front_core_input_scalar =
       tiling_params->ids_front_burst_len_front_core_input_scalar;
   tiling_params->input_front_ele_num_ub_front_part_front_core_input_scalar =
       tiling_params->ids_ele_num_ub_front_part_front_core_input_scalar;
   tiling_params->input_front_rows_front_part_front_core_input_scalar =
       tiling_params->input_front_ele_num_ub_front_part_front_core_input_scalar;
   // last part front core
   tiling_params->input_front_burst_len_last_part_front_core_input_scalar =
       tiling_params->ids_last_burst_len_front_core_input_scalar;
   tiling_params->input_front_ele_num_ub_last_part_front_core_input_scalar =
       tiling_params->ids_ele_num_ub_last_part_front_core_input_scalar;
   tiling_params->input_front_rows_last_part_front_core_input_scalar =
       tiling_params->input_front_ele_num_ub_last_part_front_core_input_scalar;
   // front part last core
   tiling_params->input_front_burst_len_front_part_last_core_input_scalar =
       tiling_params->ids_front_burst_len_last_core_input_scalar;
   tiling_params->input_front_ele_num_ub_front_part_last_core_input_scalar =
       tiling_params->ids_ele_num_ub_front_part_last_core_input_scalar;
   tiling_params->input_front_rows_front_part_last_core_input_scalar =
       tiling_params->input_front_ele_num_ub_front_part_last_core_input_scalar;
   // last part last core
   tiling_params->input_front_burst_len_last_part_last_core_input_scalar =
       tiling_params->ids_last_burst_len_last_core_input_scalar;
   tiling_params->input_front_ele_num_ub_last_part_last_core_input_scalar =
       tiling_params->ids_ele_num_ub_last_part_last_core_input_scalar;
   tiling_params->input_front_rows_last_part_last_core_input_scalar =
       tiling_params->input_front_ele_num_ub_last_part_last_core_input_scalar;

   // output init params
   // front part front core
   tiling_params->output_ub_init_times_front_part_front_core_input_scalar =
       tiling_params->ids_ele_num_ub_front_part_front_core_input_scalar / (MASK_FP32 * MULTI);
   tiling_params->output_ub_init_last_repeat_time_front_part_front_core_input_scalar =
       ComputeDivRemainders(tiling_params->ids_ele_num_ub_front_part_front_core_input_scalar, MASK_FP32 * MULTI,
                            tiling_params->output_ub_init_times_front_part_front_core_input_scalar) /
       MASK_FP32;
   // last part front core
   tiling_params->output_ub_init_times_last_part_front_core_input_scalar =
       tiling_params->ids_ele_num_ub_last_part_front_core_input_scalar / (MASK_FP32 * MULTI);
   tiling_params->output_ub_init_last_repeat_time_last_part_front_core_input_scalar =
       ComputeDivRemainders(tiling_params->ids_ele_num_ub_last_part_front_core_input_scalar, MASK_FP32 * MULTI,
                            tiling_params->output_ub_init_times_last_part_front_core_input_scalar) /
       MASK_FP32;
   // front part last core
   tiling_params->output_ub_init_times_front_part_last_core_input_scalar =
       tiling_params->ids_ele_num_ub_front_part_last_core_input_scalar / (MASK_FP32 * MULTI);
   tiling_params->output_ub_init_last_repeat_time_front_part_last_core_input_scalar =
       ComputeDivRemainders(tiling_params->ids_ele_num_ub_front_part_last_core_input_scalar, MASK_FP32 * MULTI,
                            tiling_params->output_ub_init_times_front_part_last_core_input_scalar) /
       MASK_FP32;
   // last part last core
   // multi 64 part
   tiling_params->output_ub_init_times_last_part_last_core_input_scalar =
       tiling_params->ids_ele_num_ub_last_part_last_core_input_scalar / (MASK_FP32 * MULTI);
   // single 64 part
   tiling_params->output_ub_init_last_repeat_time_last_part_last_core_input_scalar =
       ComputeDivRemainders(tiling_params->ids_ele_num_ub_last_part_last_core_input_scalar, MASK_FP32 * MULTI,
                            tiling_params->output_ub_init_times_last_part_last_core_input_scalar) /
       MASK_FP32;
   // last mask part
   tiling_params->last_part_vadd_mask_input_scalar =
       ComputeDivRemainders(tiling_params->ids_ele_num_ub_last_part_last_core_input_scalar, MASK_FP32 * MULTI,
                            tiling_params->output_ub_init_times_last_part_last_core_input_scalar) -
       tiling_params->output_ub_init_last_repeat_time_last_part_last_core_input_scalar * MASK_FP32;

   return;
 }

 void CalcTilingFloat4NumSegmentOne(UnsortedSegmentSumTilingParamsFp32* tiling_params, const EleByte input_ele_byte,
                                    const int32_t x_ub_size, const int32_t e_size, const int32_t ub_tensor_ele_num) {
   tiling_params->e_mov_times_gm2ub_input_scalar = UssCeilDiv(e_size, ub_tensor_ele_num);
   tiling_params->e_ub2gm_front_burst_len_input_scalar = x_ub_size / BYTE_BLOCK;
   tiling_params->e_num_front_part_input_scalar = ub_tensor_ele_num;
   tiling_params->e_num_last_part_input_scalar = ComputeDivRemainders(
       e_size, tiling_params->e_num_front_part_input_scalar, tiling_params->e_mov_times_gm2ub_input_scalar - 1);
   tiling_params->e_ub2gm_last_burst_len_input_scalar =
       UssCeilDiv(tiling_params->e_num_last_part_input_scalar * input_ele_byte, BYTE_BLOCK);

   return;
 }

 ge::graphStatus CalcTiling4Float(gert::TilingContext* context, const int32_t e_size, const int32_t num_segments,
                                  const int32_t input_size, const int32_t ids_size,
                                  const int32_t output_ub_ele_num_one_row, const EleByte input_ele_byte,
                                  const EleByte ids_ele_byte) {
   auto compile_info = static_cast<const TilingPrepareForUnsortedSegmentSumCompileInfo*>(context->GetCompileInfo());
   OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);

   int32_t ids_ub_size = 0;
   int32_t x_ub_size = 0;
   int32_t x_ub_size_pad = 0;
   CommParas commParas = {0};
   commParas.e_num = e_size;
   commParas.x_e_num = input_size;
   commParas.num_segments = num_segments;
   commParas.ub_size = compile_info->ub_size;
   commParas.impl_mode = compile_info->impl_mode;

   ComputeUbTensorSize(commParas, x_ub_size, ids_ub_size, x_ub_size_pad);
   OP_LOGD(context->GetNodeName(), "x_ub_size_pad is %d", x_ub_size_pad);
   OP_LOGD(context->GetNodeName(), "ub_tensor_size=%d, %d", ids_ub_size, x_ub_size);
   OP_CHECK_IF(
       input_ele_byte == 0,
       OP_LOGE(context->GetNodeName(), "CalcTiling4Float check failed input_ele_byte is 0."),
       return ge::GRAPH_FAILED);
   int32_t ub_tensor_ele_num = x_ub_size / input_ele_byte;
   int32_t ub_tensor_ele_num_pad = x_ub_size_pad / input_ele_byte;
   commParas.ub_tensor_ele_num = ub_tensor_ele_num;
   commParas.ub_tensor_ele_num_pad = ub_tensor_ele_num_pad;
   commParas.e_size_align = UssCeil(e_size, FP32_ELE_NUM_ALIGN_32B);
   // fp32 tiling params
   auto tiling_params = context->GetTilingData<UnsortedSegmentSumTilingParamsFp32>();
   OP_CHECK_NULL_WITH_CONTEXT(context, tiling_params);
   InitTilingParams(tiling_params, num_segments);
   tiling_params->tiling_core_num = compile_info->core_num;
   OP_LOGD(context->GetNodeName(), "CalcTiling4Float set tiling core num: %d", tiling_params->tiling_core_num);

   // select key
   OP_CHECK_IF(!GetTilingMode(commParas, x_ub_size, tiling_params->select_key_input_scalar),
                   OP_LOGE(context->GetNodeName(), "GetTilingMode failed."),
                   return ge::GRAPH_FAILED);
   tiling_params->e_num_input_scalar = e_size;
   tiling_params->input_last_axis_align_floor_ele_num_input_scalar = commParas.e_size_align;
   tiling_params->e_num_sub = commParas.e_size_align - e_size;
   tiling_params->move_pad = e_size * input_ele_byte;
   tiling_params->max_cache_n_num = ub_tensor_ele_num_pad / commParas.e_size_align;
   if (tiling_params->max_cache_n_num > num_segments) {
     tiling_params->max_cache_n_num = num_segments;
   }
   tiling_params->repeat_remove_pad = tiling_params->max_cache_n_num / PAD_ONCE;
   int32_t remain = ComputeDivRemainders(tiling_params->max_cache_n_num, PAD_ONCE, tiling_params->repeat_remove_pad);
   tiling_params->col_block_remove_pad = UssCeilDiv(remain * commParas.e_size_align, INT32_BLOCK_NUM);
   tiling_params->cache_num_block = UssCeilDiv(tiling_params->max_cache_n_num * e_size, INT32_BLOCK_NUM);
   OP_LOGD(context->GetNodeName(), "e_size_align is %d, e_num_sub is %d, max_cache_n_num is %d",
           commParas.e_size_align, tiling_params->e_num_sub, tiling_params->max_cache_n_num);

   // ids params compute is common
   tiling_params->ids_size_input_scalar = ids_size;
   OP_CHECK_IF(
       ids_ele_byte == 0,
       OP_LOGE(context->GetNodeName(), "CalcTiling4Float check failed ids_ele_byte is 0."),
       return ge::GRAPH_FAILED);

   // calc need core num
   OP_CHECK_IF(CalcNeededCoreNum(context, commParas, ids_size, compile_info->core_num,
                                     tiling_params->need_core_num_input_scalar),
                   OP_LOGE(context->GetNodeName(), "CalcNeededCoreNum failed."),
                   return ge::GRAPH_FAILED);
   ComputeEleNumOneCore(ids_size, e_size, num_segments, tiling_params);

   // ids params front core
   uint32_t mode = FRONT_CORE;
   ComputeIdsParamsMovGm2ub(ids_ub_size, ids_ele_byte, mode, tiling_params);

   // ids params last core
   mode = LAST_CORE;
   ComputeIdsParamsMovGm2ub(ids_ub_size, ids_ele_byte, mode, tiling_params);

   if (tiling_params->select_key_input_scalar == SELECT_KEY_MODE_FP32_INPUT_LAST_AXIS_ALIGN_SMALL_E ||
       tiling_params->select_key_input_scalar == SELECT_KEY_MODE_FP32_INPUT_LAST_AXIS_ALIGN_SMALL_E_HP) {
     CalcTilingFloat4LastAxisSmallE(tiling_params, commParas, x_ub_size, input_ele_byte, e_size);
   } else if (tiling_params->select_key_input_scalar == SELECT_KEY_MODE_FP32_INPUT_LAST_AXIS_ONE) {
     CalcTilingFloat4LastAxisOne(tiling_params, output_ub_ele_num_one_row);
   } else if (tiling_params->select_key_input_scalar == SELECT_KEY_MODE_FP32_INPUT_LAST_AXIS_NOT_ALIGN_SMALL_E ||
              tiling_params->select_key_input_scalar == SELECT_KEY_MODE_FP32_INPUT_LAST_AXIS_NOT_ALIGN_SMALL_E_HP ||
              tiling_params->select_key_input_scalar == SELECT_KEY_MODE_FP32_INPUT_LAST_AXIS_NOT_ALIGN_SMALL_E_HP_PAD) {
     CalcTilingFloat4LastAxisNotAlignSmallE(tiling_params, x_ub_size, input_ele_byte, commParas,
                                            output_ub_ele_num_one_row);
   } else if (tiling_params->select_key_input_scalar == SELECT_KEY_MODE_FP32_INPUT_LAST_AXIS_ALIGN_BIG_E) {
     CalcTilingFloat4LastAxisAlignBigE(x_ub_size, input_ele_byte, e_size, ub_tensor_ele_num, tiling_params);
   } else if (tiling_params->select_key_input_scalar == SELECT_KEY_MODE_FP32_INPUT_LAST_AXIS_NOT_ALIGN_BIG_E) {
     CalcTilingFloat4LastAxisNotAlignBigE(tiling_params, x_ub_size, input_ele_byte, e_size, ub_tensor_ele_num);
   } else if (tiling_params->select_key_input_scalar == SELECT_KEY_MODE_FP32_INPUT_LAST_AXIS_ONE_MODIFY) {
     CalcTilingFloat4LastAxisModify(tiling_params);
   } else if (tiling_params->select_key_input_scalar == SELECT_KEY_MODE_FP32_INPUT_LAST_AXIS_ONE_MULTI) {
     CalcTilingFloat4LastAxisOneMulti(tiling_params);
   } else if (tiling_params->select_key_input_scalar == SELECT_KEY_MODE_FP32_INPUT_NUM_SEGMENT_ONE) {
     CalcTilingFloat4NumSegmentOne(tiling_params, input_ele_byte, x_ub_size, e_size, ub_tensor_ele_num);
   }

   // cout tiling params
   PrintTilingParams(context, tiling_params);
   // BlockDim, core num used in tik op
   context->SetBlockDim(tiling_params->need_core_num_input_scalar);

   return ge::GRAPH_SUCCESS;
 }

 ge::graphStatus CalcTiling4Int(gert::TilingContext* context, const ge::DataType input_dtype,
                                const int32_t output_ub_ele_num_one_row,
                                const int32_t num_segments, const int32_t e_size, const int32_t input_size,
                                const int32_t ids_size, const EleByte input_ele_byte, const EleByte ids_ele_byte) {
   auto compile_info = static_cast<const TilingPrepareForUnsortedSegmentSumCompileInfo*>(context->GetCompileInfo());
   OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);

   // int32 tiling params
   auto tiling_params = context->GetTilingData<UnsortedSegmentSumTilingParamsInt32>();
   OP_CHECK_NULL_WITH_CONTEXT(context, tiling_params);
   InitTilingParams(tiling_params, num_segments);
   tiling_params->tiling_core_num = compile_info->core_num;
   OP_LOGD(context->GetNodeName(), "CalcTiling4Int set tiling core num: %d", tiling_params->tiling_core_num);

   // commmn params
   // select key
   int32_t e_once_num = 0;
   int32_t id_once_num = 0;
   int32_t mask = 0;
   mask = (input_dtype == ge::DT_INT32) ? MASK_INT32 : MASK_FP16;

   CommParas4UbSizeNoAtomic comm_ub_size_params;
   comm_ub_size_params.ub_size = compile_info->ub_size;
   comm_ub_size_params.input_dtype = input_dtype;
   comm_ub_size_params.e_size = e_size;
   comm_ub_size_params.output_ub_ele_num_one_row = output_ub_ele_num_one_row;
   comm_ub_size_params.mask = mask;
   comm_ub_size_params.num_segments = num_segments;

   OP_CHECK_IF(CalcNeededCoreByNumSegments(context, comm_ub_size_params, compile_info->core_num,
                                               tiling_params->need_core_num_input_scalar),
                   OP_LOGE(context->GetNodeName(), "CalcNeededCoreByNumSegments failed."),
                   return ge::GRAPH_FAILED);
   comm_ub_size_params.need_core_num = tiling_params->need_core_num_input_scalar;

   int32_t ub_tensor_size = 0;
   int32_t ub_tensor_size_input = 0;
   int32_t ub_tensor_size_output = 0;

   ComputeUbTensorSizeNoAtomic(comm_ub_size_params, ub_tensor_size, ub_tensor_size_input, ub_tensor_size_output);
   OP_LOGD(context->GetNodeName(), "ub_tensor_size_id is=%d, ub_tensor_size_input is %d, ub_tensor_size_output is %d",
           ub_tensor_size, ub_tensor_size_input, ub_tensor_size_output);

   CommParas4TilingModeNoAtomic comm_tiling_mode_params;
   comm_tiling_mode_params.e_size = e_size;
   comm_tiling_mode_params.ids_size = ids_size;
   comm_tiling_mode_params.input_dtype = input_dtype;
   comm_tiling_mode_params.ub_tensor_size = ub_tensor_size;
   comm_tiling_mode_params.ub_tensor_size_input = ub_tensor_size_input;
   comm_tiling_mode_params.need_core = tiling_params->need_core_num_input_scalar;
   comm_tiling_mode_params.output_ub_ele_num_one_row = output_ub_ele_num_one_row;
   comm_tiling_mode_params.all_size = input_size;
   comm_tiling_mode_params.num_segments = num_segments;
   OP_CHECK_IF(!GetTilingModeNoAtomic(comm_tiling_mode_params, tiling_params->select_key_input_scalar, e_once_num,
                                          id_once_num, tiling_params->num_segment_max),
                   OP_LOGE(context->GetNodeName(), "GetTilingModeNoAtomic failed."),
                   return ge::GRAPH_FAILED);
   ComputeNumSegmentsParams(num_segments, e_size, output_ub_ele_num_one_row, tiling_params);

   // ids params
   tiling_params->ids_size_input_scalar = ids_size;
   ComputeIdsParamsMovGm2ubNoAtomic(ids_size, id_once_num, ids_ele_byte, tiling_params);

   // e num params
   tiling_params->e_num_input_scalar = e_size;

   CommParas4Int32 comm_paras;
   comm_paras.input_dytpe = input_dtype;
   comm_paras.ele_byte = input_ele_byte;
   comm_paras.e_once_num = e_once_num;
   comm_paras.all_size = input_size;
   comm_paras.num_segments = num_segments;
   ComputeENumParams(comm_paras, tiling_params);

   // cout tiling params
   PrintTilingParams(context, tiling_params);
   // BlockDim, core num used in tik op
   context->SetBlockDim(tiling_params->need_core_num_input_scalar);

   return ge::GRAPH_SUCCESS;
 }

 ge::graphStatus Tiling4SegmentSumComm(gert::TilingContext* context, const int32_t num_segments) {
   OP_LOGI(context->GetNodeName(), "Tiling4SegmentSumComm running.");

   auto input_data_shape = context->GetInputShape(INPUT_DATA_IDX);
   OP_CHECK_IF(input_data_shape == nullptr,
                   OP_LOGE(context->GetNodeName(), "get input_data_shape failed."),
                   return ge::GRAPH_FAILED);
   const gert::Shape& input_data_shape_sizes = Ops::Base::EnsureNotScalar(input_data_shape->GetStorageShape());

   auto input_segment_ids_shape = context->GetInputShape(INPUT_SEGMENT_IDS_IDX);
   OP_CHECK_IF(input_segment_ids_shape == nullptr,
                   OP_LOGE(context->GetNodeName(), "get input_segment_ids_shape failed."),
                   return ge::GRAPH_FAILED);
   const gert::Shape& input_segment_ids_shape_sizes = Ops::Base::EnsureNotScalar(input_segment_ids_shape->GetStorageShape());

   const int32_t input_size = input_data_shape_sizes.GetShapeSize();
   const int32_t ids_size = input_segment_ids_shape_sizes.GetShapeSize();
   OP_LOGI(context->GetNodeName(), "input_size=%d, ids_size=%d", input_size, ids_size);
   OP_CHECK_IF(input_size < ids_size,
                   OP_LOGE(context->GetNodeName(),
                                                   "dim of input must be greater than or equal with dim of ids"),
                   return ge::GRAPH_FAILED);

   for (size_t i = 0; i < input_segment_ids_shape_sizes.GetDimNum(); i++) {
     OP_LOGD(context->GetNodeName(), "input_segment_ids_shape_sizes[%zu] is %ld", i,
             input_segment_ids_shape_sizes.GetDim(i));
     OP_CHECK_IF(input_data_shape_sizes.GetDim(i) != input_segment_ids_shape_sizes.GetDim(i),
                     OP_LOGE(
                         context->GetNodeName(),
                         "front shape of input must be equal with ids shape, but input_data_shape_sizes[%ld] is %ld.", i,
                         input_data_shape_sizes.GetDim(i)),
                     return ge::GRAPH_FAILED);
   }

   if (input_size == 0 || ids_size == 0) {
     return CalcSpecialTiling(context, num_segments);
   }

   int32_t e_size = input_size / ids_size;
   OP_LOGD(context->GetNodeName(), " e_size is %d", e_size);

   auto input_data_dec_ptr = context->GetInputDesc(INPUT_DATA_IDX);
   OP_CHECK_IF(input_data_dec_ptr == nullptr,
                   OP_LOGE(context->GetNodeName(), "get input_data_dec_ptr failed."),
                   return ge::GRAPH_FAILED);
   const ge::DataType input_dtype = input_data_dec_ptr->GetDataType();

   auto segment_ids_dec_ptr = context->GetInputDesc(INPUT_SEGMENT_IDS_IDX);
   OP_CHECK_IF(segment_ids_dec_ptr == nullptr,
                   OP_LOGE(context->GetNodeName(), "get segment_ids_dec_ptr failed."),
                   return ge::GRAPH_FAILED);
   const ge::DataType ids_dtype = segment_ids_dec_ptr->GetDataType();

   // get input dtype
   EleByte input_ele_byte = FP32_BYTE;
   OP_CHECK_IF(!GetEleDtype(input_dtype, input_ele_byte),
                   OP_LOGE(context->GetNodeName(), "get input_ele_byte failed."),
                   return ge::GRAPH_FAILED);

   EleByte output_ele_byte = input_ele_byte;
   int32_t output_ub_ele_num_one_row = BYTE_BLOCK / output_ele_byte;
   OP_LOGD(context->GetNodeName(), " output_ub_ele_num_one_row is %d", output_ub_ele_num_one_row);

   // get ids dtype
   EleByte ids_ele_byte = FP32_BYTE;
   OP_CHECK_IF(!GetEleDtype(ids_dtype, ids_ele_byte),
                   OP_LOGE(context->GetNodeName(), "get ids_ele_byte failed."),
                   return ge::GRAPH_FAILED);

   if (input_dtype == ge::DT_FLOAT) {
     if (CalcTiling4Float(context, e_size, num_segments, input_size, ids_size, output_ub_ele_num_one_row, input_ele_byte,
                          ids_ele_byte) != ge::GRAPH_SUCCESS) {
       OP_LOGE(context->GetNodeName(), "exec CalcTiling4Float failed.");

       return ge::GRAPH_FAILED;
     }
   } else if (input_dtype == ge::DT_INT32 || input_dtype == ge::DT_FLOAT16) {
     if (CalcTiling4Int(context, input_dtype, output_ub_ele_num_one_row, num_segments, e_size, input_size,
                        ids_size, input_ele_byte, ids_ele_byte) != ge::GRAPH_SUCCESS) {
       OP_LOGE(context->GetNodeName(), "exec CCalcTilingForInt failed.");

       return ge::GRAPH_FAILED;
     }
   }

   OP_LOGD(context->GetNodeName(), "set block dim: %u", context->GetBlockDim());
   OP_LOGI(context->GetNodeName(), "op tiling run success.");

   return ge::GRAPH_SUCCESS;
 }

 ge::graphStatus SegmentTIKTiling(gert::TilingContext* context) {
   int32_t num_segments;
   OP_CHECK_IF(!Ops::Base::GetConstInt(context, INPUT_NUM_SEGMENTS_IDX, num_segments),
                   OP_LOGE(context->GetNodeName(), "num_segments not exists."),
                   return ge::GRAPH_FAILED);
   OP_CHECK_IF(num_segments <= 0,
                   OP_LOGE(context->GetNodeName(), "num_segments is small than 0."),
                   return ge::GRAPH_FAILED);
   OP_LOGD(context->GetNodeName(), "num_segments=%d", num_segments);

   return Tiling4SegmentSumComm(context, num_segments);
 }

 // tiling function
 ge::graphStatus Tiling4UnsortedSegmentSum(gert::TilingContext* context) {
   OP_LOGI(context->GetNodeName(), "Tiling4UnsortedSegmentSum running.");
   auto compile_info = static_cast<const TilingPrepareForUnsortedSegmentSumCompileInfo*>(context->GetCompileInfo());
   OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);
   OP_CHECK_IF(Tiling4UnsortedSegmentSumForAscendC(context) != ge::GRAPH_SUCCESS,
                  OP_LOGE(context->GetNodeName(), "call SimtTiling failed"),
                  return ge::GRAPH_FAILED);
   
   OP_LOGD(context->GetNodeName(), "Tiling4UnsortedSegmentSum running end");
   return ge::GRAPH_SUCCESS;
 }

 ge::graphStatus TilingPrepare4SegmentSumComm(gert::TilingParseContext* context) {
   OP_LOGD(context->GetNodeName(), "begin to do TilingPrepare4SegmentSumComm.");
   auto compile_info = GetCompileInfoPtr<TilingPrepareForUnsortedSegmentSumCompileInfo>(context);
   OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);

   std::unique_ptr<nlohmann::json> parsed_object_cinfo = GetCompileInfoJson(context);
   OP_CHECK_NULL_WITH_CONTEXT(context, parsed_object_cinfo);
   
   OP_LOGD(context->GetNodeName(), "Ascend C Tiling starting GRAPH_SUCCESS");
   auto platformInfo = context->GetPlatformInfo();
   OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
   auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
   compile_info->core_num = ascendcPlatform.GetCoreNumAiv();
   compile_info->max_thread = Ops::Base::GetSimtMaxThreadNum<gert::TilingParseContext>(context);
   OP_CHECK_IF((compile_info->core_num <= 0),
                    OP_LOGE(context->GetNodeName(),
                    "The core num is invaild."),
                    return ge::GRAPH_FAILED);
   OP_CHECK_IF((compile_info->max_thread <= 0),
                    OP_LOGE(context->GetNodeName(),
                    "The max thread from platform is invaild."),
                    return ge::GRAPH_FAILED);
   return ge::GRAPH_SUCCESS;
 }

 IMPL_OP_OPTILING(UnsortedSegmentSum)
     .Tiling(Tiling4UnsortedSegmentSum)
     .TilingParse<TilingPrepareForUnsortedSegmentSumCompileInfo>(TilingPrepare4SegmentSumComm);
 }  // namespace optiling
