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
 * \file masked_softmax_with_rel_pos_bias.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "masked_softmax_with_rel_pos_bias_B.h"
#include "masked_softmax_with_rel_pos_bias_BW.h"
#include "masked_softmax_with_rel_pos_bias_S1_bias.h"
#include "masked_softmax_with_rel_pos_bias_BWN.h"
#include "masked_softmax_with_rel_pos_bias_BWNS1.h"
#include "masked_softmax_with_rel_pos_bias_ONES2.h"

using namespace MaskedSoftmaxWithRelPosBias;

extern "C" __global__ __aicore__ void masked_softmax_with_rel_pos_bias(GM_ADDR x, GM_ADDR atten_mask, GM_ADDR bias,
                                                                       GM_ADDR y, GM_ADDR work_space, GM_ADDR tiling) {
  GET_TILING_DATA(tiling_data, tiling);
  if (TILING_KEY_IS(5001)) {
    MaskedSoftmaxWithRelPosBiasB<float, true, true, true> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(5002)) {
    MaskedSoftmaxWithRelPosBiasB<float, true, false, true> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(5003)) {
    MaskedSoftmaxWithRelPosBiasB<float, false, true, true> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(5004)) {
    MaskedSoftmaxWithRelPosBiasB<float, false, false, true> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(5011)) {
    MaskedSoftmaxWithRelPosBiasBBf16AndHalf<half, true, true, true> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(5012)) {
    MaskedSoftmaxWithRelPosBiasBBf16AndHalf<half, true, false, true> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(5013)) {
    MaskedSoftmaxWithRelPosBiasBBf16AndHalf<half, false, true, true> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(5014)) {
    MaskedSoftmaxWithRelPosBiasBBf16AndHalf<half, false, false, true> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(5021)) {
#if (__CCE_AICORE__ > 200)
    MaskedSoftmaxWithRelPosBiasBBf16AndHalf<bfloat16_t, true, true, true> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(5022)) {
    MaskedSoftmaxWithRelPosBiasBBf16AndHalf<bfloat16_t, true, false, true> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(5023)) {
    MaskedSoftmaxWithRelPosBiasBBf16AndHalf<bfloat16_t, false, true, true> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(5024)) {
    MaskedSoftmaxWithRelPosBiasBBf16AndHalf<bfloat16_t, false, false, true> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
#endif
  } else if (TILING_KEY_IS(5101)) {
    MaskedSoftmaxWithRelPosBiasB<float, true, true, false> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(5102)) {
    MaskedSoftmaxWithRelPosBiasB<float, true, false, false> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(5103)) {
    MaskedSoftmaxWithRelPosBiasB<float, false, true, false> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(5104)) {
    MaskedSoftmaxWithRelPosBiasB<float, false, false, false> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(5111)) {
    MaskedSoftmaxWithRelPosBiasBBf16AndHalf<half, true, true, false> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(5112)) {
    MaskedSoftmaxWithRelPosBiasBBf16AndHalf<half, true, false, false> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(5113)) {
    MaskedSoftmaxWithRelPosBiasBBf16AndHalf<half, false, true, false> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(5114)) {
    MaskedSoftmaxWithRelPosBiasBBf16AndHalf<half, false, false, false> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(5121)) {
#if (__CCE_AICORE__ > 200)
    MaskedSoftmaxWithRelPosBiasBBf16AndHalf<bfloat16_t, true, true, false> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(5122)) {
    MaskedSoftmaxWithRelPosBiasBBf16AndHalf<bfloat16_t, true, false, false> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(5123)) {
    MaskedSoftmaxWithRelPosBiasBBf16AndHalf<bfloat16_t, false, true, false> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(5124)) {
    MaskedSoftmaxWithRelPosBiasBBf16AndHalf<bfloat16_t, false, false, false> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
#endif
  } else if (TILING_KEY_IS(401)) {
    MaskedSoftmaxWithRelPosBiasBW<float, true, true> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(402)) {
    MaskedSoftmaxWithRelPosBiasBW<float, true, false> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(403)) {
    MaskedSoftmaxWithRelPosBiasBW<float, false, true> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(404)) {
    MaskedSoftmaxWithRelPosBiasBW<float, false, false> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(411)) {
    MaskedSoftmaxWithRelPosBiasBW<half, true, true> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(412)) {
    MaskedSoftmaxWithRelPosBiasBW<half, true, false> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(413)) {
    MaskedSoftmaxWithRelPosBiasBW<half, false, true> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(414)) {
    MaskedSoftmaxWithRelPosBiasBW<half, false, false> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(421)) {
#if (__CCE_AICORE__ > 200)
    MaskedSoftmaxWithRelPosBiasBW<bfloat16_t, true, true> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(422)) {
    MaskedSoftmaxWithRelPosBiasBW<bfloat16_t, true, false> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(423)) {
    MaskedSoftmaxWithRelPosBiasBW<bfloat16_t, false, true> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(424)) {
    MaskedSoftmaxWithRelPosBiasBW<bfloat16_t, false, false> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
#endif
  } else if (TILING_KEY_IS(201)) {
    MaskedSoftmaxWithRelPosBiasBWNS1<float, true, true> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(202)) {
    MaskedSoftmaxWithRelPosBiasBWNS1<float, true, false> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(203)) {
    MaskedSoftmaxWithRelPosBiasBS1Bias<float, true, true> op(&tiling_data);
    op.Init(x, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(204)) {
    MaskedSoftmaxWithRelPosBiasBS1Bias<float, false, true> op(&tiling_data);
    op.Init(x, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(211)) {
    MaskedSoftmaxWithRelPosBiasBWNS1<half, true, true> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(212)) {
    MaskedSoftmaxWithRelPosBiasBWNS1<half, true, false> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(213)) {
    MaskedSoftmaxWithRelPosBiasBS1Bias<half, true, true> op(&tiling_data);
    op.Init(x, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(214)) {
    MaskedSoftmaxWithRelPosBiasBS1Bias<half, false, true> op(&tiling_data);
    op.Init(x, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(221)) {
#if (__CCE_AICORE__ > 200)
    MaskedSoftmaxWithRelPosBiasBWNS1<bfloat16_t, true, true> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(222)) {
    MaskedSoftmaxWithRelPosBiasBWNS1<bfloat16_t, true, false> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(223)) {
    MaskedSoftmaxWithRelPosBiasBS1Bias<bfloat16_t, true, true> op(&tiling_data);
    op.Init(x, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(224)) {
    MaskedSoftmaxWithRelPosBiasBS1Bias<bfloat16_t, false, true> op(&tiling_data);
    op.Init(x, bias, y);
    op.Process();
#endif
  } else if (TILING_KEY_IS(2103)) {
    MaskedSoftmaxWithRelPosBiasBS1Bias<float, true, false> op(&tiling_data);
    op.Init(x, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(2104)) {
    MaskedSoftmaxWithRelPosBiasBS1Bias<float, false, false> op(&tiling_data);
    op.Init(x, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(2113)) {
    MaskedSoftmaxWithRelPosBiasBS1Bias<half, true, false> op(&tiling_data);
    op.Init(x, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(2114)) {
    MaskedSoftmaxWithRelPosBiasBS1Bias<half, false, false> op(&tiling_data);
    op.Init(x, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(2123)) {
#if (__CCE_AICORE__ > 200)
    MaskedSoftmaxWithRelPosBiasBS1Bias<bfloat16_t, true, false> op(&tiling_data);
    op.Init(x, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(2124)) {
    MaskedSoftmaxWithRelPosBiasBS1Bias<bfloat16_t, false, false> op(&tiling_data);
    op.Init(x, bias, y);
    op.Process();
#endif
  } else if (TILING_KEY_IS(301)) {
    MaskedSoftmaxWithRelPosBiasBWN<float, true, true> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(302)) {
    MaskedSoftmaxWithRelPosBiasBWN<float, true, false> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(303)) {
    MaskedSoftmaxWithRelPosBiasBWN<float, false, true> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(304)) {
    MaskedSoftmaxWithRelPosBiasBWN<float, false, false> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(311)) {
    MaskedSoftmaxWithRelPosBiasBWN<half, true, true> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(312)) {
    MaskedSoftmaxWithRelPosBiasBWN<half, true, false> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(313)) {
    MaskedSoftmaxWithRelPosBiasBWN<half, false, true> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(314)) {
    MaskedSoftmaxWithRelPosBiasBWN<half, false, false> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(321)) {
#if (__CCE_AICORE__ > 200)
    MaskedSoftmaxWithRelPosBiasBWN<bfloat16_t, true, true> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(322)) {
    MaskedSoftmaxWithRelPosBiasBWN<bfloat16_t, true, false> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(323)) {
    MaskedSoftmaxWithRelPosBiasBWN<bfloat16_t, false, true> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(324)) {
    MaskedSoftmaxWithRelPosBiasBWN<bfloat16_t, false, false> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
#endif
  } else if (TILING_KEY_IS(100)) {
    MaskedSoftmaxWithRelPosBiasONES2<float> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(110)) {
    MaskedSoftmaxWithRelPosBiasONES2<half> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(120)) {
#if (__CCE_AICORE__ > 200)
    MaskedSoftmaxWithRelPosBiasONES2<bfloat16_t> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
#endif
  }
}
