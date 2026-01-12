/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file softmax_cross_entropy_with_logits_apt.cpp
 * \brief softmax_cross_entropy_with_logits_apt
 */
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "arch35/softmax_cross_entropy_with_logits_full_load.h"
#include "arch35/softmax_cross_entropy_with_logits_split_r.h"
#include "arch35/softmax_cross_entropy_with_logits_tiling_data.h"
#include "arch35/softmax_cross_entropy_with_logits_tiling_key.h"

using namespace AscendC;

template <uint64_t schId, uint64_t featuresBrc, uint64_t labelsBrc, uint64_t db>
__global__ __aicore__ void softmax_cross_entropy_with_logits(GM_ADDR features, GM_ADDR labels, GM_ADDR loss, GM_ADDR backProp,  GM_ADDR workspace, GM_ADDR tiling) {
	REGISTER_TILING_DEFAULT(SoftmaxCrossEntropyWithLogitsTilingData);

	GET_TILING_DATA_WITH_STRUCT(SoftmaxCrossEntropyWithLogitsTilingData, tilingData, tiling);

	GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
	TPipe pipe;

    if constexpr (schId == 1) {
		SoftmaxCrossEntropyWithLogits::SoftmaxCrossEntropyWithLogitsSplitR<DTYPE_FEATURES, schId, featuresBrc, labelsBrc, db> op;
		op.Init(features, labels, loss, backProp, userWorkspace, &tilingData, &pipe);
		op.Process();
	}
    if constexpr (schId == 0 || schId == 2) {
	    SoftmaxCrossEntropyWithLogits::SoftmaxCrossEntropyWithLogitsFullLoad<DTYPE_FEATURES, schId, featuresBrc, labelsBrc, db> op;
	    op.Init(features, labels, loss, backProp, userWorkspace, &tilingData, &pipe);
	    op.Process();
    }
}
