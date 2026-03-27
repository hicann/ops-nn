/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file reverse_sequence_sba_tiling.cpp
 * \brief
 */

#include "reverse_sequence_sba_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "platform/platform_info.h"
#include "error_util.h"
#include "../op_kernel/arch35/reverse_sequence_struct.h"

using namespace AscendC;
using namespace ge;

namespace optiling
{
using namespace ReverseSequence;

static constexpr int64_t DIM_A1 = 0;
static constexpr int64_t DIM_S = 1;
static constexpr int64_t DIM_B = 2;
static constexpr int64_t DIM_A = 3;
static constexpr int64_t SBA_RESERVED_SIZE = 131072; // 128k

void ReverseSequenceSBATiling::InitializationVars()
{
    OP_LOGD("ReverseSequenceSBATiling::InitializationVars begin");
    oneBlockNum_ = Ops::Base::GetUbBlockSize(context_) / inputData_.xDtypeSize;
    availableUb_ = ubSize_ / inputData_.xDtypeSize;
    sbaResvervedNum_ = SBA_RESERVED_SIZE / inputData_.xDtypeSize;
    inputData_.inputDim[DIM_A] = inputData_.inputDim[DIM_A - 1];
    inputData_.inputDim[DIM_B] = inputData_.inputDim[DIM_B - 1];
    inputData_.inputDim[DIM_S] = inputData_.inputDim[DIM_S - 1];
    inputData_.inputDim[DIM_A1] = 1;
}

REGISTER_TILING_TEMPLATE("ReverseSequence", ReverseSequenceSBATiling, 8);

}  // namespace optiling