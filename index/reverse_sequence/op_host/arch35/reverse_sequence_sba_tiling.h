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
 * \file reverse_sequence_sba_tiling.h
 * \brief
 */
#pragma once
#include "op_common/log/log.h"
#include "reverse_sequence_sba_common_tiling.h"
#include "reverse_sequence_tiling_common.h"

namespace optiling
{

class ReverseSequenceSBATiling : public ReverseSequenceSBACommonTiling
{
public:
    explicit ReverseSequenceSBATiling(gert::TilingContext* context) : ReverseSequenceSBACommonTiling(context)
    {
    }

    ~ReverseSequenceSBATiling() override
    {
    }

protected:
    void InitializationVars() override;
};

} // namespace optiling