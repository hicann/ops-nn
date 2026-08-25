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
 * \file reverse_sequence_a1sba_tiling.h
 * \brief
 */

#pragma once
#include "op_common/log/log.h"
#include "reverse_sequence_sba_common_tiling.h"
#include "reverse_sequence_tiling_common.h"

namespace optiling {

class ReverseSequenceA1SBATiling : public ReverseSequenceSBACommonTiling {
public:
    explicit ReverseSequenceA1SBATiling(gert::TilingContext* context) : ReverseSequenceSBACommonTiling(context) {}

    ~ReverseSequenceA1SBATiling() override {}

protected:
    void InitializationVars() override;
};

} // namespace optiling
