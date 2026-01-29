/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tiling_parse_context_faker.h"

namespace gert {
TilingParseContextFaker& TilingParseContextFaker::operator=(TilingParseContextFaker&& faker)
{
    KernelRunContextHolder::operator=(std::move(faker));
    return *this;
}

TilingParseContextFaker::TilingParseContextFaker(TilingParseContextFaker&& faker)
    : KernelRunContextHolder(std::move(faker))
{}

TilingParseContextFaker& TilingParseContextFaker::SetOpType(const std::string opType)
{
    opType_ = opType;
    OpTilingParseContextBuilder::OpType(opType.c_str()).OpName(opType.c_str());
    return *this;
}

TilingParseContextFaker& TilingParseContextFaker::KernelIONum(size_t inputNum, size_t outputNum)
{
    OpTilingParseContextBuilder::IONum(inputNum, outputNum);
    return *this;
}

TilingParseContextFaker& TilingParseContextFaker::Inputs(const std::vector<void*>& inputs)
{
    void* compileInfo = inputs.front();
    void* platformInfo = inputs.back();
    OpTilingParseContextBuilder::CompiledJson((const char*)compileInfo);
    OpTilingParseContextBuilder::PlatformInfo(platformInfo);
    return *this;
}

TilingParseContextFaker& TilingParseContextFaker::Outputs(const std::vector<void*>& outputs)
{
    void* compileInfo = outputs.front();
    OpTilingParseContextBuilder::CompiledInfo(compileInfo);
    return *this;
}

KernelRunContextHolder TilingParseContextFaker::Build()
{
    if (opType_.empty()) {
        SetOpType("fakeOp");
    }

    tilingParseContextHolder_ = std::move(OpTilingParseContextBuilder::Build());
    SetContext(tilingParseContextHolder_.GetContext());
    return std::move(*this);
}
} // namespace gert
