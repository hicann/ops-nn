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
 * \file dynamic_rnn_tiling.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_DYNAMIC_RNN_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_DYNAMIC_RNN_H_
#include "dynamic_rnn/op_host/dynamic_rnn_common.h"
#include "log/log.h"                           // 如果涉及LOG日志打印
#include "register/op_impl_registry.h"         // 必需
#include "register/tilingdata_base.h"          // 必需
#include "op_host/tiling_base.h"                // 如果涉及TilingBaseClass类继承
#include "op_host/tiling_templates_registry.h" // 如果涉及TilingBaseClass类继承
#include "util/math_util.h"                   // 如果涉及CeilDiv等对齐运算
#include "error_util.h"

namespace optiling {
struct DynamicRnnTilingData {
  int32_t sequenceLength;
  int32_t dynamicRnnBatch;
  int32_t chequeIndex;
};

struct DynamicRnnCompileInfo : public optiling::CompileInfoBase {
  std::vector<std::vector<int64_t>> tuneShapeList;
};

class DynamicRNNTiling : public LstmBaseTiling {
  protected:
    bool CheckParamsShape(gert::TilingContext* context) override;
    bool CheckInitParamsShape(gert::TilingContext* context) override;
    bool CheckAttrOps(gert::TilingContext* context) override;
    bool CheckAttrTiling(gert::TilingContext* context) override;

    ge::graphStatus GetOpInfo(const gert::TilingContext* context, DynamicRnnTiling& dynamicRnnParams) override;
    ge::graphStatus GetAttr(const gert::TilingContext* context, DynamicRnnTiling& dynamicRnnParams) override;
};

}  // namespace optiling
#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_DYNAMIC_RNN_H_
