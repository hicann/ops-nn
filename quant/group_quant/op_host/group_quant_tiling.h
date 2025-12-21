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
 * \file group_quant_tiling.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_GROUP_QUANT_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_GROUP_QUANT_H
#include <cstdint>
#include <vector>
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "tiling_base/tiling_templates_registry.h"

namespace optiling {
using Ops::NN::Optiling::TilingBaseClass;
struct GroupQuantCompileInfo {
  int64_t coreNum = 0;
  uint64_t ubSizePlatForm = 0;
};

BEGIN_TILING_DATA_DEF(GroupQuantTilingData)
TILING_DATA_FIELD_DEF(int64_t, dimS);
TILING_DATA_FIELD_DEF(int64_t, dimE);
TILING_DATA_FIELD_DEF(int64_t, dimH);
TILING_DATA_FIELD_DEF(int64_t, hasOffset);
TILING_DATA_FIELD_DEF(int64_t, needCoreNum);
TILING_DATA_FIELD_DEF(int64_t, preCoreNum);
TILING_DATA_FIELD_DEF(int64_t, xRowNumPreCore);
TILING_DATA_FIELD_DEF(int64_t, xRowNumPostCore);

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GroupQuant, GroupQuantTilingData)

class GroupQuantTiling : public TilingBaseClass {
 public:
  explicit GroupQuantTiling(gert::TilingContext* context) : TilingBaseClass(context) {
  }

 protected:
  bool IsCapable() override;
  // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
  ge::graphStatus GetPlatformInfo() override;
  // 2、获取INPUT/OUTPUT/ATTR信息
  ge::graphStatus GetShapeAttrsInfo() override;
  // 3、计算数据切分TilingData
  ge::graphStatus DoOpTiling() override;
  // 4、计算高阶API的TilingData
  ge::graphStatus DoLibApiTiling() override;
  // 5、计算TilingKey
  uint64_t GetTilingKey() const override;
  // 6、计算Workspace 大小
  ge::graphStatus GetWorkspaceSize() override;
  // 7、保存Tiling数据
  ge::graphStatus PostTiling() override;

 private:
  int64_t coreNumVar{0};
  int64_t dimS{0};
  int64_t dimE{0};
  int64_t dimH{0};
  int64_t hasOffset{0};
  int64_t needCoreNum{0};
  int64_t preCoreNum{0};
  int64_t xRowNumPreCore{0};
  int64_t xRowNumPostCore{0};
  GroupQuantTilingData tilingData;
};

}  // namespace optiling
#endif  // OPS_BUILT_IN_OP_TILING_RUNTIME_GROUP_QUANT_H
