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
 * \file sorted_sparse_segment_mean_grad_tiling_base.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_SORTED_SPARSE_SEGMENT_MEAN_GRAD_TILING_BASE_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_SORTED_SPARSE_SEGMENT_MEAN_GRAD_TILING_BASE_H_

#include <cstdint>
#include "util/math_util.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "error_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "register/op_def_registry.h"
#include "index/sorted_sparse_segment_mean_grad/op_kernel/arch35/sorted_sparse_segment_mean_grad_struct.h" 
using namespace std;

namespace optiling {

struct SortedSparseSegmentMeanGradInputInfo {
    int64_t innerSize{1};
    int64_t segmentNum{0};
    int64_t outterSize{0};
    int32_t outputDim0{0};
    ge::DataType inputDtype{ge::DataType::DT_FLOAT};
    ge::DataType indicesDtype{ge::DataType::DT_INT32};
    ge::DataType segmentIdsDtype{ge::DataType::DT_INT32};
    ge::DataType indicesLocationDtype{ge::DataType::DT_INT32};
};

struct SortedSparseSegmentMeanGradHardwareInfo {
    int64_t coreNum{0};
    int64_t ubSize{0};
};

struct SortedSparseSegmentMeanGradCompileInfo {
    uint64_t coreNum{0};
    uint64_t ubSize{0};
};

class SortedSparseSegmentMeanGradBaseTiling : public Ops::NN::Optiling::TilingBaseClass {
 public:
  explicit SortedSparseSegmentMeanGradBaseTiling(gert::TilingContext* context) : TilingBaseClass(context) {
  }

  ~SortedSparseSegmentMeanGradBaseTiling() override = default;

 protected:
  bool IsCapable() override;
  ge::graphStatus GetPlatformInfo() override;
  ge::graphStatus GetShapeAttrsInfo() override;
  ge::graphStatus DoOpTiling() override;
  ge::graphStatus DoLibApiTiling() override;
  uint64_t GetTilingKey() const override;
  ge::graphStatus GetWorkspaceSize() override;
  ge::graphStatus PostTiling() override;
  void PrintHardwareData() const;
  void PrintInputData() const;

 private:
    ge::graphStatus GetXInfoAndCheck();
    ge::graphStatus GetYInfoAndCheck();
    ge::graphStatus GetIndicesInfoAndCheck();
    ge::graphStatus GetSegmentIdsInfoAndCheck();
    ge::graphStatus GetOutputDim0InfoAndCheck();
    ge::graphStatus GetIndicesLocationInfoAndCheck();
 
 public : 
   SortedSparseSegmentMeanGradInputInfo inputData;
   SortedSparseSegmentMeanGradHardwareInfo hardwareData;
   gert::Shape xShape_;
   gert::Shape indicesShape_;
 };
 }  // namespace optiling
 
 #endif
 