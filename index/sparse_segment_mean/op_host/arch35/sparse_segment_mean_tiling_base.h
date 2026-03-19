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
 * \file sparse_segment_mean_tiling_base.h
 * \brief
 */

 #ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_SPARSE_SEGMENT_MEAN_TILING_BASE_H_
 #define AIR_CXX_RUNTIME_V2_OP_IMPL_SPARSE_SEGMENT_MEAN_TILING_BASE_H_
 
 #include <cstdint>
 #include "util/math_util.h"
 #include "register/tilingdata_base.h"
 #include "tiling/tiling_api.h"
 #include "op_host/tiling_base.h"
 #include "op_host/tiling_templates_registry.h"
 #include "error_util.h"
 #include "op_common/op_host/util/platform_util.h"
 #include "register/op_def_registry.h"
 #include "index/sparse_segment_mean/op_kernel/arch35/sparse_segment_mean_struct.h"
 
 using namespace std;
 
 namespace optiling {
 
 struct SparseSegmentMeanInputInfo {
     int64_t innerSize{1};
     int64_t gatherSize{0};
     int64_t segmentNum{0};
     int64_t outterSize{0};
     ge::DataType inputDtype{ge::DataType::DT_FLOAT};
     int64_t inputBytes{1};
     int64_t indicesBytes{1};
     int64_t segmentIdsBytes{1};
     ge::DataType indicesDtype{ge::DataType::DT_INT32};
     ge::DataType segmentIdsDtype{ge::DataType::DT_INT32};
 };
 
 struct SparseSegmentMeanHardwareInfo {
     int64_t coreNum{0};
     int64_t ubSize{0};
 };
 
 struct SparseSegmentMeanCompileInfo {
     uint64_t coreNum{0};
     uint64_t ubSize{0};
 };
 
 class SparseSegmentMeanBaseTiling : public Ops::NN::Optiling::TilingBaseClass {
  public:
   explicit SparseSegmentMeanBaseTiling(gert::TilingContext* context) : TilingBaseClass(context) {
   }
 
   ~SparseSegmentMeanBaseTiling() override {
   }
 
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
    ge::graphStatus GetXAndYInfoAndCheck();
    ge::graphStatus GetIndicesAndSegmentIdsInfoAndCheck();
 
  public : 
   SparseSegmentMeanInputInfo inputData;
   SparseSegmentMeanHardwareInfo hardwareData;
 };
 }  // namespace optiling
 
 #endif