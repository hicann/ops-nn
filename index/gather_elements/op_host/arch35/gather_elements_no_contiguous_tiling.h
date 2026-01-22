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
 * \file gather_elements_no_contiguous_tiling.h
 * \brief
 */
#ifndef GATHER_ELEMENTS_NO_CONTIGUOUS_TILING_H
#define GATHER_ELEMENTS_NO_CONTIGUOUS_TILING_H

#include <iostream>
#include <string>
#include <vector>
#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/runtime/kernel_run_context.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling_base/tiling_base.h"
#include "tiling_base/tiling_templates_registry.h"

namespace optiling {

constexpr int64_t ARRAY_LEN_EIGHT = 8;
BEGIN_TILING_DATA_DEF(GatherElementsNoContiguousTilingData)
TILING_DATA_FIELD_DEF(int64_t, axis);                                        // 属性axis(dim)的值
TILING_DATA_FIELD_DEF(int64_t, usedCore);                                    // 使用核数
TILING_DATA_FIELD_DEF(int64_t, perCoreNum);                                  // simt:非尾核的元素个数 s
TILING_DATA_FIELD_DEF(int64_t, tailCoreNum);                                 // simt:尾核的元素个数 
TILING_DATA_FIELD_DEF_ARR(int64_t, ARRAY_LEN_EIGHT, indexShape);
TILING_DATA_FIELD_DEF_ARR(int64_t, ARRAY_LEN_EIGHT, xShape);
TILING_DATA_FIELD_DEF_ARR(int64_t, ARRAY_LEN_EIGHT, xStride);      // 输入x的stride
TILING_DATA_FIELD_DEF_ARR(int64_t, ARRAY_LEN_EIGHT, indexStride);  // 输入index的stride
TILING_DATA_FIELD_DEF_ARR(int64_t, ARRAY_LEN_EIGHT, yStride); 
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GatherElements_20100, GatherElementsNoContiguousTilingData)
REGISTER_TILING_DATA_CLASS(GatherElements_20200, GatherElementsNoContiguousTilingData)
REGISTER_TILING_DATA_CLASS(GatherElements_20201, GatherElementsNoContiguousTilingData)
REGISTER_TILING_DATA_CLASS(GatherElements_20300, GatherElementsNoContiguousTilingData)
REGISTER_TILING_DATA_CLASS(GatherElements_20301, GatherElementsNoContiguousTilingData)
REGISTER_TILING_DATA_CLASS(GatherElements_20302, GatherElementsNoContiguousTilingData)
REGISTER_TILING_DATA_CLASS(GatherElements_20400, GatherElementsNoContiguousTilingData)
REGISTER_TILING_DATA_CLASS(GatherElements_20401, GatherElementsNoContiguousTilingData)
REGISTER_TILING_DATA_CLASS(GatherElements_20402, GatherElementsNoContiguousTilingData)
REGISTER_TILING_DATA_CLASS(GatherElements_20403, GatherElementsNoContiguousTilingData)
REGISTER_TILING_DATA_CLASS(GatherElements_20500, GatherElementsNoContiguousTilingData)
REGISTER_TILING_DATA_CLASS(GatherElements_20501, GatherElementsNoContiguousTilingData)
REGISTER_TILING_DATA_CLASS(GatherElements_20502, GatherElementsNoContiguousTilingData)
REGISTER_TILING_DATA_CLASS(GatherElements_20503, GatherElementsNoContiguousTilingData)
REGISTER_TILING_DATA_CLASS(GatherElements_20504, GatherElementsNoContiguousTilingData)
REGISTER_TILING_DATA_CLASS(GatherElements_20600, GatherElementsNoContiguousTilingData)
REGISTER_TILING_DATA_CLASS(GatherElements_20601, GatherElementsNoContiguousTilingData)
REGISTER_TILING_DATA_CLASS(GatherElements_20602, GatherElementsNoContiguousTilingData)
REGISTER_TILING_DATA_CLASS(GatherElements_20603, GatherElementsNoContiguousTilingData)
REGISTER_TILING_DATA_CLASS(GatherElements_20604, GatherElementsNoContiguousTilingData)
REGISTER_TILING_DATA_CLASS(GatherElements_20605, GatherElementsNoContiguousTilingData)
REGISTER_TILING_DATA_CLASS(GatherElements_20700, GatherElementsNoContiguousTilingData)
REGISTER_TILING_DATA_CLASS(GatherElements_20701, GatherElementsNoContiguousTilingData)
REGISTER_TILING_DATA_CLASS(GatherElements_20702, GatherElementsNoContiguousTilingData)
REGISTER_TILING_DATA_CLASS(GatherElements_20703, GatherElementsNoContiguousTilingData)
REGISTER_TILING_DATA_CLASS(GatherElements_20704, GatherElementsNoContiguousTilingData)
REGISTER_TILING_DATA_CLASS(GatherElements_20705, GatherElementsNoContiguousTilingData)
REGISTER_TILING_DATA_CLASS(GatherElements_20706, GatherElementsNoContiguousTilingData)
REGISTER_TILING_DATA_CLASS(GatherElements_20800, GatherElementsNoContiguousTilingData)
REGISTER_TILING_DATA_CLASS(GatherElements_20801, GatherElementsNoContiguousTilingData)
REGISTER_TILING_DATA_CLASS(GatherElements_20802, GatherElementsNoContiguousTilingData)
REGISTER_TILING_DATA_CLASS(GatherElements_20803, GatherElementsNoContiguousTilingData)
REGISTER_TILING_DATA_CLASS(GatherElements_20804, GatherElementsNoContiguousTilingData)
REGISTER_TILING_DATA_CLASS(GatherElements_20805, GatherElementsNoContiguousTilingData)
REGISTER_TILING_DATA_CLASS(GatherElements_20806, GatherElementsNoContiguousTilingData)
REGISTER_TILING_DATA_CLASS(GatherElements_20807, GatherElementsNoContiguousTilingData)

REGISTER_TILING_DATA_CLASS(GatherElements_20110, GatherElementsNoContiguousTilingData)
REGISTER_TILING_DATA_CLASS(GatherElements_20210, GatherElementsNoContiguousTilingData)
REGISTER_TILING_DATA_CLASS(GatherElements_20211, GatherElementsNoContiguousTilingData)
REGISTER_TILING_DATA_CLASS(GatherElements_20310, GatherElementsNoContiguousTilingData)
REGISTER_TILING_DATA_CLASS(GatherElements_20311, GatherElementsNoContiguousTilingData)
REGISTER_TILING_DATA_CLASS(GatherElements_20312, GatherElementsNoContiguousTilingData)
REGISTER_TILING_DATA_CLASS(GatherElements_20410, GatherElementsNoContiguousTilingData)
REGISTER_TILING_DATA_CLASS(GatherElements_20411, GatherElementsNoContiguousTilingData)
REGISTER_TILING_DATA_CLASS(GatherElements_20412, GatherElementsNoContiguousTilingData)
REGISTER_TILING_DATA_CLASS(GatherElements_20413, GatherElementsNoContiguousTilingData)
REGISTER_TILING_DATA_CLASS(GatherElements_20510, GatherElementsNoContiguousTilingData)
REGISTER_TILING_DATA_CLASS(GatherElements_20511, GatherElementsNoContiguousTilingData)
REGISTER_TILING_DATA_CLASS(GatherElements_20512, GatherElementsNoContiguousTilingData)
REGISTER_TILING_DATA_CLASS(GatherElements_20513, GatherElementsNoContiguousTilingData)
REGISTER_TILING_DATA_CLASS(GatherElements_20514, GatherElementsNoContiguousTilingData)
REGISTER_TILING_DATA_CLASS(GatherElements_20610, GatherElementsNoContiguousTilingData)
REGISTER_TILING_DATA_CLASS(GatherElements_20611, GatherElementsNoContiguousTilingData)
REGISTER_TILING_DATA_CLASS(GatherElements_20612, GatherElementsNoContiguousTilingData)
REGISTER_TILING_DATA_CLASS(GatherElements_20613, GatherElementsNoContiguousTilingData)
REGISTER_TILING_DATA_CLASS(GatherElements_20614, GatherElementsNoContiguousTilingData)
REGISTER_TILING_DATA_CLASS(GatherElements_20615, GatherElementsNoContiguousTilingData)
REGISTER_TILING_DATA_CLASS(GatherElements_20710, GatherElementsNoContiguousTilingData)
REGISTER_TILING_DATA_CLASS(GatherElements_20711, GatherElementsNoContiguousTilingData)
REGISTER_TILING_DATA_CLASS(GatherElements_20712, GatherElementsNoContiguousTilingData)
REGISTER_TILING_DATA_CLASS(GatherElements_20713, GatherElementsNoContiguousTilingData)
REGISTER_TILING_DATA_CLASS(GatherElements_20714, GatherElementsNoContiguousTilingData)
REGISTER_TILING_DATA_CLASS(GatherElements_20715, GatherElementsNoContiguousTilingData)
REGISTER_TILING_DATA_CLASS(GatherElements_20716, GatherElementsNoContiguousTilingData)
REGISTER_TILING_DATA_CLASS(GatherElements_20810, GatherElementsNoContiguousTilingData)
REGISTER_TILING_DATA_CLASS(GatherElements_20811, GatherElementsNoContiguousTilingData)
REGISTER_TILING_DATA_CLASS(GatherElements_20812, GatherElementsNoContiguousTilingData)
REGISTER_TILING_DATA_CLASS(GatherElements_20813, GatherElementsNoContiguousTilingData)
REGISTER_TILING_DATA_CLASS(GatherElements_20814, GatherElementsNoContiguousTilingData)
REGISTER_TILING_DATA_CLASS(GatherElements_20815, GatherElementsNoContiguousTilingData)
REGISTER_TILING_DATA_CLASS(GatherElements_20816, GatherElementsNoContiguousTilingData)
REGISTER_TILING_DATA_CLASS(GatherElements_20817, GatherElementsNoContiguousTilingData)

REGISTER_TILING_DATA_CLASS(GatherElements_21302, GatherElementsNoContiguousTilingData)
REGISTER_TILING_DATA_CLASS(GatherElements_21312, GatherElementsNoContiguousTilingData)

class GatherElementsNoContiguousTiling : public Ops::NN::Optiling::TilingBaseClass {
public:
    explicit GatherElementsNoContiguousTiling(gert::TilingContext *context) : Ops::NN::Optiling::TilingBaseClass(context)
    {
    }

protected:
    bool IsCapable() override;
    // 顺序执行1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8
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
    static constexpr uint64_t DEFAULT_WORKSPACE_SIZE = 16 * 1024 * 1024;
    static constexpr int64_t SMALL_CASE_THREAD_NUM = 512;
    static constexpr int64_t MAX_DIM_LEN_EIGHT = 8;
    static constexpr int64_t NUM_TWO = 2;
    static constexpr int64_t DIGIT_THOUSAND = 1000;
    static constexpr int64_t DIGIT_HUNDRED = 100;
    static constexpr int64_t DIGIT_TEN = 10;

    inline bool ParamTypeIsInvalid(ge::DataType &x);
    ge::graphStatus GetInAndOutInfo();
    inline ge::graphStatus GetAttrInfo();
    inline int HandleCount(int &count, int i, int j);
    void CalcuCore();
    void PrintTilingData();
    bool IsContiguous(const gert::Shape &xShape, const gert::Stride &xStride);
    bool CheckIsIndexTranspose();
    ge::graphStatus GetContiguousTensorInfo(gert::Shape &shape, gert::Stride &stride, size_t idx, bool isOut);
    ge::graphStatus GetTensorInfo(gert::Shape &shape, gert::Stride &stride, size_t idx, bool isOut=false);
    void SetTilingData();
    bool IsEnableInt64(gert::Shape shape, gert::Stride stride);
    bool CanMerge(int64_t* xShape, int64_t* xStride, int index);
    void InitVector(vector<int64_t> &tempXShape, vector<int64_t> &tempXStride, vector<int64_t> &tempIndexShape, vector<int64_t> &tempIndexStride, vector<int64_t> &tempYStride);
    void DoCoalesce(vector<int64_t> &tempXShape, vector<int64_t> &tempXStride, vector<int64_t> &tempIndexShape, vector<int64_t> &tempIndexStride, vector<int64_t> &tempYStride);
    void UpdateResultFromVector(vector<int64_t> &tempXShape, vector<int64_t> &tempXStride, vector<int64_t> &tempIndexShape, vector<int64_t> &tempIndexStride, vector<int64_t> &tempYStride);
    void CoalesceGatherElements();
private:
    int64_t axis_ = 0;
    int64_t dimSize_ = 0;
    int64_t xSize_ = 1;
    int64_t ySize_ = 1;

    int64_t threadNum_ = 1024;
    int64_t coreNum_ = 0;
    int64_t usedCoreNum_ = 0;
    int64_t perCoreNum_ = 0;
    int64_t tailCoreNum_ = 0;
    int64_t ubSize_ = 0;
    int64_t enableInt64_ = 0;
    int64_t indexDtypeSize_ = 0;
    int64_t xDtypeSize_ = 0;
    bool isIndexTranspose_ = false;
   
    gert::Shape xShape_;
    gert::Shape indexShape_;
    gert::Shape yShape_;

    gert::Stride xStride_;
    gert::Stride indexStride_;
    gert::Stride yStride_;

    ge::DataType xDtype_;
    ge::DataType indexDtype_;
 
    const char *opName_ = "GatherElements";
    GatherElementsNoContiguousTilingData m_tilingData_;
};
} // namespace optiling
#endif // GATHER_ELEMENTS_NO_CONTIGUOUS_TILING_H