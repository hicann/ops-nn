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
 * \file gather_elements_tiling_arch35.h
 * \brief
 */
#ifndef GATHER_ELEMENTS_SIMT_TILING_ARCH35_H
#define GATHER_ELEMENTS_SIMT_TILING_ARCH35_H

#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/runtime/kernel_run_context.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "gather_elements_tiling.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"

namespace optiling {

constexpr int64_t TILING_ARRAY_LEN_EIGHT = 8;
constexpr int64_t M_SHIFT_OFFSET = 7;
BEGIN_TILING_DATA_DEF(GatherElementsTilingData)
TILING_DATA_FIELD_DEF(int64_t, axis);                                        // 属性axis(dim)的值
TILING_DATA_FIELD_DEF(int64_t, usedCore);                                    // 使用核数
TILING_DATA_FIELD_DEF(int64_t, perCoreNum);                                  // simt:非尾核的元素个数 simd:非尾核需要处理的高轴个数
TILING_DATA_FIELD_DEF(int64_t, tailCoreNum);                                 // simt:尾核的元素个数 simd:尾核需要处理的高轴个数
TILING_DATA_FIELD_DEF(int64_t, xLoadInUbNum);
TILING_DATA_FIELD_DEF(int64_t, indexLoadInUbNum);
TILING_DATA_FIELD_DEF(int64_t, xUbFactor);
TILING_DATA_FIELD_DEF(int64_t, indexUbFactor);
TILING_DATA_FIELD_DEF(int64_t, xAfterAxis);
TILING_DATA_FIELD_DEF(int64_t, idxAfterAxis);
TILING_DATA_FIELD_DEF_ARR(int64_t, TILING_ARRAY_LEN_EIGHT, xStrideArr);      // 输入x的stride
TILING_DATA_FIELD_DEF_ARR(int64_t, TILING_ARRAY_LEN_EIGHT, indexStrideArr);  // 输入index的stride
TILING_DATA_FIELD_DEF_ARR(uint64_t, M_SHIFT_OFFSET, magic);
TILING_DATA_FIELD_DEF_ARR(uint64_t, M_SHIFT_OFFSET, shift);
TILING_DATA_FIELD_DEF_ARR(int64_t, TILING_ARRAY_LEN_EIGHT, indexShape);
TILING_DATA_FIELD_DEF_ARR(int64_t, TILING_ARRAY_LEN_EIGHT, xShape);

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GatherElements, GatherElementsTilingData)

class GatherElementsSimtTiling : public Ops::NN::Optiling::TilingBaseClass {
public:
    explicit GatherElementsSimtTiling(gert::TilingContext *context) : Ops::NN::Optiling::TilingBaseClass(context)
    {
        Reset();
    }
    ~GatherElementsSimtTiling() override = default;

    void Reset(gert::TilingContext *context) override
    {
        Ops::NN::Optiling::TilingBaseClass::Reset(context);
        Reset();
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
    // 8、reset重置
    void Reset();

private:
    static constexpr uint64_t DEFAULT_WORKSPACE_SIZE = static_cast<uint64_t>(16) * 1024 * 1024;
    static constexpr int64_t SMALL_CASE_THREAD_NUM = 512;
    static constexpr int64_t MAX_DIM_LEN_EIGHT = 8;
    static constexpr int64_t NUM_TWO = 2;
    static constexpr int64_t DIGIT_THOUSAND = 1000;
    static constexpr int64_t DIGIT_HUNDRED = 100;
    static constexpr int64_t DIGIT_TEN = 10;
    static constexpr int64_t INT32_MAX_BOUND = 2147483647;

    inline bool ParamTypeIsInvalid(ge::DataType &x);
    ge::graphStatus GetInAndOutInfo();
    inline ge::graphStatus GetAttrInfo();
    inline int HandleCount(int &count, int i, int j);
    void UpdateTilingKey();
    void ReductionDim();
    void CalculateFullLoadCondition(int64_t xDtypeSize, int64_t indexDtypeSize);
    void ComputeCoreNum();
    void ComputeStride();
    void SetTilingData();
    void MergeAxis();
    void CalcuCore();
    void PrintTilingData();
    void GetMagicAndShift();

private:
    int64_t axis_ = 0;
    int64_t dimSize_ = 0;
    int64_t xSize_ = 1;
    int64_t indexSize_ = 1;
    int64_t ySize_ = 1;

    int64_t threadNum_ = 1024;
    int64_t coreNum_ = 0;
    int64_t usedCoreNum_ = 0;
    int64_t ubSize_ = 0;
    int64_t enableInt64_ = 0;
    int64_t xLoadInUbNum_ = 0;
    int64_t indexLoadInUbNum_ = 0;
    int64_t xUbFactor_ = 0;
    int64_t indexUbFactor_ = 0;
    int64_t xAfterAxis_ = 0;
    int64_t idxAfterAxis_ = 0;
    int64_t isFullLoad_ = 0;

    std::vector<int64_t> xShapeArr_;
    std::vector<int64_t> indexShapeArr_;
    std::vector<int64_t> xStrideArr_;
    std::vector<int64_t> indexStrideArr_;
    uint64_t magic_[M_SHIFT_OFFSET] = {0,0,0,0,0,0,0};
    uint64_t shift_[M_SHIFT_OFFSET] = {0,0,0,0,0,0,0};
    
    gert::Shape xShape_;
    gert::Shape xShapeMerge_;
    gert::Shape xShape8d_;
    gert::Shape indexShape_;
    gert::Shape indexShapeMerge_;
    gert::Shape indexShape8d_;
    gert::Shape xStride_;
    gert::Shape indexStride_;
    gert::Shape yShape_;
    ge::DataType xDtype_;
    ge::DataType indexDtype_;
 
    const char *opName_ = "";
    GatherElementsTilingData m_tilingData_;
};
} // namespace optiling
#endif // GATHER_ELEMENTS_SIMT_TILING_ARCH35_H