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
 * \file max_pool_grad_with_argmax_v3_nchw_tiling_scalar.h
 * \brief
 */

#ifndef MAX_POOL_GRAD_WITH_AGRMAX_V3_NCHW_TILING_SCALAR_H_
#define MAX_POOL_GRAD_WITH_AGRMAX_V3_NCHW_TILING_SCALAR_H_

#include "max_pool_grad_with_argmax_v3_tiling_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MaxPoolGradWithArgmaxV3NCHWScalarTilingData)
TILING_DATA_FIELD_DEF(int64_t, hArgmax);
TILING_DATA_FIELD_DEF(int64_t, wArgmax);
TILING_DATA_FIELD_DEF(int64_t, hOutput);
TILING_DATA_FIELD_DEF(int64_t, wOutput);
TILING_DATA_FIELD_DEF(int64_t, hKernel);
TILING_DATA_FIELD_DEF(int64_t, wKernel);
TILING_DATA_FIELD_DEF(int64_t, hStride);
TILING_DATA_FIELD_DEF(int64_t, wStride);
TILING_DATA_FIELD_DEF(int64_t, padH);
TILING_DATA_FIELD_DEF(int64_t, padW);
TILING_DATA_FIELD_DEF(int64_t, dilationH);
TILING_DATA_FIELD_DEF(int64_t, dilationW);
TILING_DATA_FIELD_DEF(int64_t, highAxisInner);
TILING_DATA_FIELD_DEF(int64_t, highAxisTail);
TILING_DATA_FIELD_DEF(int64_t, highAxisOuter);
TILING_DATA_FIELD_DEF(int64_t, hOutputInner);
TILING_DATA_FIELD_DEF(int64_t, hOutputTail);
TILING_DATA_FIELD_DEF(int64_t, hOutputOuter);
TILING_DATA_FIELD_DEF(int64_t, wOutputInner);
TILING_DATA_FIELD_DEF(int64_t, wOutputTail);
TILING_DATA_FIELD_DEF(int64_t, wOutputOuter);
TILING_DATA_FIELD_DEF(int64_t, normalCoreProcessNum);
TILING_DATA_FIELD_DEF(int64_t, tailCoreProcessNum);
TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);
TILING_DATA_FIELD_DEF(int64_t, outputBufferSize);
TILING_DATA_FIELD_DEF(int64_t, gradBufferSize);
TILING_DATA_FIELD_DEF(int64_t, argmaxBufferSize);
TILING_DATA_FIELD_DEF(int64_t, argmaxNcInner);
TILING_DATA_FIELD_DEF(int64_t, argmaxNcOuter);
TILING_DATA_FIELD_DEF(int64_t, argmaxNcTail);
TILING_DATA_FIELD_DEF(int64_t, argmaxHInner);
TILING_DATA_FIELD_DEF(int64_t, argmaxHOuter);
TILING_DATA_FIELD_DEF(int64_t, argmaxHTail);
TILING_DATA_FIELD_DEF(int64_t, argmaxWInner);
TILING_DATA_FIELD_DEF(int64_t, argmaxWOuter);
TILING_DATA_FIELD_DEF(int64_t, argmaxWTail);
TILING_DATA_FIELD_DEF(int64_t, argmaxInnerLoop);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MaxPoolGradWithArgmaxV3_301, MaxPoolGradWithArgmaxV3NCHWScalarTilingData);

class MaxPoolGradWithArgmaxV3ScalarTilingInfo {
public:
    int64_t highAxisInner = 0;
    int64_t highAxisTail = 0;
    int64_t highAxisOuter = 0;
    int64_t hOutputInner = 0;
    int64_t hOutputTail = 0;
    int64_t hOutputOuter = 0;
    int64_t wOutputInner = 0;
    int64_t wOutputTail = 0;
    int64_t wOutputOuter = 0;
    int64_t normalCoreProcessNum = 0;
    int64_t tailCoreProcessNum = 0;
    int64_t usedCoreNum = 0;
    int64_t outputBufferSize = 0;
    int64_t gradBufferSize = 0;
    int64_t argmaxBufferSize = 0;
    int64_t argmaxNcInner = 0;
    int64_t argmaxNcOuter = 0;
    int64_t argmaxNcTail = 0;
    int64_t argmaxHInner = 0;
    int64_t argmaxHOuter = 0;
    int64_t argmaxHTail = 0;
    int64_t argmaxWInner = 0;
    int64_t argmaxWOuter = 0;
    int64_t argmaxWTail = 0;
    int64_t argmaxInnerLoop = 0;

public:
    std::string ToString() const
    {
        std::stringstream info;
        info << "MaxPoolGradWithArgmaxV3ScalarTilingInfo {";
        info << "highAxisInner:" << highAxisInner << ",highAxisTail:" << highAxisTail
             << ",highAxisOuter:" << highAxisOuter << ",hOutputInner:" << hOutputInner
             << ", hOutputTail:" << hOutputTail << ", hOutputOuter:" << hOutputOuter
             << ", wOutputInner:" << wOutputInner << ", wOutputTail:" << wOutputTail
             << ", wOutputOuter:" << wOutputOuter << ", normalCoreProcessNum:" << normalCoreProcessNum
             << ", tailCoreProcessNum:" << tailCoreProcessNum << ", usedCoreNum:" << usedCoreNum
             << ", outputBufferSize:" << outputBufferSize << ", gradBufferSize:" << gradBufferSize
             << ", argmaxBufferSize:" << argmaxBufferSize << ", argmaxNcInner:" << argmaxNcInner
             << ", argmaxNcOuter:" << argmaxNcOuter << ", argmaxNcTail:" << argmaxNcTail
             << ", argmaxHInner:" << argmaxHInner << ", argmaxHOuter:" << argmaxHOuter
             << ", argmaxHTail:" << argmaxHTail << ", argmaxWInner:" << argmaxWInner
             << ", argmaxWOuter:" << argmaxWOuter << ", argmaxWTail:" << argmaxWTail
             << ", argmaxInnerLoop:" << argmaxInnerLoop;
        info << " }";
        return info.str();
    }
};

class MaxPoolGradWithArgmaxV3NCHWScalarTiling : public MaxPoolGradWithArgmaxV3BaseTiling {
public:
    explicit MaxPoolGradWithArgmaxV3NCHWScalarTiling(gert::TilingContext* context)
        : MaxPoolGradWithArgmaxV3BaseTiling(context)
    {}
    ~MaxPoolGradWithArgmaxV3NCHWScalarTiling() override
    {}

protected:
    void SetTilingData();
    uint64_t GetTilingKey() const override;
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    void PrintData() const;
    void CalcBase();
    void CalcParamsEachCore();
    ge::graphStatus CalcGradArgmaxInner(int64_t argmaxCountInUB);
    ge::graphStatus CalcGradArgmax();
    MaxPoolGradWithArgmaxV3NCHWScalarTilingData tilingData_;
    MaxPoolGradWithArgmaxV3ScalarTilingInfo scalarTilingData_;
};

} // namespace optiling
#endif