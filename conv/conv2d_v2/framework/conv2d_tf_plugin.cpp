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
 * \file conv2d_tf_plugin.cpp
 * \brief
 */
#include "register/register.h"
#include "framework/plugin_util.h"
#include "graph/operator.h"
#include "error_util.h"

#include "log/log.h"

namespace domi {
const int kInputX = 0;
const int kInputFilter = 1;
const size_t kPaddingSize = 8;

/*!
 * @brief Replace GE ParseParams fuction to process graph conv2d node attrs
 * @param opSrc the source op info from tf.
 * @param op the dest GE op.
 * @return status whether this operation success.
 */
static Status ParseParamsConv2D(const ge::Operator& opSrc, ge::Operator& op)
{
    // Convert original tf graph conv2d attrs to GE graph attrs
    if (AutoMappingByOpFn(opSrc, op) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "auto mapping failed.");
        return FAILED;
    }

    // The filter format shuold be HWCN, not NHWC or NCHW, so set here to fix this problem
    ge::TensorDesc orgTensorW = op.GetInputDesc(kInputFilter);
    orgTensorW.SetOriginFormat(ge::FORMAT_HWCN);
    orgTensorW.SetFormat(ge::FORMAT_HWCN);
    auto ret = op.UpdateInputDesc(kInputFilter, orgTensorW);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "update filter format failed.");
        return FAILED;
    }

    int32_t padTop = 0;
    int32_t padBottom = 0;
    int32_t padLeft = 0;
    int32_t padRight = 0;
    // String type padding is processed during infershape
    std::vector<int32_t> paddingList;
    if (op.GetAttr("explicit_paddings", paddingList) == ge::GRAPH_SUCCESS && paddingList.size() == kPaddingSize) {
        ge::TensorDesc orgTensorX = op.GetInputDesc(kInputX);
        auto x_format = orgTensorX.GetOriginFormat();
        if (x_format == ge::FORMAT_NCHW) {
            padTop = paddingList[4];    // 4: pad of H (top)
            padBottom = paddingList[5]; // 5: pad of H (bottom)
            padLeft = paddingList[6];   // 6: pad of W (left)
            padRight = paddingList[7];  // 7: pad of W (right)
        } else if (x_format == ge::FORMAT_NHWC) {
            padTop = paddingList[2];    // 2: pad of H (top)
            padBottom = paddingList[3]; // 3: pad of H (bottom)
            padLeft = paddingList[4];   // 4: pad of W (left)
            padRight = paddingList[5];  // 5: pad of W (right)
        }
    }

    // Escape GE require attr [pads] check here
    std::vector<int32_t> padList = {padTop, padBottom, padLeft, padRight};
    op.SetAttr("pads", padList);
    OP_LOGD(GetOpName(op).c_str(), "set pads [%d,%d,%d,%d] here.", padList[0], padList[1], padList[2],
            padList[3]); // 0:pad_top, 1:pad_bottom, 2:pad_left, 3:pad_right

    return SUCCESS;
}

REGISTER_CUSTOM_OP("Conv2D")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Conv2D")
    .ParseParamsByOperatorFn(ParseParamsConv2D)
    .ImplyType(ImplyType::TVM);
} // namespace domi
