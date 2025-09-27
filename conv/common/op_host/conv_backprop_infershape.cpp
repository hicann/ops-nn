/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file conv_backprop_infershape.cpp
 * \brief
 */
#include "conv_backprop_infershape.h"

namespace {
inline bool IsConstTensor(const gert::Tensor* input_tensor) {
  if (input_tensor != nullptr) {
    if (input_tensor->GetAddr() == nullptr) {
      // empty tensor
      return input_tensor->GetShapeSize() == 0;
    }
    return true;
  }
  return false;
}

}

namespace Ops {
namespace NN {
namespace Conv {
    constexpr size_t IDX_0 = 0;
    constexpr size_t IDX_1 = 1;
    constexpr size_t IDX_2 = 2;
    constexpr size_t kConv2dDimSizeLimit = 4;
    constexpr int32_t extendDimSizeLimit = 5;
    constexpr int32_t UNKNOWN_SHAPE_DIM = -1;

    ge::graphStatus InferShapeForConvBackprop(
        gert::InferShapeContext* context, size_t const_tensor_idx, const char* const_tensor_name, size_t dim_num)
    {
        OP_CHECK_IF(context == nullptr, CUBE_INNER_ERR_REPORT("", "Get %s failed", "context"), return ge::GRAPH_FAILED);
        const auto op_name = context->GetNodeName();
        auto y_shape = context->GetOutputShape(0);
        OP_CHECK_IF(y_shape == nullptr, CUBE_INNER_ERR_REPORT("", "Get %s failed", "y shape"), return ge::GRAPH_FAILED);
        auto const_tensor = context->GetInputTensor(const_tensor_idx);
        OP_CHECK_IF(
            const_tensor == nullptr, CUBE_INNER_ERR_REPORT(op_name, "get null %s tensor", const_tensor_name),
            return ge::GRAPH_FAILED);
        size_t const_tensor_dim_num = static_cast<size_t>(const_tensor->GetOriginShape().GetShapeSize());
        OP_CHECK_IF(
            const_tensor_dim_num != dim_num,
            CUBE_INNER_ERR_REPORT(op_name, "%s dim num %zu invalid", const_tensor_name, const_tensor_dim_num),
            return ge::GRAPH_FAILED);
        y_shape->SetDimNum(dim_num);

        auto first_input_shape = context->GetInputShape(const_tensor_idx == IDX_0 ? IDX_1 : IDX_0);
        OP_CHECK_IF(
            first_input_shape == nullptr, CUBE_INNER_ERR_REPORT(op_name, "get null input tensor"), return ge::GRAPH_FAILED);
        if (Ops::Base::IsUnknownShape(*first_input_shape) || !IsConstTensor(const_tensor)) {
            for (size_t idx = 0; idx < const_tensor_dim_num; ++idx) {
                y_shape->SetDim(idx, UNKNOWN_SHAPE_DIM);
            }
            return ge::GRAPH_SUCCESS;
        }

        auto dtype = const_tensor->GetDataType();
        if (dtype == ge::DT_INT32) {
            auto tensor_data = const_tensor->GetData<int32_t>();
            for (size_t idx = 0; idx < const_tensor_dim_num; ++idx) {
                y_shape->SetDim(idx, tensor_data[idx]);
            }
        } else if (dtype == ge::DT_INT64) {
            auto tensor_data = const_tensor->GetData<int64_t>();
            for (size_t idx = 0; idx < const_tensor_dim_num; ++idx) {
                y_shape->SetDim(idx, tensor_data[idx]);
            }
        } else {
            CUBE_INNER_ERR_REPORT(
                op_name, "tensor %s not support dtype %s", const_tensor_name,
                ge::TypeUtils::DataTypeToAscendString(dtype).GetString());
            return ge::GRAPH_FAILED;
        }

        OP_LOGD(context->GetNodeName(), "y_shape: %s", Ops::Base::ToString(*y_shape).c_str());
        return ge::GRAPH_SUCCESS;
    }

    // Conv2DBackpropFilterV2 和 Conv2DBackpropFilterV3 共用
    ge::graphStatus InferShapeForConv2DBackpropFilter(gert::InferShapeContext* context)
    {
        auto ret = InferShapeForConvBackprop(context, 1, "filter_size", kConv2dDimSizeLimit);
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }

        const auto runtime_attrs = context->GetAttrs();
        OP_LOGE_IF(runtime_attrs == nullptr, false, context->GetNodeName(), "failed to get runtime attrs");
        const auto from_depthwise = runtime_attrs->GetBool(6); // from_depthwise attr idx: 6
        if (from_depthwise == nullptr || !*from_depthwise) {
            return ge::GRAPH_SUCCESS;
        }

        OP_LOGD(context->GetNodeName(), "transfer from DepthwiseConv2DBackpropFilter, need to reset y shape");
        auto y_shape = context->GetOutputShape(0);
        OP_CHECK_IF(
            y_shape == nullptr, CUBE_INNER_ERR_REPORT(context->GetNodeName(), "y shape is null"), return ge::GRAPH_FAILED);

        const auto y_desc = context->GetOutputDesc(0);
        OP_CHECK_IF(
            y_desc == nullptr, CUBE_INNER_ERR_REPORT(context->GetNodeName(), "y desc is null"), return ge::GRAPH_FAILED);
        const auto y_format = y_desc->GetOriginFormat();
        if (y_format == ge::FORMAT_NCHW) {
            y_shape->SetDim(0, y_shape->GetDim(0) * y_shape->GetDim(1));
            y_shape->SetDim(1, 1);
        } else if (y_format == ge::FORMAT_HWCN) {
            y_shape->SetDim(3, y_shape->GetDim(2) * y_shape->GetDim(3)); // 2: C dim, 3: N dim
            y_shape->SetDim(2, 1);
        }

        OP_LOGD(context->GetNodeName(), "y_shape: %s", Ops::Base::ToString(*y_shape).c_str());
        return ge::GRAPH_SUCCESS;
    }
    // Conv2DBackpropFilterV2 和 Conv2DBackpropFilterV3 共用
    ge::graphStatus InferDataTypeForConv2DBackpropFilter(gert::InferDataTypeContext* context)
    {
        OP_LOGD(context->GetNodeName(), "InferDataTypeForConv2DBackpropFilterV2 enter");
        ge::graphStatus ret = context->SetOutputDataType(0, ge::DT_FLOAT);
        OP_CHECK_IF(
            ret != ge::GRAPH_SUCCESS, CUBE_INNER_ERR_REPORT(context->GetNodeName(), "[InferDataType] Failed."),
            return ge::GRAPH_FAILED);
        OP_LOGD(context->GetNodeName(), "InferDataTypeForConv2DBackpropFilterV2 enter");
        return ge::GRAPH_SUCCESS;
    }
    // Conv2DBackpropInputV2 和 Conv3DBackpropInputV2 共用
    ge::graphStatus InferDataTypeForConvBackpropInputV2(gert::InferDataTypeContext* context)
    {
        OP_LOGD(context->GetNodeName(), "InferDataTypeForConvBackpropInputV2 enter");
        auto filterDataType = context->GetInputDataType(IDX_1);
        ge::graphStatus ret = context->SetOutputDataType(IDX_0, filterDataType);
        OP_CHECK_IF(
            ret != ge::GRAPH_SUCCESS, CUBE_INNER_ERR_REPORT(context->GetNodeName(), "[InferDataType] Failed."),
            return ge::GRAPH_FAILED);
        OP_LOGD(context->GetNodeName(), "InferDataTypeForConvBackpropInputV2 end");
        return ge::GRAPH_SUCCESS;
    }
    // Conv2DTransposeV2 和 Conv3DTransposeV2 共用
    bool CheckOutputAllZero(const gert::Shape* shape)
    {
        size_t dim_num = shape->GetDimNum();
        for (size_t idx = 0; idx < dim_num; ++idx) {
            if (shape->GetDim(idx) != 0) {
                return false;
            }
        }

        return true;
    }
    // Conv2DTransposeV2 和 Conv3DTransposeV2 共用
    ge::graphStatus InferDataTypeForConvTransposeV2(gert::InferDataTypeContext* context)
    {
        OP_LOGD(context->GetNodeName(), "InferDataTypeForConvTransposeV2 enter");
        auto xDataType = context->GetInputDataType(1);
        ge::graphStatus ret = context->SetOutputDataType(0, xDataType);
        OP_CHECK_IF(
            ret != ge::GRAPH_SUCCESS, CUBE_INNER_ERR_REPORT(context->GetNodeName(), "[InferDataType] Failed."),
            return ge::GRAPH_FAILED);
        OP_LOGD(context->GetNodeName(), "InferDataTypeForConvTransposeV2 end");
        return ge::GRAPH_SUCCESS;
    }

    // Conv2DBackpropFilter/Conv2DBackpropInput/Conv2DTranspose 动态场景
    ge::graphStatus InferShapeForConvBackpropExtend3D(gert::InferShapeContext* context, size_t const_tensor_idx, const char* const_tensor_name)
    {
        OP_CHECK_IF(context == nullptr, CUBE_INNER_ERR_REPORT("", "Get %s failed", "context"), return ge::GRAPH_FAILED);
        auto y_shape = context->GetOutputShape(0);
        OP_CHECK_IF(y_shape == nullptr, CUBE_INNER_ERR_REPORT("", "Get %s failed", "y shape"), return ge::GRAPH_FAILED);
        auto const_tensor = context->GetInputTensor(const_tensor_idx);
        const auto op_name = context->GetNodeName();
        OP_CHECK_IF(const_tensor == nullptr, CUBE_INNER_ERR_REPORT(op_name, "get null %s tensor", const_tensor_name), return ge::GRAPH_FAILED);
        size_t const_tensor_dim_num = static_cast<size_t>(const_tensor->GetOriginShape().GetShapeSize());
        OP_CHECK_IF(const_tensor_dim_num != kConv2dDimSizeLimit,
            CUBE_INNER_ERR_REPORT(op_name, "%s dim num %zu invalid", const_tensor_name, const_tensor_dim_num), return ge::GRAPH_FAILED);
        y_shape->SetDimNum(extendDimSizeLimit);
        auto input_shape = context->GetInputShape(const_tensor_idx == IDX_0 ? IDX_1 : IDX_0);
        OP_CHECK_IF(input_shape == nullptr, CUBE_INNER_ERR_REPORT(op_name, "get null input tensor"), return ge::GRAPH_FAILED);
        // 存在动态场景，推导的shape直接需修改为-1
        if (Ops::Base::IsUnknownShape(*input_shape) || !IsConstTensor(const_tensor)) {
            for (size_t idx = 0; idx < extendDimSizeLimit; ++idx) {
                y_shape->SetDim(idx, UNKNOWN_SHAPE_DIM);
            }
            return ge::GRAPH_SUCCESS;
        }
        return SetOutputShapeDim(context, const_tensor, y_shape);
    }

    ge::graphStatus SetOutputShapeDim(const gert::InferShapeContext* context, const gert::Tensor* const_tensor, gert::Shape* y_shape)
    {
        const auto y_desc = context->GetOutputDesc(0);
        OP_CHECK_IF(y_desc == nullptr, CUBE_INNER_ERR_REPORT(context->GetNodeName(), "y desc is null"), return ge::GRAPH_FAILED);
        const auto y_format = y_desc->GetOriginFormat();
        int32_t d_index = GetConvBackpropIndex(y_format);
        OP_CHECK_IF(d_index == -1, CUBE_INNER_ERR_REPORT(context->GetNodeName(), "y format is not support"), return ge::GRAPH_FAILED);
        auto dtype = const_tensor->GetDataType();
        if (dtype == ge::DT_INT32) {
            auto tensor_data = const_tensor->GetData<int32_t>();
            for (int32_t idx = 0; idx < extendDimSizeLimit; ++idx) {
                if (idx < d_index) {
                    y_shape->SetDim(idx, tensor_data[idx]);
                } else if (idx == d_index) {
                    y_shape->SetDim(idx, 1);
                } else {
                    y_shape->SetDim(idx, tensor_data[idx - 1]);
                }
            }
        } else if (dtype == ge::DT_INT64) {
            auto tensor_data = const_tensor->GetData<int64_t>();
            for (int32_t idx = 0; idx < extendDimSizeLimit; ++idx) {
                if (idx < d_index) {
                    y_shape->SetDim(idx, tensor_data[idx]);
                } else if (idx == d_index) {
                    y_shape->SetDim(idx, 1);
                } else {
                    y_shape->SetDim(idx, tensor_data[idx - 1]);
                }
            }
        } else {
            CUBE_INNER_ERR_REPORT(context->GetNodeName(), "tensor not support dtype %s", ge::TypeUtils::DataTypeToAscendString(dtype).GetString());
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }

    int32_t GetConvBackpropIndex(ge::Format format)
    {
        if (format == ge::FORMAT_NCDHW){
            return IDX_2;
        } else if (format == ge::FORMAT_NDHWC) {
            return IDX_1;
        } else if (format  == ge::FORMAT_DHWCN) {
            return IDX_0;
        }
        return UNKNOWN_SHAPE_DIM;
    }

}  // namespace Conv
}  // namespace NN
}  // namespace Ops
