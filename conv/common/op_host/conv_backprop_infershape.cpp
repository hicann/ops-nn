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
 * \file conv_backprop_infershape.cpp
 * \brief
 */
#include "conv_backprop_infershape.h"

namespace {
inline bool IsConstTensor(const gert::Tensor* input_tensor)
{
    if (input_tensor != nullptr) {
        if (input_tensor->GetAddr() == nullptr) {
            // empty tensor
            return input_tensor->GetShapeSize() == 0;
        }
        return true;
    }
    return false;
}

} // namespace

namespace Ops {
namespace NN {
namespace Conv {
constexpr size_t IDX_0 = 0;
constexpr size_t IDX_1 = 1;
constexpr size_t IDX_2 = 2;
constexpr size_t kConv2dDimSizeLimit = 4;
constexpr size_t kConv3dDimSizeLimit = 5;
constexpr int32_t extendDimSizeLimit = 5;
constexpr int32_t UNKNOWN_SHAPE_DIM = -1;
constexpr size_t kXIndex = 0;
constexpr size_t kOutBackpropIndex = 2;
constexpr size_t kGroupsAttrIndex = 3;
constexpr size_t kNDimIndexOfNFirst = 0;
constexpr size_t kCDimIndexOfNSecond = 1;
constexpr size_t kCDimIndexOfHWCN = 2;
constexpr size_t kNDimIndexOfHWCN = 3;

ge::graphStatus InferShapeForConvBackprop(gert::InferShapeContext* context, size_t const_tensor_idx,
                                          const char* const_tensor_name, size_t dim_num)
{
    OP_CHECK_IF(context == nullptr, CUBE_INNER_ERR_REPORT("", "Get %s failed", "context"), return ge::GRAPH_FAILED);
    const auto op_name = context->GetNodeName();
    auto y_shape = context->GetOutputShape(0);
    OP_CHECK_IF(y_shape == nullptr, CUBE_INNER_ERR_REPORT("", "Get %s failed", "y shape"), return ge::GRAPH_FAILED);
    auto const_tensor = context->GetInputTensor(const_tensor_idx);
    OP_CHECK_IF(const_tensor == nullptr, CUBE_INNER_ERR_REPORT(op_name, "get null %s tensor", const_tensor_name),
                return ge::GRAPH_FAILED);
    // unknown场景（[-2]unknown rank / [-1]shape未知），输出直接设为全-1，与V1行为对齐
    const auto& const_tensor_shape = const_tensor->GetOriginShape();
    if (Ops::Base::IsUnknownRank(const_tensor_shape) || Ops::Base::IsUnknownShape(const_tensor_shape)) {
        Ops::Base::SetUnknownShape(static_cast<int64_t>(dim_num), *y_shape);
        return ge::GRAPH_SUCCESS;
    }
    size_t const_tensor_dim_num = static_cast<size_t>(const_tensor_shape.GetShapeSize());
    OP_CHECK_IF(const_tensor_dim_num != dim_num,
                OP_LOGE_FOR_INVALID_SHAPEDIM(op_name, const_tensor_name, std::to_string(const_tensor_dim_num).c_str(),
                                             std::to_string(dim_num).c_str()),
                return ge::GRAPH_FAILED);
    y_shape->SetDimNum(dim_num);

    auto first_input_shape = context->GetInputShape(const_tensor_idx == IDX_0 ? IDX_1 : IDX_0);
    OP_CHECK_IF(first_input_shape == nullptr, CUBE_INNER_ERR_REPORT(op_name, "get null input tensor"),
                return ge::GRAPH_FAILED);
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
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            op_name, const_tensor_name, ge::TypeUtils::DataTypeToAscendString(dtype).GetString(),
            ("The dtype of " + std::string(const_tensor_name) + " must be within the range {DT_INT32, DT_INT64}")
                .c_str());
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(context->GetNodeName(), "y_shape: %s", Ops::Base::ToString(*y_shape).c_str());
    return ge::GRAPH_SUCCESS;
}

// 按format(4D/5D)定位C/N维下标，无法识别的format返回false
static bool GetFilterCnDimIndex(const ge::Format& format, size_t dimNum, size_t& cIdx, size_t& nIdx)
{
    if (dimNum == kConv2dDimSizeLimit) {
        if (format == ge::FORMAT_NCHW) {
            cIdx = kCDimIndexOfNSecond;
            nIdx = kNDimIndexOfNFirst;
        } else if (format == ge::FORMAT_NHWC) {
            cIdx = dimNum - 1;
            nIdx = kNDimIndexOfNFirst;
        } else if (format == ge::FORMAT_HWCN) {
            cIdx = dimNum - 2;
            nIdx = dimNum - 1;
        } else {
            return false;
        }
    } else if (dimNum == kConv3dDimSizeLimit) {
        if (format == ge::FORMAT_NCDHW) {
            cIdx = kCDimIndexOfNSecond;
            nIdx = kNDimIndexOfNFirst;
        } else if (format == ge::FORMAT_NDHWC) {
            cIdx = dimNum - 1;
            nIdx = kNDimIndexOfNFirst;
        } else if (format == ge::FORMAT_DHWCN) {
            cIdx = dimNum - 2;
            nIdx = dimNum - 1;
        } else {
            return false;
        }
    } else {
        return false;
    }
    return true;
}

// bpf家族(groups attr idx: 3，输入序x(0)/filter_size(1)/out_backprop(2))的groups读取
static const int64_t* GetFilterGroupsAttr(const gert::InferShapeContext* context)
{
    const auto runtime_attrs = context->GetAttrs();
    if (runtime_attrs == nullptr || runtime_attrs->GetAttrNum() <= kGroupsAttrIndex) {
        return nullptr;
    }
    return runtime_attrs->GetAttrPointer<int64_t>(kGroupsAttrIndex);
}

// filter_size的const数据在编译期不可见(动态图)时，把结构上可确定的维度填回：
// y的C维=x的C维/groups(cin/groups)、y的N维=out_backprop的C维(cout)，其余维度保持-1交给运行时；
// from_depthwise场景(depthwise dw的C/N语义不同)由调用方跳过。
ge::graphStatus PartialInferFilterShapeWhenConstInvisible(gert::InferShapeContext* context)
{
    const auto op_name = context->GetNodeName();
    auto y_shape = context->GetOutputShape(0);
    OP_CHECK_IF(y_shape == nullptr, CUBE_INNER_ERR_REPORT(op_name, "Get %s failed", "y shape"),
                return ge::GRAPH_FAILED);
    // 仅在输出全未知(const不可见)时补推；const可见或已部分推导的场景直接返回
    bool allUnknown = true;
    for (size_t idx = 0; idx < y_shape->GetDimNum(); ++idx) {
        if (y_shape->GetDim(idx) >= 0) {
            allUnknown = false;
            break;
        }
    }
    if (!allUnknown) {
        return ge::GRAPH_SUCCESS;
    }

    const int64_t* groups = GetFilterGroupsAttr(context);
    OP_CHECK_IF(groups == nullptr || *groups <= 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(op_name, "groups",
                                                      (groups == nullptr ? "0" : std::to_string(*groups)).c_str(),
                                                      "The value of groups must be a positive integer"),
                return ge::GRAPH_FAILED);

    const auto y_desc = context->GetOutputDesc(0);
    OP_CHECK_IF(y_desc == nullptr, CUBE_INNER_ERR_REPORT(op_name, "Get %s failed", "y desc"), return ge::GRAPH_FAILED);
    size_t yCIdx = 0;
    size_t yNIdx = 0;
    if (!GetFilterCnDimIndex(y_desc->GetOriginFormat(), y_shape->GetDimNum(), yCIdx, yNIdx)) {
        return ge::GRAPH_SUCCESS;
    }

    // y的C维 = x的C维/groups
    const auto x_desc = context->GetInputDesc(kXIndex);
    const auto x_shape = context->GetInputShape(kXIndex);
    if (x_desc != nullptr && x_shape != nullptr) {
        size_t xCIdx = 0;
        size_t xNIdx = 0;
        if (GetFilterCnDimIndex(x_desc->GetOriginFormat(), x_shape->GetDimNum(), xCIdx, xNIdx)) {
            const int64_t xC = x_shape->GetDim(xCIdx);
            if (xC > 0) {
                OP_CHECK_IF(xC % *groups != 0,
                            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                                op_name, "x channel", std::to_string(xC).c_str(),
                                ("The channel of x must be divisible by groups " + std::to_string(*groups)).c_str()),
                            return ge::GRAPH_FAILED);
                y_shape->SetDim(yCIdx, xC / *groups);
            }
        }
    }

    // y的N维 = out_backprop的C维
    const auto dedy_desc = context->GetInputDesc(kOutBackpropIndex);
    const auto dedy_shape = context->GetInputShape(kOutBackpropIndex);
    if (dedy_desc != nullptr && dedy_shape != nullptr) {
        size_t dedyCIdx = 0;
        size_t dedyNIdx = 0;
        if (GetFilterCnDimIndex(dedy_desc->GetOriginFormat(), dedy_shape->GetDimNum(), dedyCIdx, dedyNIdx)) {
            const int64_t dedyC = dedy_shape->GetDim(dedyCIdx);
            if (dedyC > 0) {
                y_shape->SetDim(yNIdx, dedyC);
            }
        }
    }

    OP_LOGD(op_name, "partial inferred y_shape: %s", Ops::Base::ToString(*y_shape).c_str());
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
    OP_CHECK_IF(runtime_attrs == nullptr, CUBE_INNER_ERR_REPORT(context->GetNodeName(), "failed to get runtime attrs"),
                return ge::GRAPH_FAILED);
    const auto from_depthwise = runtime_attrs->GetBool(6); // from_depthwise attr idx: 6
    if (from_depthwise == nullptr || !*from_depthwise) {
        return PartialInferFilterShapeWhenConstInvisible(context);
    }

    OP_LOGD(context->GetNodeName(), "transfer from DepthwiseConv2DBackpropFilter, need to reset y shape");
    auto y_shape = context->GetOutputShape(0);
    OP_CHECK_IF(y_shape == nullptr, CUBE_INNER_ERR_REPORT(context->GetNodeName(), "y shape is null"),
                return ge::GRAPH_FAILED);

    const auto y_desc = context->GetOutputDesc(0);
    OP_CHECK_IF(y_desc == nullptr, CUBE_INNER_ERR_REPORT(context->GetNodeName(), "y desc is null"),
                return ge::GRAPH_FAILED);
    const auto y_format = y_desc->GetOriginFormat();
    if (y_format == ge::FORMAT_NCHW) {
        // dw折叠：N维=C维*N维，C维置1
        y_shape->SetDim(kNDimIndexOfNFirst, y_shape->GetDim(kNDimIndexOfNFirst) * y_shape->GetDim(kCDimIndexOfNSecond));
        y_shape->SetDim(kCDimIndexOfNSecond, 1);
    } else if (y_format == ge::FORMAT_HWCN) {
        y_shape->SetDim(kNDimIndexOfHWCN, y_shape->GetDim(kCDimIndexOfHWCN) * y_shape->GetDim(kNDimIndexOfHWCN));
        y_shape->SetDim(kCDimIndexOfHWCN, 1);
    }

    OP_LOGD(context->GetNodeName(), "y_shape: %s", Ops::Base::ToString(*y_shape).c_str());
    return ge::GRAPH_SUCCESS;
}
// Conv2DBackpropFilterV2 和 Conv2DBackpropFilterV3 共用
ge::graphStatus InferDataTypeForConv2DBackpropFilter(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "InferDataTypeForConv2DBackpropFilterV2 enter");
    // 从输出desc读声明dtype透传：desc与InferDataType推导槽位独立存储
    // 推导槽位不预填充声明值(读GetOutputDataType拿不到)，硬编码fp32会污染desc
    // 导致后续to_v3/to_v2 pass读不到声明dtype而跳过Cast插入
    const auto y_desc = context->GetOutputDesc(0);
    if (y_desc != nullptr) {
        const auto declared = y_desc->GetDataType();
        if (declared == ge::DT_FLOAT || declared == ge::DT_FLOAT16 || declared == ge::DT_BF16) {
            ge::graphStatus ret = context->SetOutputDataType(0, declared);
            OP_CHECK_IF(ret != ge::GRAPH_SUCCESS,
                        CUBE_INNER_ERR_REPORT(context->GetNodeName(), "[InferDataType] Failed."),
                        return ge::GRAPH_FAILED);
            return ge::GRAPH_SUCCESS;
        }
    }
    // 无有效声明(如L0直调，输出恒为fp32)时维持fp32
    ge::graphStatus ret = context->SetOutputDataType(0, ge::DT_FLOAT);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, CUBE_INNER_ERR_REPORT(context->GetNodeName(), "[InferDataType] Failed."),
                return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "InferDataTypeForConv2DBackpropFilterV2 end");
    return ge::GRAPH_SUCCESS;
}
// Conv2DBackpropInputV2 和 Conv3DBackpropInputV2 共用
ge::graphStatus InferDataTypeForConvBackpropInputV2(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "InferDataTypeForConvBackpropInputV2 enter");
    auto filterDataType = context->GetInputDataType(IDX_1);
    ge::graphStatus ret = context->SetOutputDataType(IDX_0, filterDataType);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, CUBE_INNER_ERR_REPORT(context->GetNodeName(), "[InferDataType] Failed."),
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
// Conv2DTransposeV2 使用
bool CheckOutputAllZeroFrom2D(const gert::InferShapeContext* context, const gert::Shape* shape, int32_t d_index)
{
    size_t dim_num = shape->GetDimNum();
    OP_CHECK_IF(dim_num != kConv3dDimSizeLimit,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "tensor", std::to_string(dim_num).c_str(),
                                             std::to_string(kConv3dDimSizeLimit).c_str()),
                return false);
    if (shape->GetDim(static_cast<size_t>(d_index)) != 1) {
        return false;
    }
    for (size_t idx = 0; idx < dim_num; ++idx) {
        if (idx != static_cast<size_t>(d_index) && shape->GetDim(idx) != 0) {
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
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, CUBE_INNER_ERR_REPORT(context->GetNodeName(), "[InferDataType] Failed."),
                return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "InferDataTypeForConvTransposeV2 end");
    return ge::GRAPH_SUCCESS;
}

// Conv2DBackpropFilter/Conv2DBackpropInput/Conv2DTranspose 动态场景
ge::graphStatus InferShapeForConvBackpropExtend3D(gert::InferShapeContext* context, size_t const_tensor_idx,
                                                  const char* const_tensor_name)
{
    OP_CHECK_IF(context == nullptr, CUBE_INNER_ERR_REPORT("", "Get %s failed", "context"), return ge::GRAPH_FAILED);
    auto y_shape = context->GetOutputShape(0);
    OP_CHECK_IF(y_shape == nullptr, CUBE_INNER_ERR_REPORT("", "Get %s failed", "y shape"), return ge::GRAPH_FAILED);
    auto const_tensor = context->GetInputTensor(const_tensor_idx);
    const auto op_name = context->GetNodeName();
    OP_CHECK_IF(const_tensor == nullptr, CUBE_INNER_ERR_REPORT(op_name, "get null %s tensor", const_tensor_name),
                return ge::GRAPH_FAILED);
    // unknown场景（[-2]unknown rank / [-1]shape未知），输出直接设为全-1，与V1行为对齐
    const auto& const_tensor_shape = const_tensor->GetOriginShape();
    if (Ops::Base::IsUnknownRank(const_tensor_shape) || Ops::Base::IsUnknownShape(const_tensor_shape)) {
        Ops::Base::SetUnknownShape(extendDimSizeLimit, *y_shape);
        return ge::GRAPH_SUCCESS;
    }
    size_t const_tensor_dim_num = static_cast<size_t>(const_tensor_shape.GetShapeSize());
    OP_CHECK_IF(const_tensor_dim_num != kConv2dDimSizeLimit,
                OP_LOGE_FOR_INVALID_SHAPEDIM(op_name, const_tensor_name, std::to_string(const_tensor_dim_num).c_str(),
                                             std::to_string(kConv2dDimSizeLimit).c_str()),
                return ge::GRAPH_FAILED);
    y_shape->SetDimNum(extendDimSizeLimit);
    auto input_shape = context->GetInputShape(const_tensor_idx == IDX_0 ? IDX_1 : IDX_0);
    OP_CHECK_IF(input_shape == nullptr, CUBE_INNER_ERR_REPORT(op_name, "get null input tensor"),
                return ge::GRAPH_FAILED);
    // 存在动态场景，推导的shape直接需修改为-1
    if (Ops::Base::IsUnknownShape(*input_shape) || !IsConstTensor(const_tensor)) {
        for (size_t idx = 0; idx < extendDimSizeLimit; ++idx) {
            y_shape->SetDim(idx, UNKNOWN_SHAPE_DIM);
        }
        return ge::GRAPH_SUCCESS;
    }
    return SetOutputShapeDim(context, const_tensor, y_shape);
}

ge::graphStatus SetOutputShapeDim(const gert::InferShapeContext* context, const gert::Tensor* const_tensor,
                                  gert::Shape* y_shape)
{
    const auto y_desc = context->GetOutputDesc(0);
    OP_CHECK_IF(y_desc == nullptr, CUBE_INNER_ERR_REPORT(context->GetNodeName(), "y desc is null"),
                return ge::GRAPH_FAILED);
    const auto y_format = y_desc->GetOriginFormat();
    int32_t d_index = GetConvBackpropIndex(y_format);
    OP_CHECK_IF(
        d_index == -1,
        OP_LOGE_FOR_INVALID_FORMAT(context->GetNodeName(), "y", ge::TypeUtils::FormatToSerialString(y_format).c_str(),
                                   "NCDHW or NDHWC or DHWCN"),
        return ge::GRAPH_FAILED);
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
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "tensor",
                                              ge::TypeUtils::DataTypeToAscendString(dtype).GetString(),
                                              "The dtype of tensor must be within the range {DT_INT32, DT_INT64}");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

int32_t GetConvBackpropIndex(ge::Format format)
{
    if (format == ge::FORMAT_NCDHW) {
        return IDX_2;
    } else if (format == ge::FORMAT_NDHWC) {
        return IDX_1;
    } else if (format == ge::FORMAT_DHWCN) {
        return IDX_0;
    }
    return UNKNOWN_SHAPE_DIM;
}

} // namespace Conv
} // namespace NN
} // namespace Ops
