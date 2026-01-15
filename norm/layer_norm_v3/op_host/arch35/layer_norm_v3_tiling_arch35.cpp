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
 * \file layer_norm_v3_tiling_arch35.cpp
 * \brief
 */
#include "layer_norm_v3_tiling_arch35.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "layer_norm_v3_tiling.h"
#include "../../../../matmul/common/op_host/op_tiling/tiling_cache.h"
#include "common/runtime2_util.h"
#include "../../../../matmul/common/op_host/op_tiling/hash.h"
#include "op_host/cache_runinfo.h"
#include <nlohmann/json.hpp>

namespace optiling {
#define LN_MAX_AXIS_NUM 8
const int64_t axis = 0;
constexpr size_t ATTR_EPSILON_IDX = 2;
const gert::Shape g_vec_1_shape = {1};

struct LayerNormV3CacheKeyWord {
    uint32_t ci_key;
    int32_t axis_attr;
    size_t shape_size;
    int64_t shape_dims[LN_MAX_AXIS_NUM];
    ge::DataType dtype;
    float epsilon;
};

static Ops::NN::TilingCache<
    OpHashInput<LayerNormV3CacheKeyWord>, GenericHashItem<OpHashInput<LayerNormV3CacheKeyWord>>>
    op_tiling_cache;

static ge::graphStatus TilingPrepare4LayerNormV3(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "begin to do TilingPrepare4LayerNormV3.");
    LayerNormV3OpInfo* compile_info = GetCompileInfoPtr<LayerNormV3OpInfo>(context);
    OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);

    return TilingPrepare4LayerNormV3ForAscendC(context, compile_info->regbaseCompileInfo);
}

static inline const gert::Shape& EnsureNotScalar(const gert::Shape& in_shape)
{
    if (in_shape.IsScalar()) {
        return g_vec_1_shape;
    }
    return in_shape;
}

static ge::graphStatus LayerNormV3UnknowAxisTiling(gert::TilingContext* context, const LayerNormV3OpInfo* op_info)
{
    OP_LOGD(context->GetNodeName(), "LayerNormV3UnknowAxisTiling running.");
    const gert::StorageShape* input_shape_cls = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, input_shape_cls);
    auto src_td = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, src_td);
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const gert::Shape& input_shape = EnsureNotScalar(input_shape_cls->GetStorageShape());
    const std::size_t input_shape_dim = input_shape.GetDimNum();

    // get attr for reduce axis
    const int64_t* begin_norm_axis = attrs->GetAttrPointer<int64_t>(axis);
    OP_CHECK_NULL_WITH_CONTEXT(context, begin_norm_axis);
    int32_t reduce_attr = *begin_norm_axis < 0 ?
                              static_cast<int32_t>(*begin_norm_axis) + static_cast<int32_t>(input_shape_dim) :
                              static_cast<int32_t>(*begin_norm_axis);

    ge::DataType input_x_dtype = src_td->GetDataType();
    LayerNormV3CacheKeyWord key_word;
    memset_s(&key_word, sizeof(key_word), 0, sizeof(key_word));
    key_word.ci_key = op_info->ci_key;
    key_word.dtype = input_x_dtype;
    key_word.axis_attr = reduce_attr;
    key_word.shape_size = input_shape.GetDimNum();
    for (size_t i = 0; i < key_word.shape_size; i++) {
        key_word.shape_dims[i] = input_shape.GetDim(i);
    }

    const float* epsilon_ptr = attrs->GetAttrPointer<float>(ATTR_EPSILON_IDX);
    key_word.epsilon = (epsilon_ptr == nullptr) ? 0.0 : *epsilon_ptr;
    OpHashInput<LayerNormV3CacheKeyWord> hash_input(key_word);
    uint32_t hash_key = Ops::NN::MurmurHash(&hash_input, sizeof(hash_input));
    GenericHashItem<OpHashInput<LayerNormV3CacheKeyWord>> hash_item;
    if (op_tiling_cache.Get(hash_key, hash_input, hash_item)) {
        hash_item.GetContext(*context);
        return ge::GRAPH_SUCCESS;
    }
    std::vector<int64_t> reduce_axis(input_shape_dim - reduce_attr, 0);
    for (int32_t i = 0; i < static_cast<int32_t>(input_shape_dim - reduce_attr); i++) {
        reduce_axis[i] = reduce_attr + i;
    }
    // do autotiling
    const std::vector<gert::Shape> input_gert_shapes = {input_shape};

    // update mean cof
    gert::TilingData* tiling_data = context->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling_data);
    OP_CHECK_IF(
        op_info->reduce_mean_cof_dtype.empty(), OP_LOGD(context->GetNodeName(), "LayerNormV3UnknowAxisTiling end"),
        return ge::GRAPH_SUCCESS);

    OP_LOGD(context->GetNodeName(), "LayerNormV3UnknowAxisTiling will do AddReduceMeanCof");
    OP_CHECK_IF(
        !AddReduceMeanCof(input_shape, op_info->reduce_mean_cof_ge_dtype, reduce_axis, tiling_data),
        OP_LOGE(context->GetNodeName(), "do AddReduceMeanCof failed"), return ge::GRAPH_FAILED);

    if (hash_item.SetContext(*context, hash_input)) {
        op_tiling_cache.Add(hash_key, hash_input, hash_item);
    };
    OP_LOGD(context->GetNodeName(), "LayerNormV3UnknowAxisTiling end.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4LayerNormV3(gert::TilingContext* context)
{
    // compile info
    const LayerNormV3OpInfo* compile_info = reinterpret_cast<const LayerNormV3OpInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);
    // norm template tiling_stratery
    OP_LOGD(context->GetNodeName(), "LayerNormV3Tiling running.");

    return Tiling4LayerNormV3ForAscendC(context);
}

// register tiling interface of LayerNormV3 op.
IMPL_OP_OPTILING(LayerNormV3).Tiling(Tiling4LayerNormV3).TilingParse<LayerNormV3OpInfo>(TilingPrepare4LayerNormV3);
} // namespace optiling
