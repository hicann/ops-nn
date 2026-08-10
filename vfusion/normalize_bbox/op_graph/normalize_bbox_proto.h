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
 * \file normalize_bbox_proto.h
 * \brief NormalizeBBox operator graph prototype (REG_OP registration).
 *
 * Declares the graph-level prototype so NormalizeBBox is usable in GE graph mode
 * (compiled into libopgraph_nn.so via add_graph_plugin_sources).
 * dtype list aligns with the legacy built-in declaration (nn_detect_ops.h):
 * fp16/fp32 only.
 */
#ifndef OPS_VFUSION_NORMALIZE_BBOX_PROTO_H_
#define OPS_VFUSION_NORMALIZE_BBOX_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief Computes Normalize bbox function.
 * Normalize the pre-selected boxes (after NMS) by the image height and width:
 *   y = boxes / [h, w, h, w]  (per batch, h = shape_hw[b, 0], w = shape_hw[b, 1])
 *
 * @par Inputs:
 * Inputs include:
 * @li boxes: A Tensor. Must be float16 or float32.
 *            rank in [2, 8]; dim0 = batch; coord axis (size 4) at the last dim
 *            (reversed_box=false) or at dim1 (reversed_box=true); the product of
 *            the remaining middle dims is the box number per batch.
 * @li shape_hw: A Tensor. Must be int32, shape (batch, 3).
 *            Per batch [h, w, *]; only the first two elements are used,
 *            the third is reserved (legacy img_shape (h, w, c) convention).
 *
 * @par Attributes:
 * reversed_box: optional, bool. Defaults to "false".
 *               false: coord axis at the last dim, e.g. (batch, ..., 4);
 *               true:  coord axis at dim1,        e.g. (batch, 4, ...).
 *
 * @par Outputs:
 * y: A Tensor. Must have the same type and shape as boxes.
 */
#ifndef OPS_PROTO_DEF_NORMALIZEBBOX
#define OPS_PROTO_DEF_NORMALIZEBBOX
REG_OP(NormalizeBBox)
    .INPUT(boxes, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(shape_hw, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(reversed_box, Bool, false)
    .OP_END_FACTORY_REG(NormalizeBBox)
#endif // OPS_PROTO_DEF_NORMALIZEBBOX
} // namespace ge

#endif // OPS_VFUSION_NORMALIZE_BBOX_PROTO_H_
