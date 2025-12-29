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
 * \file embedding_hash_table_apply_adam_w_proto.h
 * \brief
 */

#ifndef EMBEDDING_HASH_TABLE_APPLY_ADAM_W_PROTO_H_
#define EMBEDDING_HASH_TABLE_APPLY_ADAM_W_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {

/**
* @brief Updates embedding hash table according to the AdamW algorithm.

* @par Inputs:
* @li table_handle: A Tensor, dtype is int64. 1-D. Stores the address of the hashtable.
* @li keys: A Tensor, dtype is int64. 1-D. Indicates the hashtable key.
*           Should be deduplicated, otherwise can not guarantee the correctness of the results.
* @li m: A Tensor, dtype is float16 or float, 2-D. Indicates the first moment estimate.
*        Shape is (bucket_size, embedding_dim).
* @li v: A Tensor, dtype is float16 or float, 2-D. Indicates the second moment estimate.
*        Shape is (bucket_size, embedding_dim).
* @li beta1_power: A Tensor, dtype is float16 or float. 1-D. Used to cache intermediate calculated values.
*                  Value is beta1*(step-1). Range is [0.0, 1.0).
* @li beta2_power: A Tensor, dtype is the same as "beta1_power". 1-D. Used to cache intermediate calculated values.
*                  Value is beta2*(step-1). Range is [0.0, 1.0).
* @li lr: A Tensor, dtype is the same as "beta1_power". 1-D. Indicates the learning rate. Range is [0.0, INF).
* @li weight_decay: A Tensor, dtype is the same as "beta1_power". 1-D. Indicates the weight decay. Range is [0.0, INF).
* @li beta1: A Tensor, dtype is the same as "beta1_power". 1-D. Indicates the first order momentum. Range is [0.0, 1.0).
* @li beta2: A Tensor, dtype is the same as "beta1_power". 1-D. Indicates the second order momentum. Range is [0.0, 1.0).
* @li epsilon: A Tensor, dtype is the same as "beta1_power". 1-D. Indicates the small value param. Range is [0.0, 1.0).
* @li grad: A Tensor, dtype is the same as "beta1_power". 2-D. Indicates the grad. Shape is (keys number, embedding_dim).
* @li max_grad_norm: A mutable Tensor of the same type as "m". Indicates the gradient parameter.
*                    Shape is (bucket_size, embedding_dim).

* @par Outputs:
* @li m: A Tensor, dtype is float16 or float, 2-D. Update of the input m value.
* @li v: A Tensor, dtype is float16 or float, 2-D. Update of the input v value.
* @li beta1_power: Update the value of input beta1_power.
*                  Used to cache intermediate calculated values. Dtype is float16 or float. 1-D.
* @li beta2_power: Update the value of input beta2_power.
*                  Used to cache intermediate calculated values. Dtype is the same as "beta1_power". 1-D.
* @li max_grad_norm: A mutable Tensor of the same type as "m". Update of the input max_grad_norm value.\n

* @par Attributes:
* @li embedding_dim: An Int, indicates the dim of embedding value in hashtable.
* @li bucket_size: An Int, indicates the size of hash map.
* @li amsgrad: An optional bool, indicates whether to use the AMSGrad variant of htis algorithm from
*     the paper On the Convergence of Adam and Beyond.
*     If "True", max_grad_norm input and output must be entered.
* @li maximize: An optional bool, maximize the params based on the objective. \n
*/
REG_OP(EmbeddingHashTableApplyAdamW)
    .INPUT(table_handle, TensorType({DT_INT64}))
    .INPUT(keys, TensorType({DT_INT64}))
    .INPUT(m, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(v, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(beta1_power, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(beta2_power, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(lr, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(weight_decay, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(beta1, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(beta2, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(epsilon, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(grad, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(max_grad_norm, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(m, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(v, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(beta1_power, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(beta2_power, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(max_grad_norm, TensorType({DT_FLOAT16, DT_FLOAT}))
    .REQUIRED_ATTR(embedding_dim, Int)
    .REQUIRED_ATTR(bucket_size, Int)
    .ATTR(amsgrad, Bool, false)
    .ATTR(maximize, Bool, false)
    .OP_END_FACTORY_REG(EmbeddingHashTableApplyAdamW)

} // namespace ge
#endif  // EMBEDDING_HASH_TABLE_APPLY_ADAM_W_PROTO_H_
