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
 * \file advance_step_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_IS_ADVANCE_STEP_H_
#define OPS_OP_PROTO_INC_IS_ADVANCE_STEP_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {
/**
*@brief The main function of the advcance_step operator is to advance the inference step in vLLM, that is,
* update the model status and generate new inputTokens, inputPostions, seqLen, and slotMapping in each generation step.
* Improves the efficiency of vLLM inference. \n

*@par Inputs:
* Six inputs, including:
*@li input_tokens: A 1-D input tensor. When spec_token and accepted_num are None, length equal to num_seqs. When spec_token and accepted_num are NOT None, length equal to num_seqs * (spec_num + 1). Must be int64 type. Must be int64 type. Format is ND.
*@li sampled_token_ids: A 2-D input tensor. When spec_token and accepted_num are None, the first dim equal to num_queries and the second dim equal to one. When spec_token and accepted_num are NOT None, the first dim equal to num_seqs and the second dim equal to spec_num+1. Must be int64 type.
* Must be int64 type. Format is ND.
*@li input_positions: A 1-D input tensor. When spec_token and accepted_num are None, length equal to num_seqs. When spec_token and accepted_num are NOT None, length equal to num_seqs * (spec_num + 1). Must be int64 type. Format is ND.
*@li seq_lens: A 1-D input tensor. When spec_token and accepted_num are None, length equal to num_seqs. When spec_token and accepted_num are NOT None, length equal to num_seqs * (spec_num + 1). Must be int64 type. Must be int64 type. Format is ND.
*@li slot_mapping: A 1-D input tensor. When spec_token and accepted_num are None, length equal to num_seqs. When spec_token and accepted_num are NOT None, length equal to num_seqs * (spec_num + 1). Must be int64 type. Must be int64 type. Format is ND.
*@li block_tables: When spec_token and accepted_num are None, A 1-D input tensor, and length equal to num_seqs. When spec_token and accepted_num are NOT None, A 2-D input tensor, the first dim equal to num_seqs and the second dim equal to spec_num+1. Must be int64 type. Format is ND.
*@li spec_token: A 2-D optional input tensor, which the first dim equal to num_seqs and the second dim equal to spec_num. Must be int64 type. Format is ND.
*@li accepted_num: A 1-D optional input tensor, and length equal to num_seqs. Must be int64 type. Format is ND. \n

*@par Attributes:
*@li num_seqs: A required Int, which equal to the length of input_tokens, input_positions, seq_lens,
* slot_mapping and block_tables. The value of it must bigger than 0.
*@li num_queries: A required Int, which equal to the length of sampled_token_ids's first dim.
* The value of it must bigger than 0.
*@li block_size: A required Int, which means the basic block length of each block. The value of it must bigger than 0.
\n

*@par Outputs:
*@li input_tokens: A 1-D output tensor. The input tensor input_tokens will self-updating and save as itself.
* Must be int64 type. Format is ND.
*@li input_positions: A 1-D output tensor. The input tensor input_positions will self-updating and save as itself.
* Must be int64 type. Format is ND.
*@li seq_lens: A 1-D output tensor. The input tensor seq_lens will self-updating and save as itself.
* Must be int64 type. Format is ND.
*@li slot_mapping: A 1-D output tensor. The input tensor slot_mapping will self-updating and save as itself.
* Must be int64 type. Format is ND. \n

*/

REG_OP(AdvanceStep)
    .INPUT(input_tokens, TensorType({DT_INT64}))
    .INPUT(sampled_token_ids, TensorType({DT_INT64}))
    .INPUT(input_positions, TensorType({DT_INT64}))
    .INPUT(seq_lens, TensorType({DT_INT64}))
    .INPUT(slot_mapping, TensorType({DT_INT64}))
    .INPUT(block_tables, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(spec_token, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(accepted_num, TensorType({DT_INT64}))
    .OUTPUT(input_tokens, TensorType({DT_INT64}))
    .OUTPUT(input_positions, TensorType({DT_INT64}))
    .OUTPUT(seq_lens, TensorType({DT_INT64}))
    .OUTPUT(slot_mapping, TensorType({DT_INT64}))
    .REQUIRED_ATTR(num_seqs, Int)
    .REQUIRED_ATTR(num_queries, Int)
    .REQUIRED_ATTR(block_size, Int)
    .OP_END_FACTORY_REG(AdvanceStep)

} // namespace ge

#endif // OPS_OP_PROTO_INC_IS_ADVANCE_STEP_H_
