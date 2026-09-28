# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

# aclnnLSTM dispatches to SingleLayerLstm on Ascend950, so op_api/aclnn_lstm.cpp and
# op_api/lstm_single_layer_adapter.h call l0op::SingleLayerLstm and l0op::SingleLayerLstmSupports,
# which rnn/single_layer_lstm/op_api/single_layer_lstm.cpp defines. A whole-repository build puts
# both in one libopapi_nn.so and links; the single-operator package built for the smoke test
# compiles only the operators DEPENDENCIES names, so without this its libcust_opapi.so exports a
# body whose two callees are absent and the example fails to link.
add_modules_sources(
    HOSTNAME ${OPHOST_NAME}
    MODE PRIVATE
    DIR ${CMAKE_CURRENT_SOURCE_DIR}
    OPTYPE dynamic_rnn
    ACLNNTYPE aclnn_exclude
    DEPENDENCIES single_layer_lstm
)
