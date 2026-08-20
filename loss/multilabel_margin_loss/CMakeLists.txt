# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

# A5(arch35) 的 tiling 独立成 op_host/arch35/,注册点唯一且仍留在根 tiling,根文件只多一处
# IsRegbaseSocVersion 分派。该分派对 DoMultilabelMarginLossTiling950 的调用在所有 soc 上都会
# 编进根 tiling,故 arch35 那份必须无条件参与编译——若按 COMPUTE_UNIT/TILING_DIR 只在
# ascend950 编,其余 soc(如 kirinx90)链接期即 undefined symbol。运行期由 IsRegbaseSocVersion
# 分流,非 950 不会走到该函数,A2 行为不变。
# add_modules_sources 保持与上游逐字一致(不引入 COMPUTE_UNIT/TILING_DIR,不触碰
# AUTO_SYNC/DISABLE_IN_OPP),arch35 源文件用 target_sources 追加。
# 设置算子定义时支持的芯片类型
set(SUPPORT_COMPUTE_UNIT "ascend950")
# 设置每种芯片类型对应的tiling文件目录，即采用op_host目录下哪个文件夹下的tiling文件编译
set(SUPPORT_TILING_DIR "arch35")
add_modules_sources(HOSTNAME ${OPHOST_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR} OPTYPE multilabel_margin_loss ACLNNTYPE aclnn_exclude COMPUTE_UNIT ${SUPPORT_COMPUTE_UNIT} TILING_DIR ${SUPPORT_TILING_DIR} DISABLE_IN_OPP TRUE)
