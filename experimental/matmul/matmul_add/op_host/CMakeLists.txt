# Copyright (c) 2025 Huawei Technologies Co., Ltd.

add_modules_sources(HOSTNAME ${OPHOST_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR} OPTYPE matmul_add ACLNNTYPE aclnn_exclude)
