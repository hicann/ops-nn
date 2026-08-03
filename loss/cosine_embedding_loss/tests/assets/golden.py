#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""Torch CPU golden for CosineEmbeddingLoss."""

import numpy as np
import torch

__golden__ = {"kernel": {"cosine_embedding_loss": "__golden_cosine_embedding_loss"}}

EPS = 1.0e-12


def _resolve(kwargs, margin, reduction):
    attrs = kwargs.get("attrs")
    if isinstance(attrs, dict):
        margin = attrs.get("margin", margin)
        reduction = attrs.get("reduction", reduction)
    return float(margin), str(reduction)


def cosine_embedding_loss_golden(
    x1, x2, target, margin=0.0, reduction="mean", **kwargs
):
    margin, reduction = _resolve(kwargs, margin, reduction)
    a = torch.as_tensor(np.asarray(x1).astype(np.float32), dtype=torch.float32)
    b = torch.as_tensor(np.asarray(x2).astype(np.float32), dtype=torch.float32)
    a, b = torch.broadcast_tensors(a, b)
    if a.dim() < 2:
        raise ValueError("broadcast rank of x1 and x2 must be at least 2")
    t = torch.as_tensor(np.asarray(target).astype(np.float32), dtype=torch.float32)

    dot = torch.sum(a * b, dim=1, dtype=torch.float32)
    s1 = torch.sum(a * a, dim=1, dtype=torch.float32)
    s2 = torch.sum(b * b, dim=1, dtype=torch.float32)
    eps = torch.tensor(EPS, dtype=torch.float32)
    denom = torch.sqrt(s1 + eps) * torch.sqrt(s2 + eps)
    cos = dot / denom

    cos, t = torch.broadcast_tensors(cos, t)
    pos = torch.tensor(1.0, dtype=torch.float32) - cos
    neg = torch.maximum(
        torch.tensor(0.0, dtype=torch.float32),
        cos - torch.tensor(margin, dtype=torch.float32),
    )
    loss = torch.where(
        t == 1.0, pos, torch.where(t == -1.0, neg, torch.zeros_like(pos))
    )

    if reduction == "none":
        return loss.numpy().astype(np.float32, copy=False)
    if reduction == "sum":
        return np.asarray(
            [torch.sum(loss, dtype=torch.float32).item()], dtype=np.float32
        )
    denom_n = loss.numel() if loss.numel() > 0 else 1
    return np.asarray(
        [(torch.sum(loss, dtype=torch.float32) / denom_n).item()], dtype=np.float32
    )


def __golden_cosine_embedding_loss(x1, x2, target, **kwargs):
    return [cosine_embedding_loss_golden(x1, x2, target, **kwargs)]
