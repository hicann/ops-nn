#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""TTK golden and third-party reference for INTrainingUpdateGradGammaBeta."""

import importlib
import itertools
import math

import numpy as np


def _accurate_sum(value):
    """Use a fast FP64 sum, recovering ill-conditioned finite columns with fsum."""
    # TTK Promote supplies FP64 inputs. Never narrow them or the CPU truth.
    value = np.asarray(value)
    with np.errstate(invalid="ignore", over="ignore"):
        total = np.sum(value, axis=0, keepdims=True, dtype=np.float64)
        if value.size:
            magnitude = np.sum(np.abs(value), axis=0, keepdims=True, dtype=np.float64)
            error_bound = (
                np.finfo(np.float64).eps * max(value.shape[0] - 1, 0) * magnitude
            )
            risky = np.isfinite(magnitude) & (
                error_bound > np.abs(total) * np.finfo(np.float32).eps / 4
            )
            rows = value.reshape(value.shape[0], total.size)
            for column in np.flatnonzero(risky):
                total.flat[column] = math.fsum(float(x) for x in rows[:, column])
        return total


class _TorchReference:
    """Native Torch composition shared by competitor precision and timing."""

    def __init__(self, **_):
        pass

    @staticmethod
    def _sum(value):
        # Deferred import: this class only runs on the remote XPU server.
        # A top-level torch import makes local TTK workers pre-load torch,
        # which segfaults against the TBE native libraries.
        torch = importlib.import_module("torch")
        return torch.sum(value, dim=0, keepdim=True)

    def __call__(self, res_gamma, res_beta, **_):
        return [self._sum(res_gamma), self._sum(res_beta)]


class InTrainingUpdateGradGammaBetaSpec:
    @staticmethod
    def golden(res_gamma, res_beta, **_):
        return [_accurate_sum(res_gamma), _accurate_sum(res_beta)]

    @staticmethod
    def customize_inputs(res_gamma, res_beta, **kwargs):
        testcase_name = kwargs.get("testcase_name", "")
        if "_multiscale_" in testcase_name:
            sequence = np.array(
                [2.0**100, 2.0**60, 1.0, -(2.0**100), -(2.0**60), 2.0**-100, -0.5],
                dtype=np.float32,
            )
            gamma = np.zeros_like(res_gamma)
            rows = gamma.reshape(gamma.shape[0], -1)
            if testcase_name.endswith("permutations"):
                rows[:] = np.array(
                    list(itertools.permutations(sequence)), dtype=np.float32
                ).T
            elif testcase_name.endswith("full_range"):
                powers = np.ldexp(
                    np.ones(12, dtype=np.float32), np.arange(127, -150, -24)
                )
                values = np.concatenate(
                    (powers, -powers, np.array([0.5], dtype=np.float32))
                )
                rows[:] = values[:, None]
            elif testcase_name.endswith("four_levels"):
                values = np.array(
                    [
                        2.0**100,
                        2.0**60,
                        2.0**20,
                        1.0,
                        -(2.0**100),
                        -(2.0**60),
                        -(2.0**20),
                    ],
                    dtype=np.float32,
                )
                rows[:] = values[:, None]
            else:
                full_rows = rows.shape[0] // sequence.size * sequence.size
                if full_rows:
                    rows[:full_rows] = np.tile(sequence, full_rows // sequence.size)[
                        :, None
                    ]
                else:
                    rows[:] = sequence[: rows.shape[0], None]
                if testcase_name.endswith("mixed_columns"):
                    maximum = np.finfo(np.float32).max
                    tiny = np.nextafter(np.float32(0), np.float32(1))
                    patterns = np.array(
                        [
                            sequence,
                            sequence[::-1],
                            np.zeros(7),
                            np.ones(7),
                            [maximum, 1.0, -maximum, 2.0**60, 1.0, -(2.0**60), -0.5],
                            [np.inf, 0, 0, 0, 0, 0, 0],
                            [-np.inf, 0, 0, 0, 0, 0, 0],
                            [np.inf, -np.inf, 0, 0, 0, 0, 0],
                            [np.nan, 0, 0, 0, 0, 0, 0],
                            [maximum, maximum, 0, 0, 0, 0, 0],
                            [tiny, tiny, tiny, -tiny, 0, 0, 0],
                        ],
                        dtype=np.float32,
                    ).T
                    rows[:] = patterns[:, np.arange(rows.shape[1]) % patterns.shape[1]]
            return gamma, -np.roll(gamma, 1, axis=-1)
        if testcase_name.endswith("cancellation_orders"):
            maximum = np.finfo(np.float32).max
            permutations = np.array(
                list(itertools.permutations([maximum, 1.0, -maximum, 0.0])),
                dtype=np.float32,
            ).T
            gamma = permutations.reshape(res_gamma.shape).copy()
            return gamma, -gamma
        if testcase_name.endswith(
            ("overflow_boundary_equal", "overflow_boundary_below")
        ):
            threshold = np.nextafter(
                np.float32(float(np.finfo(np.float32).max) / res_gamma.shape[0]),
                np.float32(0),
            )
            if testcase_name.endswith("overflow_boundary_below"):
                threshold = np.nextafter(threshold, np.float32(0))
            gamma = np.full_like(res_gamma, threshold)
            return gamma, -gamma
        if testcase_name.endswith("scale_boundary_neighbors"):
            ceiling_log2 = (res_gamma.shape[0] - 1).bit_length()
            threshold = np.float32(math.ldexp(1.0, 127 - ceiling_log2))
            values = np.array(
                [
                    np.nextafter(threshold, np.float32(0)),
                    threshold,
                    np.nextafter(threshold, np.float32(np.inf)),
                    1.0,
                    -threshold,
                ],
                dtype=np.float32,
            )
            gamma = np.empty_like(res_gamma)
            rows = gamma.reshape(gamma.shape[0], -1)
            rows[:] = np.resize(values, rows.shape[1])
            return gamma, -gamma
        if testcase_name.endswith("extreme_cancellation"):
            max_value = np.finfo(np.float32).max
            gamma = np.empty_like(res_gamma)
            beta = np.empty_like(res_beta)
            gamma[0].fill(max_value)
            gamma[1].fill(-max_value)
            gamma[2].fill(1.0)
            gamma[3].fill(0.0)
            beta[0].fill(max_value)
            beta[1].fill(max_value)
            beta[2].fill(-max_value)
            beta[3].fill(-max_value)
            return gamma, beta
        if testcase_name.endswith("extreme_overflow"):
            max_value = np.finfo(np.float32).max
            gamma = np.full_like(res_gamma, max_value)
            beta = np.full_like(res_beta, -max_value)
            return gamma, beta
        if testcase_name.endswith("extreme_nonfinite"):
            max_value = np.finfo(np.float32).max
            gamma = np.zeros_like(res_gamma)
            beta = np.zeros_like(res_beta)
            gamma_rows = gamma.reshape(gamma.shape[0], -1)
            beta_rows = beta.reshape(beta.shape[0], -1)
            gamma_rows[:, 0] = [np.nan, 1.0, 2.0, 3.0]
            gamma_rows[:, 1] = [np.inf, 1.0, -2.0, 3.0]
            gamma_rows[:, 2] = [-np.inf, 1.0, -2.0, 3.0]
            gamma_rows[:, 3] = [np.inf, -np.inf, 0.0, 0.0]
            gamma_rows[:, 4] = [max_value, -max_value, 1.0, 0.0]
            gamma_rows[:, 5] = [max_value, max_value, 0.0, 0.0]
            gamma_rows[:, 6] = [-max_value, -max_value, 0.0, 0.0]
            gamma_rows[:, 7] = [1.0, -1.0, 0.0, 0.0]
            beta_rows[:, 0] = [np.inf, -np.inf, 1.0, 0.0]
            beta_rows[:, 1] = [np.nan, 0.0, 0.0, 0.0]
            beta_rows[:, 2] = [np.inf, 0.0, 0.0, 0.0]
            beta_rows[:, 3] = [-np.inf, 0.0, 0.0, 0.0]
            beta_rows[:, 4] = [max_value, -max_value, -1.0, 0.0]
            beta_rows[:, 5] = [max_value, max_value, -max_value, -max_value]
            beta_rows[:, 6] = [0.0, 0.0, 0.0, 0.0]
            beta_rows[:, 7] = [1.0, 2.0, 3.0, 4.0]
            return gamma, beta
        if "_dfx_" in testcase_name:
            gamma = np.zeros_like(res_gamma)
            beta = np.zeros_like(res_beta)
            if testcase_name.endswith("_dfx_nan"):
                gamma[0].reshape(-1)[0] = np.nan
                beta[-1].reshape(-1)[-1] = np.nan
            elif testcase_name.endswith("_dfx_single_inf"):
                gamma[0].reshape(-1)[0] = np.inf
                beta[0].reshape(-1)[1] = -np.inf
            elif testcase_name.endswith("_dfx_mixed_inf"):
                gamma[0].reshape(-1)[0] = np.inf
                gamma[1].reshape(-1)[0] = -np.inf
                beta[0].reshape(-1)[1] = np.inf
            elif testcase_name.endswith("_dfx_finite_pattern"):
                values = np.arange(gamma.size, dtype=np.float32).reshape(gamma.shape)
                gamma[...] = (values % 17.0 - 8.0) * 0.25
                beta[...] = ((values * 3.0) % 19.0 - 9.0) * 0.125
            elif testcase_name.endswith("_dfx_subnormal"):
                positive = np.nextafter(np.float32(0.0), np.float32(1.0))
                negative = np.nextafter(np.float32(0.0), np.float32(-1.0))
                gamma[...] = positive
                beta[...] = negative
            elif testcase_name.endswith("_dfx_cancellation"):
                gamma[0, ...] = np.float32(1.0e8)
                gamma[1, ...] = np.float32(1.0)
                gamma[2, ...] = np.float32(-1.0e8)
                beta[0, ...] = np.float32(-1.0e8)
                beta[1, ...] = np.float32(-1.0)
                beta[2, ...] = np.float32(1.0e8)
            elif testcase_name.endswith("_dfx_gamma_zero_beta_pattern"):
                values = np.arange(beta.size, dtype=np.float32).reshape(beta.shape)
                beta[...] = (values % 23.0 - 11.0) * 0.0625
            elif testcase_name.endswith("_dfx_finite_overflow"):
                max_finite = np.finfo(np.float32).max
                gamma[...] = max_finite
                beta[...] = -max_finite
            return gamma, beta
        return res_gamma, res_beta

    third_party = {"torch": _TorchReference}
    tolerance = {
        "float32": {"standard": "stat_rel_err", "threshold": 1.0e-4},
    }


def in_training_update_grad_gamma_beta_golden(res_gamma, res_beta, **kwargs):
    return tuple(
        InTrainingUpdateGradGammaBetaSpec.golden(
            res_gamma,
            res_beta,
            **kwargs,
        )
    )


__spec__ = {
    "in_training_update_grad_gamma_beta": "InTrainingUpdateGradGammaBetaSpec",
}
__golden__ = {
    "kernel": {
        "in_training_update_grad_gamma_beta": "in_training_update_grad_gamma_beta_golden",
    }
}
