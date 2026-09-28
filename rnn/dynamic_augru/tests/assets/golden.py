#!/usr/bin/env python3
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""DynamicAUGRU FP64 CPU golden, FP32 XPU reference and CPU self-tests.

CPU projections, activations and recurrence use FP64; XPU computation uses
ordinary FP32 without compensated arithmetic. Outputs use the state input
dtype in ordinary calls. TTK Promote calls retain FP64 golden outputs for
three-way comparison. Output rounding does not feed either recurrence.
"""

import re
from functools import wraps

import numpy as np
import torch

__spec__ = {"dynamic_augru": "DynamicAUGRUTestSpec"}
_OUTPUT_NAMES = ("y", "output_h", "update", "update_att", "reset", "new", "hidden_new")


_NONFINITE_CASE = re.compile(
    r"^dynamic_augru_special_\d+_nonfinite_"
    r"(?P<value>nan|posinf|neginf)_"
    r"(?P<input>x|weight_input|weight_hidden|weight_att|bias_input|bias_hidden|init_h)_"
    r"(?P<position>first|middle|tail|padding)_"
)


def dynamic_augru_inputs(
    x,
    weight_input,
    weight_hidden,
    weight_att,
    bias_input=None,
    bias_hidden=None,
    seq_length=None,
    init_h=None,
    **kwargs,
):
    """Constrain semantic inputs and inject named NaN/Inf test values."""
    testcase_name = str(kwargs.get("testcase_name", ""))
    np.asarray(weight_att)[...] = np.clip(weight_att, 0.0, 1.0)
    if seq_length is not None and np.asarray(seq_length).dtype == np.float16:
        time, batch, hidden = np.asarray(seq_length).shape
        lengths = np.arange(batch, dtype=np.int32) % (time + 1)
        mask = np.arange(time, dtype=np.int32)[:, None, None] < lengths[None, :, None]
        np.asarray(seq_length)[...] = np.broadcast_to(mask, (time, batch, hidden))

    match = _NONFINITE_CASE.match(testcase_name)
    if match is not None:
        arrays = {
            "x": x,
            "weight_input": weight_input,
            "weight_hidden": weight_hidden,
            "weight_att": weight_att,
            "bias_input": bias_input,
            "bias_hidden": bias_hidden,
            "init_h": init_h,
        }
        target_name = match.group("input")
        target = arrays[target_name]
        if target is None:
            raise ValueError(f"{testcase_name}: target input {target_name} is absent")
        target = np.asarray(target)
        value = {
            "nan": np.nan,
            "posinf": np.inf,
            "neginf": -np.inf,
        }[match.group("value")]
        position = match.group("position")
        if position == "padding":
            if target_name == "x":
                target[-1, 0, 0] = value
            elif target_name == "weight_att":
                target[-1, 0] = value
            else:
                raise ValueError(
                    f"{testcase_name}: padding position is invalid for {target_name}"
                )
        else:
            flat_index = {
                "first": 0,
                "middle": target.size // 2,
                "tail": target.size - 1,
            }[position]
            target.flat[flat_index] = value
        arrays[target_name] = target
        x = arrays["x"]
        weight_input = arrays["weight_input"]
        weight_hidden = arrays["weight_hidden"]
        weight_att = arrays["weight_att"]
        bias_input = arrays["bias_input"]
        bias_hidden = arrays["bias_hidden"]
        init_h = arrays["init_h"]
    return (
        x,
        weight_input,
        weight_hidden,
        weight_att,
        bias_input,
        bias_hidden,
        seq_length,
        init_h,
    )


def _torch_dynamic_augru_golden(
    x,
    weight_input,
    weight_hidden,
    weight_att,
    bias_input=None,
    bias_hidden=None,
    seq_length=None,
    init_h=None,
    gate_order="zrh",
    state_dtype=None,
    *,
    compute_dtype=torch.float64,
    promoted=False,
):
    if compute_dtype not in (torch.float32, torch.float64):
        raise ValueError("compute dtype must be float32 or float64")
    device = "cpu" if compute_dtype == torch.float64 else None
    x, wi, wh, attention = (
        torch.as_tensor(v, device=device)
        for v in (x, weight_input, weight_hidden, weight_att)
    )
    operand_dtype = torch.float32 if promoted else torch.float16
    if any(v.dtype != operand_dtype for v in (x, wi, wh, attention)):
        raise ValueError("x, weights and attention have an unexpected dtype")
    if x.ndim != 3 or wh.ndim != 2 or wi.ndim != 2 or attention.ndim != 2:
        raise ValueError(
            "x/weight_input/weight_hidden/weight_att ranks must be 3/2/2/2"
        )
    time, batch, input_size = x.shape
    hidden = wh.shape[0]
    if min(time, batch, input_size, hidden) <= 0:
        raise ValueError("T/B/I/H must all be positive")
    if (
        wi.shape != (input_size, 3 * hidden)
        or wh.shape != (hidden, 3 * hidden)
        or attention.shape != (time, batch)
    ):
        raise ValueError("inconsistent input shapes")
    if gate_order not in ("zrh", "rzh"):
        raise ValueError("gate_order must be zrh or rzh")
    states = [
        torch.as_tensor(v, device=x.device) if v is not None else None
        for v in (bias_input, bias_hidden, init_h)
    ]
    inferred = next((v.dtype for v in states if v is not None), x.dtype)
    state_dtype = inferred if state_dtype is None else state_dtype
    state_types = (
        (torch.float32, torch.float64) if promoted else (torch.float16, torch.float32)
    )
    if state_dtype not in state_types or state_dtype != inferred:
        raise ValueError("state dtype must match the first present state input")
    for value, shape in zip(states, ((3 * hidden,), (3 * hidden,), (1, batch, hidden))):
        if value is not None and (value.dtype != state_dtype or value.shape != shape):
            raise ValueError("state input dtype or shape mismatch")
    sequence = (
        None if seq_length is None else torch.as_tensor(seq_length, device=x.device)
    )
    if sequence is not None:
        shape = (batch,) if sequence.dtype == torch.int32 else (time, batch, hidden)
        if (
            sequence.dtype not in (torch.int32, operand_dtype)
            or sequence.shape != shape
        ):
            raise ValueError("seq_length must be int32 [B] or float16 [T,B,H]")
    x, wi, wh, attention = (v.to(dtype=compute_dtype) for v in (x, wi, wh, attention))
    bi, bh, initial = (None if v is None else v.to(dtype=compute_dtype) for v in states)
    previous = (
        torch.zeros((batch, hidden), dtype=compute_dtype, device=x.device)
        if initial is None
        else initial[0]
    )
    projection = x @ wi
    if bi is not None:
        projection = projection + bi
    outputs = [[] for _ in _OUTPUT_NAMES]
    zi, ri = (0, 1) if gate_order == "zrh" else (1, 0)
    for t in range(time):
        gh = previous @ wh
        if bh is not None:
            gh = gh + bh
        gx, gh = projection[t].chunk(3, dim=-1), gh.chunk(3, dim=-1)
        update = torch.sigmoid(gx[zi] + gh[zi])
        reset = torch.sigmoid(gx[ri] + gh[ri])
        candidate = torch.tanh(gx[2] + reset * gh[2])
        update_att = (1.0 - attention[t, :, None]) * update
        current = candidate + update_att * (previous - candidate)
        if sequence is not None:
            if sequence.dtype == torch.int32:
                condition = t < sequence[:, None]
                current = torch.where(condition, current, previous)
            else:
                mask = sequence[t].to(dtype=compute_dtype)
                current = previous + mask * (current - previous)
        values = (current, current, update, update_att, reset, candidate, gh[2])
        for out, value in zip(outputs, values):
            out.append(value.to(dtype=torch.float64 if promoted else state_dtype))
        previous = current
    return {name: torch.stack(out) for name, out in zip(_OUTPUT_NAMES, outputs)}


def dynamic_augru_golden(*args, **kwargs):
    # Preserve the previous NumPy dtype argument at the I/O boundary.
    dtype = kwargs.get("state_dtype")
    if dtype is not None and not isinstance(dtype, torch.dtype):
        kwargs["state_dtype"] = torch.from_numpy(np.empty((), dtype=dtype)).dtype
    return {
        name: value.detach().cpu().numpy()
        for name, value in _torch_dynamic_augru_golden(*args, **kwargs).items()
    }


def _golden_tensors(
    x,
    weight_input,
    weight_hidden,
    weight_att,
    bias_input=None,
    bias_hidden=None,
    seq_length=None,
    init_h=None,
    direction="UNIDIRECTIONAL",
    cell_depth=1,
    keep_prob=1.0,
    cell_clip=-1.0,
    num_proj=0,
    time_major=True,
    activation="tanh",
    gate_order="zrh",
    reset_after=True,
    is_training=True,
    **kwargs,
):
    fixed = (
        direction,
        cell_depth,
        keep_prob,
        cell_clip,
        num_proj,
        time_major,
        activation,
        reset_after,
    )
    if fixed != ("UNIDIRECTIONAL", 1, 1.0, -1.0, 0, True, "tanh", True):
        raise ValueError("unsupported DynamicAUGRU attributes")
    outputs = _torch_dynamic_augru_golden(
        x,
        weight_input,
        weight_hidden,
        weight_att,
        bias_input,
        bias_hidden,
        seq_length,
        init_h,
        gate_order,
        compute_dtype=kwargs.get("_compute_dtype", torch.float64),
        promoted=(
            kwargs.get("golden_mode") == "Promote"
            and kwargs.get("_compute_dtype", torch.float64) == torch.float64
        ),
    )
    return [outputs[name] for name in _OUTPUT_NAMES]


@wraps(_golden_tensors)
def _third_party_golden(*args, **kwargs):
    # TTK supplies tensors on the selected XPU; keep all seven outputs there.
    x = args[0] if args else kwargs["x"]
    device_type = x.device.type
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        if device_type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
        with torch.no_grad(), torch.autocast(device_type=device_type, enabled=False):
            kwargs["_compute_dtype"] = torch.float32
            return _golden_tensors(*args, **kwargs)
    finally:
        if device_type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = old_tf32


class DynamicAUGRUTestSpec:
    @staticmethod
    def customize_inputs(*args, **kwargs):
        return dynamic_augru_inputs(*args, **kwargs)

    @staticmethod
    @wraps(_golden_tensors)
    def golden(*args, **kwargs):
        return [
            value.detach().cpu().numpy() for value in _golden_tensors(*args, **kwargs)
        ]

    third_party = {"torch": _third_party_golden}
    tolerance = {
        "float16": {"standard": "cross_check", "level": "L0"},
        "float32": {"standard": "cross_check", "level": "L0"},
    }


def _run_self_tests():
    import math
    from types import SimpleNamespace
    import unittest

    g = SimpleNamespace(
        dynamic_augru_golden=_torch_dynamic_augru_golden,
        DynamicAUGRUTestSpec=DynamicAUGRUTestSpec,
    )
    cpu = SimpleNamespace(
        dynamic_augru_golden=dynamic_augru_golden,
        DynamicAUGRUTestSpec=DynamicAUGRUTestSpec,
    )

    def inputs(time=3, batch=1, hidden=1, state_dtype=torch.float32):
        return dict(
            x=torch.zeros(time, batch, 1, dtype=torch.float16),
            weight_input=torch.zeros(1, 3 * hidden, dtype=torch.float16),
            weight_hidden=torch.zeros(hidden, 3 * hidden, dtype=torch.float16),
            weight_att=torch.zeros(time, batch, dtype=torch.float16),
            init_h=torch.ones(1, batch, hidden, dtype=state_dtype),
        )

    class GoldenTests(unittest.TestCase):
        def test_seven_outputs_hand_computed_recurrence(self):
            for dtype in (torch.float16, torch.float32):
                with self.subTest(dtype=dtype):
                    result = g.dynamic_augru_golden(**inputs(state_dtype=dtype))
                expected = torch.tensor([0.5, 0.25, 0.125], dtype=dtype).reshape(
                    3, 1, 1
                )
                for name in ("y", "output_h"):
                    torch.testing.assert_close(result[name], expected, rtol=0, atol=0)
                for name in ("update", "update_att", "reset"):
                    torch.testing.assert_close(
                        result[name], torch.full_like(expected, 0.5), rtol=0, atol=0
                    )
                for name in ("new", "hidden_new"):
                    torch.testing.assert_close(
                        result[name], torch.zeros_like(expected), rtol=0, atol=0
                    )

        def test_attention_and_gate_order(self):
            args = inputs(time=2)
            args["bias_input"] = torch.tensor([0.7, -0.4, 0.2])
            args["bias_hidden"] = torch.tensor([0.1, 0.2, -0.3])
            args["weight_att"][:] = 1
            zrh = g.dynamic_augru_golden(**args)
            for name in ("bias_input", "bias_hidden"):
                args[name] = args[name][[1, 0, 2]]
            rzh = g.dynamic_augru_golden(**args, gate_order="rzh")
            for name in zrh:
                torch.testing.assert_close(zrh[name], rzh[name], rtol=0, atol=0)
            torch.testing.assert_close(zrh["y"], zrh["new"], rtol=0, atol=0)
            self.assertTrue(bool((zrh["update_att"] == 0).all()))

        def test_dense_recurrence_against_scalar_mathematical_reference(self):
            torch.manual_seed(42)
            args = inputs(time=3, batch=2, hidden=2)
            for name in ("x", "weight_input", "weight_hidden", "weight_att", "init_h"):
                args[name] = (torch.randn_like(args[name]) * 0.2).to(args[name].dtype)
            args["bias_input"] = torch.randn(6) * 0.2
            args["bias_hidden"] = torch.randn(6) * 0.2
            x, wi, wh, att, initial = (
                args[n].tolist()
                for n in ("x", "weight_input", "weight_hidden", "weight_att", "init_h")
            )
            bi, bh = args["bias_input"].tolist(), args["bias_hidden"].tolist()
            expected = torch.empty(7, 3, 2, 2)
            previous = initial[0]
            for t in range(3):
                next_state = [[0.0] * 2 for _ in range(2)]
                for b in range(2):
                    gx = [x[t][b][0] * wi[0][j] + bi[j] for j in range(6)]
                    gh = [
                        sum(previous[b][k] * wh[k][j] for k in range(2)) + bh[j]
                        for j in range(6)
                    ]
                    for h in range(2):
                        z = 1.0 / (1.0 + math.exp(-gx[h] - gh[h]))
                        r = 1.0 / (1.0 + math.exp(-gx[2 + h] - gh[2 + h]))
                        n = math.tanh(gx[4 + h] + r * gh[4 + h])
                        za = (1.0 - att[t][b]) * z
                        current = n + za * (previous[b][h] - n)
                        next_state[b][h] = current
                        expected[:, t, b, h] = torch.tensor(
                            [current, current, z, za, r, n, gh[4 + h]]
                        )
                previous = next_state
            result = g.dynamic_augru_golden(**args)
            torch.testing.assert_close(
                torch.stack(list(result.values())), expected, rtol=2e-6, atol=2e-7
            )

        def test_length_padding_selects_previous_despite_nan(self):
            args = inputs()
            args["weight_att"][1:] = float("nan")
            args["seq_length"] = torch.tensor([1], dtype=torch.int32)
            result = g.dynamic_augru_golden(**args)
            torch.testing.assert_close(
                result["y"], torch.full((3, 1, 1), 0.5), rtol=0, atol=0
            )
            self.assertTrue(bool(torch.isnan(result["update_att"][1:]).all()))
            args["seq_length"] = torch.zeros(3, 1, 1, dtype=torch.float16)
            self.assertTrue(
                bool(torch.isnan(g.dynamic_augru_golden(**args)["y"][1:]).all())
            )

        def test_optional_inputs_and_invalid_contract(self):
            args = inputs()
            args.pop("init_h")
            result = g.dynamic_augru_golden(**args)
            self.assertEqual(result["y"].dtype, torch.float16)
            self.assertEqual(result["y"].count_nonzero(), 0)
            for changes in (
                {"x": args["x"].float()},
                {"gate_order": "invalid"},
                {
                    "bias_input": torch.zeros(3),
                    "init_h": torch.zeros(1, 1, 1, dtype=torch.float16),
                },
                {"seq_length": torch.zeros(3, dtype=torch.int32)},
                {"x": torch.zeros(0, 1, 1, dtype=torch.float16)},
            ):
                with (
                    self.subTest(changes=tuple(changes)),
                    self.assertRaises(ValueError),
                ):
                    g.dynamic_augru_golden(**(args | changes))

        def test_third_party_preserves_tensor_outputs(self):
            for dtype in (torch.float16, torch.float32):
                args = inputs(state_dtype=dtype)
                actual = DynamicAUGRUTestSpec.third_party["torch"](**args)
                expected = _golden_tensors(**args, _compute_dtype=torch.float32)
                self.assertEqual(len(actual), 7)
                for value, reference in zip(actual, expected):
                    self.assertEqual(value.device, args["x"].device)
                    self.assertEqual(value.dtype, dtype)
                    torch.testing.assert_close(value, reference, rtol=0, atol=0)

        def test_cpu_fp64_and_xpu_fp32_intermediates(self):
            from torch.utils._python_dispatch import TorchDispatchMode

            class CaptureMath(TorchDispatchMode):
                def __init__(self):
                    self.dtypes = []

                def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                    if func in (
                        torch.ops.aten.mm.default,
                        torch.ops.aten.bmm.default,
                        torch.ops.aten.sigmoid.default,
                        torch.ops.aten.tanh.default,
                    ):
                        self.dtypes.append(args[0].dtype)
                    return func(*args, **(kwargs or {}))

            for entry, dtype in (
                (DynamicAUGRUTestSpec.golden, torch.float64),
                (DynamicAUGRUTestSpec.third_party["torch"], torch.float32),
            ):
                capture = CaptureMath()
                with capture:
                    entry(**inputs())
                self.assertTrue(capture.dtypes)
                self.assertEqual(set(capture.dtypes), {dtype})

        def test_promoted_cpu_golden_preserves_fp64_outputs(self):
            for state_dtype in (torch.float16, torch.float32):
                args = inputs(state_dtype=state_dtype)
                args["seq_length"] = torch.ones(3, 1, 1, dtype=torch.float16)
                promoted = {
                    name: value.to(
                        torch.float32 if value.dtype == torch.float16 else torch.float64
                    )
                    for name, value in args.items()
                }
                result = DynamicAUGRUTestSpec.golden(**promoted, golden_mode="Promote")
                for output in result:
                    self.assertEqual(output.dtype, np.float64)
                np.testing.assert_array_equal(result[0].reshape(-1), [0.5, 0.25, 0.125])

        def test_large_shape_and_numpy_bridge(self):
            import numpy as np

            args = inputs(time=5, hidden=769)
            args = {name: value.numpy() for name, value in args.items()}
            result = cpu.DynamicAUGRUTestSpec.golden(**args)
            self.assertEqual(len(result), 7)
            self.assertEqual(result[0].shape, (5, 1, 769))
            self.assertEqual(result[0].dtype, np.float32)
            # The previous public NumPy dtype argument remains a supported I/O bridge.
            small = {
                name: value[:1] if name in ("x", "weight_att") else value
                for name, value in args.items()
            }
            direct = cpu.dynamic_augru_golden(**small, state_dtype=np.float32)
            self.assertEqual(direct["y"].dtype, np.float32)
            expected = (
                torch.tensor([0.5, 0.25, 0.125, 0.0625, 0.03125])
                .reshape(5, 1, 1)
                .expand(5, 1, 769)
            )
            torch.testing.assert_close(
                torch.from_numpy(result[0]), expected, rtol=0, atol=0
            )

    torch.set_num_threads(1)
    unittest.main(module=SimpleNamespace(GoldenTests=GoldenTests))


if __name__ == "__main__":
    _run_self_tests()
