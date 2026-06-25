# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.

import torch

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi


def _get_reduce_name(reduce):
    reduce_map = {
        0: "none",
        1: "sum",
        2: "prod",
        3: "amax",
        4: "amin",
        5: "mean",
    }
    if reduce not in reduce_map:
        raise ValueError(f"unsupported reduce value: {reduce}")
    return reduce_map[reduce]


@register("function_scatter_reduce_for_aclnn_scatter_reduce")
class FunctionScatterReduceApi(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        torch.use_deterministic_algorithms(True)
        input_data.kwargs["input"] = input_data.kwargs["self"]
        del input_data.kwargs["self"]

        input_data.kwargs["include_self"] = bool(input_data.kwargs["include_self"])
        reduce = _get_reduce_name(input_data.kwargs["reduce"])
        del input_data.kwargs["reduce"]

        if reduce == "none":
            output = torch.scatter(
                input_data.kwargs["input"],
                input_data.kwargs["dim"],
                input_data.kwargs["index"],
                input_data.kwargs["src"],
            )
            if input_data.kwargs["include_self"]:
                base = input_data.kwargs["input"].clone()
                output = base.scatter_(
                    input_data.kwargs["dim"],
                    input_data.kwargs["index"],
                    input_data.kwargs["src"],
                )
        else:
            output = eval(self.api_name)(*input_data.args, reduce=reduce, **input_data.kwargs)

        if not with_output:
            return output

        if self.output is None:
            return output

        if isinstance(self.output, int):
            return input_data.args[self.output]
        if isinstance(self.output, str):
            return input_data.kwargs[self.output]
        raise ValueError(f"self.output {self.output} value is error")