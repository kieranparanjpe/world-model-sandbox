from copy import deepcopy
from typing import Dict, Any

from ml_commons.config import RunInfo
from ml_commons.log import Logger


class ConsoleLogger(Logger):
    def __init__(self, run_info: RunInfo, entity: str, project: str,
                 hyperparameters: Dict[str, Any], elements: Dict[str, Any], default_x_axis="global_step"):
        super().__init__()

        self._elements_start = elements
        self._elements = deepcopy(self._elements_start)
        self._elements_prefix = {}


    def finish(self):
        pass

    def add_elements(self, elements: Dict[str, Any]):
        self._elements.update(deepcopy(elements))
        self._elements_start.update(elements)

    def set_element_step_metric(self, elements: Dict[str, str]):
        pass

    def reset(self, *fields: str):
        if fields is None or len(fields) == 0:
            self._elements = deepcopy(self._elements_start)
        else:
            for field in fields:
                self._elements[field] = self._elements_start[field]

    def set_log_data(self, kvps: Dict[str, Any]):
        self._elements.update(kvps)

    def sum_log_data(self, kvps: Dict[str, Any]):
        for k, v in kvps.items():
            self._elements[k] += v

    def log_data(self, *fields):
        prefixed_elements = {f"{self._elements_prefix.get(k, "")}{k}": v for k, v in self._elements.items()}

        data = prefixed_elements if not fields else {k: v for k, v in prefixed_elements.items() if k in fields}

        line = " | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in data.items())
        print(line)

    def set_prefix(self, elements : Dict[str, str]):
        for k, v in elements.items():
            self._elements_prefix[k] = v