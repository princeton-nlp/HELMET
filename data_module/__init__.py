# Import submodules to trigger task registrations
from data_module import icl  # noqa: F401
from data_module import json_kv  # noqa: F401
from data_module import qa  # noqa: F401
from data_module import ruler  # noqa: F401
from data_module import narrativeqa  # noqa: F401
from data_module import rerank  # noqa: F401
from data_module import mrcr  # noqa: F401
from data_module import graphwalk  # noqa: F401
from data_module import multi_lexsum  # noqa: F401
from data_module import infbench  # noqa: F401
from data_module import longbenchv2  # noqa: F401
from data_module import ppl  # noqa: F401

from data_module.base import load_data, TaskConfig, TASK_REGISTRY, TestItemDataset, LENGTH_MAP
