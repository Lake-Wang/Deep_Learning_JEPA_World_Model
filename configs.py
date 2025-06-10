import argparse
import dataclasses
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Tuple, Union, cast, List

from omegaconf import OmegaConf

DataClass = Any
DataClassType = Any


@dataclass
class JEPAConfig:
    """
    Config for JEPA model
    """
    batch_size: int = 512
    num_epochs: int = 15
    learning_rate: float = 5e-4

    repr_dim: int = 128
    hidden_dim: int = 128
    action_dim: int = 2
    
    momentum: float = 0.996
    # loss
    pred_loss_w: float = 5.0
    var_loss_w: float = 5.0
    cov_loss_w: float = 1.0
    raw_mse_w: float = 1.0
    
    # path
    checkpt_path: str = "./model_weights.pth"


@dataclass
class ConfigBase:
    """Base class that should handle parsing from command line,
    json, dicts.
    """
    
    @classmethod
    def parse_from_command_line(cls):
        return omegaconf_parse(cls)
    
    @classmethod
    def parse_from_file(cls, path: str):
        oc = OmegaConf.load(path)
        return cls.parse_from_dict(OmegaConf.to_container(oc))

    
    @classmethod
    def parse_from_command_line_deprecated(cls):
        result = DataclassArgParser(
            cls, fromfile_prefix_chars="@"
        ).parse_args_into_dataclasses()
        if len(result) > 1:
            raise RuntimeError(
                f"The following arguments were not recognized: {result[1:]}"
            )
        return result[0]

    @classmethod
    def parse_from_dict(cls, inputs):
        return DataclassArgParser._populate_dataclass_from_dict(cls, inputs.copy())

    @classmethod
    def parse_from_flat_dict(cls, inputs):
        return DataclassArgParser._populate_dataclass_from_flat_dict(cls, inputs.copy())
    
    def save(self, path: str):
        with open(path, "w") as f:
            OmegaConf.save(config=self, f=f)
