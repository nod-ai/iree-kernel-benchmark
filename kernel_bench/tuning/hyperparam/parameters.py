from enum import Enum
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, List, Optional, override
import wave_lang.kernel.lang as tkl
from wave_lang.kernel._support.indexing import index_symbol


@dataclass
class TuningBounds(ABC):
    @abstractmethod
    def get_range(self) -> list[int]:
        pass

    @abstractmethod
    def get_value(self, val: int) -> int:
        pass

    @abstractmethod
    def load_value_json(self, value: Any) -> int:
        pass

    @abstractmethod
    def dump_value_json(self, value: int) -> Any:
        pass


@dataclass
class IntegerBounds(TuningBounds):
    min: int
    max: int
    step: int
    exponential: bool = False

    @override
    def get_range(self):
        range = []
        curr = self.min
        while curr <= self.max:
            range.append(curr)
            if self.exponential:
                curr *= self.step
            else:
                if curr == 1 and self.step > 1:
                    curr += self.step - 1
                else:
                    curr += self.step
        return range

    @override
    def get_value(self, val):
        return val

    @override
    def load_value_json(self, value):
        assert isinstance(value, int)
        return value

    @override
    def dump_value_json(self, value):
        return int(value)


@dataclass
class CategoricalBounds(TuningBounds):
    options: List[Any]

    def __post_init__(self):
        assert len(self.options) > 0
        option = self.options[0]
        if isinstance(option, (list, tuple)):
            self._low_option = option[0]
        else:
            self._low_option = option

    @override
    def get_range(self):
        return list(range(len(self.options)))

    @override
    def get_value(self, val):
        return self.options[val]

    @override
    def dump_value_json(self, value):
        assert len(self.options) > 0

        # Handle iterables
        if isinstance(value, list):
            return [self.dump_value_json(v) for v in value]
        if isinstance(value, tuple):
            return tuple([self.dump_value_json(v) for v in value])
        if isinstance(value, dict):
            return {k: self.dump_value_json(v) for k, v in value.items()}

        # Handle primitives & enums
        if isinstance(value, Enum):
            return value.name
        if isinstance(value, (float, int, bool, str)):
            return value

        raise ValueError(f"Type {type(value)} not supported for serialization")

    def _parse_json_value(self, value):
        if isinstance(self._low_option, Enum):
            EnumType = type(self._low_option)
            value_enum = EnumType[value]
            return value_enum
        if isinstance(value, (float, int, bool, str)):
            return value
        raise ValueError(
            f"Type {type(value)} not supported for tuning parameter serialization"
        )

    @override
    def load_value_json(self, value):
        assert len(self.options) > 0

        # Handle iterables
        if isinstance(value, (list, tuple)):
            parsed_label = tuple([self._parse_json_value(v) for v in value])
            if parsed_label in self.options:
                return self.options.index(parsed_label)
            else:
                raise ValueError(
                    f"Label {parsed_label} not supported in available options: {self.options}"
                )

        return self._parse_json_value(value)


class TuningParameter:
    def __init__(
        self,
        name: str,
        bounds: TuningBounds,
        initial_value: Optional[int] = None,
        include_hyperparam: bool = True,
    ):
        self.name = name
        self.bounds = bounds
        self.include_hyperparam = include_hyperparam
        self._default_value = initial_value
        self._value = initial_value
        self.validate_value(self._value)

    @property
    def value(self) -> Optional[Any]:
        if self._default_value is None and self._value is None:
            return None
        return self.bounds.get_value(self._value or self._default_value)

    @value.setter
    def value(self, val: int):
        self.validate_value(val)
        self._value = val

    def validate_value(self, val: int):
        if val is None:
            return
        valid_range = self.bounds.get_range()
        if val not in valid_range:
            raise ValueError(
                f"Value {val} not in valid range for parameter '{self.name}'. "
                f"Valid values: {valid_range}"
            )

    def __repr__(self):
        return f"TuningParameter(name={self.name}, value={self._value}, range={self.bounds.get_range()})"


class TuningSpec:
    def __init__(self, tuning_params: Optional[List[TuningParameter]] = None):
        if tuning_params:
            self._params = {param.name: param for param in tuning_params}
        else:
            self._params = {}

    def load_from_dict(self, obj: dict[str, Any]):
        for pname, param in self._params.items():
            if pname in obj:
                param.value = param.bounds.load_value_json(obj[pname])

    def add_parameter(self, name: str, param: TuningParameter):
        self._params[name] = param

    def get_parameter(self, name: str) -> TuningParameter:
        if name not in self._params:
            raise ValueError(f"Could not find parameter {name}.")
        return self._params.get(name)

    def get_parameter_value(self, name: str) -> Optional[int]:
        return self.get_parameter(name).value

    def set_parameter(self, name: str, value: int):
        if name not in self._params:
            raise ValueError(f"Could not find parameter {name}.")
        self._params[name].value = value

    def clear(self):
        for param in self._params.values():
            param.value = None

    def params(self) -> List[TuningParameter]:
        return list(self._params.values())

    def hyperparams(self) -> dict[tkl.IndexSymbol, int]:
        return {
            index_symbol(param.name): param.value
            for param in self._params.values()
            if param.value and param.include_hyperparam
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            param.name: param.bounds.dump_value_json(param.value)
            for param in self._params.values()
            if param.value
        }
