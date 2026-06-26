import pytest

from tunelab.models import ParamSpec


def test_param_spec_grid_values_and_serialization() -> None:
    spec = ParamSpec("depth", "int", low=1, high=5, step=2)
    assert spec.grid_values() == [1, 3, 5]
    assert spec.as_dict() == {"name": "depth", "kind": "int", "low": 1, "high": 5, "step": 2}


def test_param_spec_rejects_invalid_bounds() -> None:
    with pytest.raises(ValueError, match="ascending"):
        ParamSpec("x", "float", low=1.0, high=1.0)
    with pytest.raises(ValueError, match="positive bounds"):
        ParamSpec("x", "float", low=0.0, high=1.0, log_scale=True)


def test_bool_and_categorical_spaces_require_choices() -> None:
    with pytest.raises(ValueError, match="requires"):
        ParamSpec("flag", "bool")
    with pytest.raises(ValueError, match="booleans"):
        ParamSpec("flag", "bool", choices=(True, "yes"))
