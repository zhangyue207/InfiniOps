import functools
import importlib.util
from pathlib import Path

import pytest


pytest.importorskip("clang.cindex")


@functools.lru_cache(maxsize=1)
def _load_generator():
    script = Path(__file__).parents[1] / "scripts" / "generate_wrappers.py"
    spec = importlib.util.spec_from_file_location("generate_wrappers", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _generate_binding(op_name):
    generator = _load_generator()
    operator = generator._OperatorExtractor()(op_name)
    return generator._generate_pybind11(operator)


def test_mha_varlen_fwd_requires_out_binding():
    text = _generate_binding("mha_varlen_fwd")

    assert 'py::arg("out"), py::arg("cu_seqlens_q")' in text
    assert 'py::arg("out") = py::none()' not in text
    assert "std::optional<py::object> out" not in text
    assert "OptionalTensorFromPybind11Handle(out)" not in text


def test_mha_fwd_kvcache_requires_out_binding():
    text = _generate_binding("mha_fwd_kvcache")

    assert 'py::arg("out"), py::arg("softmax_scale")' in text
    assert 'py::arg("out") = py::none()' not in text
    assert "std::optional<py::object> out" not in text
    assert "OptionalTensorFromPybind11Handle(out)" not in text
