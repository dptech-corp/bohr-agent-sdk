"""
Unit tests for calculation_mcp_server: get_schema_annotation and _traverse_and_process.
"""
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional
from unittest.mock import patch, MagicMock

import pytest
from pydantic import BaseModel

from dp.agent.server.calculation_mcp_server import (
    get_schema_annotation,
    _schema_model_cache,
    _traverse_and_process,
)


def _clear_schema_cache():
    _schema_model_cache.clear()


# --- get_schema_annotation ---


def test_get_schema_annotation_none():
    assert get_schema_annotation(None) is None


def test_get_schema_annotation_type_none():
    assert get_schema_annotation(type(None)) is type(None)


def test_get_schema_annotation_annotated_path():
    ann = Annotated[Path, "path field"]
    result = get_schema_annotation(ann)
    assert result is str


def test_get_schema_annotation_annotated_nested():
    ann = Annotated[Annotated[Optional[Path], "x"], "y"]
    result = get_schema_annotation(ann)
    assert result == Optional[str]


def test_get_schema_annotation_optional_path():
    assert get_schema_annotation(Optional[Path]) == Optional[str]


def test_get_schema_annotation_optional_str():
    assert get_schema_annotation(Optional[str]) == Optional[str]


def test_get_schema_annotation_path():
    assert get_schema_annotation(Path) is str


def test_get_schema_annotation_list_path():
    assert get_schema_annotation(List[Path]) == List[str]


def test_get_schema_annotation_optional_list_path():
    assert get_schema_annotation(Optional[List[Path]]) == Optional[List[str]]


def test_get_schema_annotation_dict_str_path():
    assert get_schema_annotation(Dict[str, Path]) == Dict[str, str]


def test_get_schema_annotation_optional_dict_str_path():
    assert get_schema_annotation(Optional[Dict[str, Path]]) == Optional[Dict[str, str]]


def test_get_schema_annotation_dict_str_list_path():
    assert get_schema_annotation(Dict[str, List[Path]]) == Dict[str, List[str]]


def test_get_schema_annotation_list_str():
    assert get_schema_annotation(List[str]) == List[str]


def test_get_schema_annotation_list_any():
    assert get_schema_annotation(List[Any]) == List[Any]


def test_get_schema_annotation_dict_value_schema():
    assert get_schema_annotation(Dict[str, List[Path]]) == Dict[str, List[str]]


def test_get_schema_annotation_dict_no_args_unchanged():
    assert get_schema_annotation(dict) is dict


def test_get_schema_annotation_basemodel_with_path_field():
    class ModelWithPath(BaseModel):
        name: str
        data_path: Path

    _clear_schema_cache()
    schema_ann = get_schema_annotation(ModelWithPath)
    assert schema_ann is not ModelWithPath
    assert hasattr(schema_ann, "model_fields")
    assert "name" in schema_ann.model_fields
    assert "data_path" in schema_ann.model_fields
    assert schema_ann.model_fields["data_path"].annotation is str
    assert schema_ann.model_fields["name"].annotation is str


def test_get_schema_annotation_basemodel_cached():
    class CachedModel(BaseModel):
        x: Path

    _clear_schema_cache()
    first = get_schema_annotation(CachedModel)
    second = get_schema_annotation(CachedModel)
    assert first is second
    assert CachedModel in _schema_model_cache


def test_get_schema_annotation_basemodel_nested():
    class Inner(BaseModel):
        path_field: Path

    class Outer(BaseModel):
        inner: Inner
        top_path: Optional[Path] = None

    _clear_schema_cache()
    schema_ann = get_schema_annotation(Outer)
    assert "inner" in schema_ann.model_fields
    assert "top_path" in schema_ann.model_fields
    inner_schema = schema_ann.model_fields["inner"].annotation
    assert hasattr(inner_schema, "model_fields")
    assert inner_schema.model_fields["path_field"].annotation is str
    assert schema_ann.model_fields["top_path"].annotation == Optional[str]


def test_get_schema_annotation_plain_types():
    assert get_schema_annotation(str) is str
    assert get_schema_annotation(int) is int
    assert get_schema_annotation(bool) is bool


# --- _traverse_and_process ---


def test_traverse_annotation_none_returns_value():
    value = {"a": 1}
    result = _traverse_and_process(
        value, None, "local", MagicMock(), {}, "input_name"
    )
    assert result == value


def test_traverse_primitive_str_unchanged():
    result = _traverse_and_process(
        "hello", str, "local", MagicMock(), {}, "input_name"
    )
    assert result == "hello"


def test_traverse_primitive_int_unchanged():
    result = _traverse_and_process(
        42, int, "local", MagicMock(), {}, "input_name"
    )
    assert result == 42


def test_traverse_path_local_string():
    result = _traverse_and_process(
        "/tmp/foo", Path, "local", MagicMock(), {}, "input_name"
    )
    assert result == Path("/tmp/foo")


def test_traverse_path_empty_string_returns_dot():
    result = _traverse_and_process(
        "", Path, "local", MagicMock(), {}, "input_name"
    )
    assert result == Path(".")


def test_traverse_path_uri_calls_download():
    input_artifacts = {}
    mock_storage = MagicMock()
    with patch(
        "dp.agent.server.calculation_mcp_server._download_artifact",
        return_value="/downloaded/path",
    ) as mock_download:
        result = _traverse_and_process(
            "local://bucket/key",
            Path,
            "local",
            mock_storage,
            input_artifacts,
            "data",
        )
    assert result == Path("/downloaded/path")
    mock_download.assert_called_once()
    call_kw = mock_download.call_args
    assert call_kw[0][0] == "local://bucket/key"
    assert call_kw[0][-1] == []


def test_traverse_path_uri_appends_path_trace():
    input_artifacts = {}
    with patch(
        "dp.agent.server.calculation_mcp_server._download_artifact",
        return_value="/downloaded/path",
    ) as mock_download:
        _traverse_and_process(
            "http://example.com/file",
            Path,
            "local",
            MagicMock(),
            input_artifacts,
            "input_name",
            path_trace=["config", "file"],
        )
    call_args = mock_download.call_args[0]
    assert call_args[-1] == ["config", "file"]


def test_traverse_list_of_paths_local():
    value = ["/a", "/b"]
    result = _traverse_and_process(
        value, List[Path], "local", MagicMock(), {}, "input_name"
    )
    assert result == [Path("/a"), Path("/b")]


def test_traverse_list_path_trace():
    value = ["/a", "/b"]
    result = _traverse_and_process(
        value,
        List[Path],
        "local",
        MagicMock(),
        {},
        "input_name",
        path_trace=["items"],
    )
    assert result == [Path("/a"), Path("/b")]


def test_traverse_list_nested_basemodel():
    class Item(BaseModel):
        path: Path

    value = [{"path": "/p1"}, {"path": "/p2"}]
    result = _traverse_and_process(
        value, List[Item], "local", MagicMock(), {}, "input_name"
    )
    assert len(result) == 2
    assert result[0].path == Path("/p1")
    assert result[1].path == Path("/p2")


def test_traverse_dict_str_path():
    value = {"a": "/path/a", "b": "/path/b"}
    result = _traverse_and_process(
        value, Dict[str, Path], "local", MagicMock(), {}, "input_name"
    )
    assert result == {"a": Path("/path/a"), "b": Path("/path/b")}


def test_traverse_dict_path_trace():
    value = {"x": "/x"}
    result = _traverse_and_process(
        value,
        Dict[str, Path],
        "local",
        MagicMock(),
        {},
        "input_name",
        path_trace=["config"],
    )
    assert result == {"x": Path("/x")}


def test_traverse_basemodel_from_dict():
    class Model(BaseModel):
        name: str
        data_path: Path

    value = {"name": "test", "data_path": "/tmp/data"}
    result = _traverse_and_process(
        value, Model, "local", MagicMock(), {}, "input_name"
    )
    assert isinstance(result, Model)
    assert result.name == "test"
    assert result.data_path == Path("/tmp/data")


def test_traverse_basemodel_from_instance():
    class Model(BaseModel):
        path: Path

    inst = Model(path="/foo")
    result = _traverse_and_process(
        inst, Model, "local", MagicMock(), {}, "input_name"
    )
    assert isinstance(result, Model)
    assert result.path == Path("/foo")


def test_traverse_basemodel_nested():
    class Inner(BaseModel):
        p: Path

    class Outer(BaseModel):
        inner: Inner
        label: str

    value = {"inner": {"p": "/nested"}, "label": "ok"}
    result = _traverse_and_process(
        value, Outer, "local", MagicMock(), {}, "input_name"
    )
    assert isinstance(result, Outer)
    assert result.inner.p == Path("/nested")
    assert result.label == "ok"


def test_traverse_basemodel_skips_none_fields():
    class Model(BaseModel):
        required: Path
        optional: Optional[Path] = None

    value = {"required": "/r", "optional": None}
    result = _traverse_and_process(
        value, Model, "local", MagicMock(), {}, "input_name"
    )
    assert result.required == Path("/r")
    assert result.optional is None


def test_traverse_basemodel_non_dict_value_returned_unchanged():
    class Model(BaseModel):
        x: int

    result = _traverse_and_process(
        "not a dict", Model, "local", MagicMock(), {}, "input_name"
    )
    assert result == "not a dict"


def test_traverse_optional_path_present():
    result = _traverse_and_process(
        "/some/path", Optional[Path], "local", MagicMock(), {}, "input_name"
    )
    assert result == Path("/some/path")


def test_traverse_optional_path_none():
    with pytest.raises(TypeError, match="expected str, bytes or os.PathLike"):
        _traverse_and_process(
            None, Optional[Path], "local", MagicMock(), {}, "input_name"
        )
