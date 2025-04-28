import inspect
from collections.abc import Sequence
from typing import Annotated, Any, List

from mcp.server.fastmcp.exceptions import InvalidSignature
from mcp.server.fastmcp.utilities.func_metadata import ArgModelBase, _get_typed_annotation, FuncMetadata
from pydantic import Field, WithJsonSchema, create_model
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined


def get_metadata(
    func_name: str, parameters: List[inspect.Parameter], skip_names: Sequence[str] = (), globalns: dict = {},
) -> FuncMetadata:
    dynamic_pydantic_model_params: dict[str, Any] = {}
    for param in parameters:
        if param.name.startswith("_"):
            raise InvalidSignature(
                f"Parameter {param.name} of {func_name} cannot start with '_'"
            )
        if param.name in skip_names:
            continue
        annotation = param.annotation

        # `x: None` / `x: None = None`
        if annotation is None:
            annotation = Annotated[
                None,
                Field(
                    default=param.default
                    if param.default is not inspect.Parameter.empty
                    else PydanticUndefined
                ),
            ]

        # Untyped field
        if annotation is inspect.Parameter.empty:
            annotation = Annotated[
                Any,
                Field(),
                # 🤷
                WithJsonSchema({"title": param.name, "type": "string"}),
            ]

        field_info = FieldInfo.from_annotated_attribute(
            _get_typed_annotation(annotation, globalns),
            param.default
            if param.default is not inspect.Parameter.empty
            else PydanticUndefined,
        )
        dynamic_pydantic_model_params[param.name] = (field_info.annotation, field_info)
        continue

    arguments_model = create_model(
        f"{func_name}Arguments",
        **dynamic_pydantic_model_params,
        __base__=ArgModelBase,
    )
    resp = FuncMetadata(arg_model=arguments_model)
    return resp
