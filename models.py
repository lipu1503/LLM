from pydantic import BaseModel, Field


class ComponentBundle(BaseModel):
    """The three artifacts we ask the model to produce."""

    component_name: str = Field(..., description="PascalCase name of the component")
    component_code: str = Field(..., description="Full .jsx/.tsx source of the React component")
    css_code: str = Field(..., description="Full .css source for the component")
    test_code: str = Field(..., description="Full unit test file (Jest + React Testing Library)")


class GenerateComponentResponse(BaseModel):
    bundle: ComponentBundle
    raw_model: str
    usage: dict | None = None


class SavedFiles(BaseModel):
    """Returned when the generated files are written to disk instead of
    (or in addition to) being returned inline."""

    component_name: str
    component_path: str = Field(..., description="Absolute path to the saved .jsx file")
    css_path: str = Field(..., description="Absolute path to the saved .css file")
    test_path: str = Field(..., description="Absolute path to the saved .test.jsx file")
    usage: dict | None = None