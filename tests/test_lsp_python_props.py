"""
Tests for Python class property extraction in lsp_utils module.
"""

import ast
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from yellhorn_mcp.lsp_utils import _class_attributes_from_ast, extract_python_api


def test_class_attributes_from_ast():
    """Test extracting attributes from Python class AST."""
    # Class with typed and untyped attributes
    source = """
class User:
    name: str
    age: int = 0
    is_active = True
    _private = "hidden"
    __very_private = "very hidden"
    """

    tree = ast.parse(source)
    node = tree.body[0]  # First node is our class

    attrs = _class_attributes_from_ast(node)

    # Should include public attributes
    assert "name: str" in attrs
    assert "age: int" in attrs
    assert "is_active" in attrs

    # Should not include private attributes
    assert not any("_private" in attr for attr in attrs)
    assert not any("__very_private" in attr for attr in attrs)


def test_extract_python_api_with_class_attributes():
    """Test extracting Python API with class attributes."""
    with patch("builtins.open", MagicMock()) as mock_open:
        mock_file = MagicMock()
        mock_file.__enter__.return_value.read.return_value = """
class User:
    \"\"\"A user model.\"\"\"
    name: str
    age: int = 0
    is_active = True
    _private = "hidden"
    
    def get_display_name(self) -> str:
        \"\"\"Return user's display name.\"\"\"
        return self.name
        
    def _private_method(self):
        pass

from dataclasses import dataclass

@dataclass
class Product:
    \"\"\"A product model.\"\"\"
    id: int
    name: str
    price: float = 0.0
    _internal_code: str = "unknown"
    
    def get_price_with_tax(self, tax_rate: float) -> float:
        \"\"\"Calculate price with tax.\"\"\"
        return self.price * (1 + tax_rate)

# This simulates a Pydantic model
class Order:
    \"\"\"An order model.\"\"\"
    id: int
    products: list
    total: float
    _status: str = "pending"
    
    def calculate_total(self) -> float:
        \"\"\"Calculate order total.\"\"\"
        pass
"""
        mock_open.return_value = mock_file

        with patch("pathlib.Path.is_file", return_value=True):
            with patch("ast.unparse", return_value="str"):  # Mock unparse for Python < 3.9
                signatures = extract_python_api(Path("/mock/file.py"))

                # Check for class signatures with docstrings
                assert "class User  # A user model." in signatures
                assert "class Product  # A product model." in signatures
                assert "class Order  # An order model." in signatures

                # Check for attributes
                assert "    name: str" in signatures
                assert "    age: int" in signatures
                assert "    is_active" in signatures

                # Check for dataclass properties
                assert "    id: int" in signatures
                assert "    name: str" in signatures
                assert "    price: float" in signatures

                # Check for pydantic-style model properties
                assert "    id: int" in signatures
                assert "    products: list" in signatures
                assert "    total: float" in signatures

                # Check for methods with typed parameters and return types
                found_display_name = False
                found_price_with_tax = False
                found_calculate_total = False

                for sig in signatures:
                    if "def User.get_display_name" in sig:
                        found_display_name = True
                        # Just check for the presence of return type annotation
                        assert "->" in sig

                    if "def Product.get_price_with_tax" in sig:
                        found_price_with_tax = True
                        # Just check for the presence of type annotation, not the exact type
                        assert "tax_rate:" in sig
                        assert "->" in sig

                    if "def Order.calculate_total" in sig:
                        found_calculate_total = True
                        # Just check for the presence of return type annotation
                        assert "->" in sig

                assert found_display_name, "Could not find get_display_name method with return type"
                assert (
                    found_price_with_tax
                ), "Could not find get_price_with_tax method with parameter and return types"
                assert (
                    found_calculate_total
                ), "Could not find calculate_total method with return type"

                # Check for exclusion of private items
                assert not any("_private" in sig for sig in signatures)
                assert not any("_internal_code" in sig for sig in signatures)
                assert not any("_status" in sig for sig in signatures)
                assert not any("_private_method" in sig for sig in signatures)
