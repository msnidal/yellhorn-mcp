"""Tests for .yellhornignore functionality – created in workplan #40."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from yellhorn_mcp.server import get_codebase_snapshot


@pytest.mark.asyncio
async def test_yellhornignore_file_reading():
    """Test reading .yellhornignore file."""
    # Create a temporary directory with a .yellhornignore file
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create a .yellhornignore file with patterns
        yellhornignore_file = tmp_path / ".yellhornignore"
        yellhornignore_file.write_text(
            "# Comment line\n"
            "*.log\n"
            "node_modules/\n"
            "\n"  # Empty line should be skipped
            "dist/\n"
        )

        # Mock run_git_command to return a list of files
        with patch("yellhorn_mcp.server.run_git_command") as mock_git:
            mock_git.return_value = "\n".join(
                [
                    "file1.py",
                    "file2.js",
                    "file3.log",
                    "node_modules/package.json",
                    "dist/bundle.js",
                    "src/components/Button.js",
                ]
            )

            # Create a test file that can be read
            (tmp_path / "file1.py").write_text("# Test file 1")
            (tmp_path / "file2.js").write_text("// Test file 2")
            # Create directory structure for testing
            os.makedirs(tmp_path / "node_modules")
            os.makedirs(tmp_path / "dist")
            os.makedirs(tmp_path / "src/components")
            (tmp_path / "node_modules/package.json").write_text("{}")
            (tmp_path / "dist/bundle.js").write_text("/* bundle */")
            (tmp_path / "src/components/Button.js").write_text("// Button component")
            (tmp_path / "file3.log").write_text("log data")

            # Call get_codebase_snapshot
            file_paths, file_contents = await get_codebase_snapshot(tmp_path)

            # Verify that ignored files are not in results
            assert "file1.py" in file_paths
            assert "file2.js" in file_paths
            assert "src/components/Button.js" in file_paths
            assert "file3.log" not in file_paths  # Ignored by *.log
            assert "node_modules/package.json" not in file_paths  # Ignored by node_modules/
            assert "dist/bundle.js" not in file_paths  # Ignored by dist/

            # Verify contents
            assert "file1.py" in file_contents
            assert "file2.js" in file_contents
            assert "file3.log" not in file_contents
            assert "node_modules/package.json" not in file_contents
            assert "dist/bundle.js" not in file_contents


@pytest.mark.asyncio
async def test_yellhornignore_file_error_handling():
    """Test error handling when reading .yellhornignore file."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create a .yellhornignore file
        yellhornignore_path = tmp_path / ".yellhornignore"
        yellhornignore_path.write_text("*.log\nnode_modules/")

        # Mock run_git_command to return a list of files
        with patch("yellhorn_mcp.server.run_git_command") as mock_git:
            mock_git.return_value = "file1.py\nfile2.js\nfile3.log"

            # Mock open to raise an exception when reading .yellhornignore
            with patch("builtins.open") as mock_open:
                # Allow opening of files except .yellhornignore
                def side_effect(*args, **kwargs):
                    if str(args[0]).endswith(".yellhornignore"):
                        raise PermissionError("Permission denied")
                    # For other files, use the real open
                    return open(*args, **kwargs)

                mock_open.side_effect = side_effect

                # Create test files
                (tmp_path / "file1.py").write_text("# Test file 1")
                (tmp_path / "file2.js").write_text("// Test file 2")
                (tmp_path / "file3.log").write_text("log data")

                # Call get_codebase_snapshot
                file_paths, file_contents = await get_codebase_snapshot(tmp_path)

                # Since reading .yellhornignore failed, no files should be filtered
                assert len(file_paths) == 3
                assert "file1.py" in file_paths
                assert "file2.js" in file_paths
                assert (
                    "file3.log" in file_paths
                )  # Should not be filtered because .yellhornignore wasn't read


@pytest.mark.asyncio
async def test_get_codebase_snapshot_directory_handling():
    """Test handling of directories in get_codebase_snapshot."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create directory structure
        os.makedirs(tmp_path / "src")

        # Mock run_git_command to return file paths including a directory
        with patch("yellhorn_mcp.server.run_git_command") as mock_git:
            mock_git.return_value = "file1.py\nsrc"

            # Create test file
            (tmp_path / "file1.py").write_text("# Test file 1")

            # Create a mock implementation for Path.is_dir
            original_is_dir = Path.is_dir

            def mock_is_dir(self):
                # Check if the path ends with 'src'
                if str(self).endswith("/src"):
                    return True
                # Otherwise call the original
                return original_is_dir(self)

            # Apply the patch
            with patch.object(Path, "is_dir", mock_is_dir):
                # Make sure .yellhornignore doesn't exist
                with patch.object(Path, "exists", return_value=False):
                    # Call get_codebase_snapshot
                    file_paths, file_contents = await get_codebase_snapshot(tmp_path)

                    # Verify directory handling
                    assert len(file_paths) == 2
                    assert "file1.py" in file_paths
                    assert "src" in file_paths

                    # Only the file should be in contents, directories are skipped
                    assert len(file_contents) == 1
                    assert "file1.py" in file_contents
                    assert "src" not in file_contents


@pytest.mark.asyncio
async def test_get_codebase_snapshot_binary_file_handling():
    """Test handling of binary files in get_codebase_snapshot."""
    # Setup a temporary directory for testing
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create a text file and a binary file
        (tmp_path / "file1.py").write_text("# Test file 1")
        # Create binary-like content for file2.jpg
        with open(tmp_path / "file2.jpg", "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n")  # PNG file header

        # Mock run_git_command to return our test files
        with patch("yellhorn_mcp.server.run_git_command") as mock_git:
            mock_git.return_value = "file1.py\nfile2.jpg"

            # Make sure Path.is_dir returns False for our paths
            with patch.object(Path, "is_dir", return_value=False):
                # Make sure .yellhornignore doesn't exist
                with patch.object(Path, "exists", return_value=False):
                    # Mock open to raise UnicodeDecodeError for binary file
                    original_open = open

                    def mock_open(filename, *args, **kwargs):
                        if str(filename).endswith("file2.jpg") and "r" in args[0]:
                            raise UnicodeDecodeError("utf-8", b"\x80", 0, 1, "invalid start byte")
                        return original_open(filename, *args, **kwargs)

                    # Apply the patch to builtins.open
                    with patch("builtins.open", mock_open):
                        # Call get_codebase_snapshot
                        file_paths, file_contents = await get_codebase_snapshot(tmp_path)

                        # Verify binary file handling
                        assert len(file_paths) == 2
                        assert "file1.py" in file_paths
                        assert "file2.jpg" in file_paths

                        # Only the text file should be in contents
                        assert len(file_contents) == 1
                        assert "file1.py" in file_contents
                        assert "file2.jpg" not in file_contents
