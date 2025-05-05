"""Tests for .yellhornignore functionality – created in workplan #40."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from yellhorn_mcp.server import get_codebase_snapshot, parse_ignore_patterns


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


@pytest.mark.asyncio
async def test_yellhornignore_whitelist_functionality():
    """Test whitelisting files with ! prefix in .yellhornignore file."""
    # Create a temporary directory with a .yellhornignore file
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create a .yellhornignore file with patterns and whitelist
        yellhornignore_file = tmp_path / ".yellhornignore"
        yellhornignore_file.write_text(
            "# Comment line\n"
            "*.log\n"
            "node_modules/\n"
            "dist/\n"
            "# Whitelist specific files\n"
            "!important.log\n"
            "!node_modules/important-package.json\n"
        )

        # Mock run_git_command to return a list of files
        with patch("yellhorn_mcp.server.run_git_command") as mock_git:
            mock_git.return_value = "\n".join(
                [
                    "file1.py",
                    "file2.js",
                    "regular.log",
                    "important.log",
                    "node_modules/package.json",
                    "node_modules/important-package.json",
                    "dist/bundle.js",
                    "src/components/Button.js",
                ]
            )

            # Create files for testing
            (tmp_path / "file1.py").write_text("# Test file 1")
            (tmp_path / "file2.js").write_text("// Test file 2")
            os.makedirs(tmp_path / "node_modules")
            os.makedirs(tmp_path / "dist")
            os.makedirs(tmp_path / "src/components")
            (tmp_path / "regular.log").write_text("regular log data")
            (tmp_path / "important.log").write_text("important log data")
            (tmp_path / "node_modules/package.json").write_text("{}")
            (tmp_path / "node_modules/important-package.json").write_text('{"name": "important"}')
            (tmp_path / "dist/bundle.js").write_text("/* bundle */")
            (tmp_path / "src/components/Button.js").write_text("// Button component")

            # Call get_codebase_snapshot
            file_paths, file_contents = await get_codebase_snapshot(tmp_path)

            # Verify that ignored files are not in results
            assert "file1.py" in file_paths
            assert "file2.js" in file_paths
            assert "src/components/Button.js" in file_paths
            
            # Verify that regular ignored files are not included
            assert "regular.log" not in file_paths  # Ignored by *.log
            assert "node_modules/package.json" not in file_paths  # Ignored by node_modules/
            assert "dist/bundle.js" not in file_paths  # Ignored by dist/
            
            # Verify whitelisted files are included despite matching ignore patterns
            assert "important.log" in file_paths  # Whitelisted despite *.log
            assert "node_modules/important-package.json" in file_paths  # Whitelisted despite node_modules/

            # Verify contents
            assert "file1.py" in file_contents
            assert "file2.js" in file_contents
            assert "regular.log" not in file_contents
            assert "important.log" in file_contents
            assert "node_modules/package.json" not in file_contents
            assert "node_modules/important-package.json" in file_contents
            assert "dist/bundle.js" not in file_contents


def test_parse_ignore_patterns():
    """Test the parse_ignore_patterns function that extracts blacklist and whitelist patterns."""
    # Basic test with standard format
    result = """```ignorefile
# BLACKLIST PATTERNS
*.log
node_modules/
dist/
__pycache__/

# WHITELIST PATTERNS
!important.log
!node_modules/config.json
```"""
    ignore_patterns, whitelist_patterns = parse_ignore_patterns(result)
    
    assert "*.log" in ignore_patterns
    assert "node_modules/" in ignore_patterns
    assert "dist/" in ignore_patterns
    assert "__pycache__/" in ignore_patterns
    assert len(ignore_patterns) == 4
    
    assert "!important.log" in whitelist_patterns
    assert "!node_modules/config.json" in whitelist_patterns
    assert len(whitelist_patterns) == 2
    
    # Test with non-standard format (without code blocks)
    result = """
# BLACKLIST PATTERNS
*.log
node_modules/

# WHITELIST PATTERNS
!important.log
"""
    ignore_patterns, whitelist_patterns = parse_ignore_patterns(result)
    
    assert "*.log" in ignore_patterns
    assert "node_modules/" in ignore_patterns
    assert len(ignore_patterns) == 2
    
    assert "!important.log" in whitelist_patterns
    assert len(whitelist_patterns) == 1
    
    # Test with empty result
    result = ""
    ignore_patterns, whitelist_patterns = parse_ignore_patterns(result)
    
    assert len(ignore_patterns) == 0
    assert len(whitelist_patterns) == 0


@pytest.mark.asyncio
async def test_curate_ignore_file():
    """Test the curate_ignore_file tool functionality."""
    from yellhorn_mcp.server import curate_ignore_file, YellhornMCPError
    
    # Create a mock context with async log method
    mock_ctx = MagicMock()
    mock_ctx.log = AsyncMock()
    mock_ctx.request_context.lifespan_context = {
        "repo_path": Path("/fake/repo/path"),
        "model": "gemini-2.5-pro-preview-03-25",
        "gemini_client": MagicMock(),
    }
    
    # Sample user task
    user_task = "Implementing a user authentication system with JWT tokens"
    
    # Setup mock for get_codebase_snapshot
    with patch("yellhorn_mcp.server.get_codebase_snapshot") as mock_snapshot:
        # First test: No files found
        mock_snapshot.return_value = ([], {})
        
        # Test error handling when no files are found
        with pytest.raises(YellhornMCPError, match="No files found in repository to analyze"):
            await curate_ignore_file(mock_ctx, user_task)
        
        # Second test: Normal operation with files using file_structure mode
        mock_sample_files = [
            "src/main.py",
            "src/utils.py",
            "node_modules/package1/index.js",
            "dist/bundle.js",
            "docs/README.md",
            "tests/test_main.py",
        ]
        mock_snapshot.return_value = (mock_sample_files, {})
        
        # Mock the Gemini API response
        mock_response = MagicMock()
        mock_response.text = """```ignorefile
# BLACKLIST PATTERNS
node_modules/
dist/
*.pyc
__pycache__/

# WHITELIST PATTERNS
!docs/README.md
!src/auth/
```"""
        # Set up the async response for gemini client
        gemini_client_mock = mock_ctx.request_context.lifespan_context["gemini_client"]
        gemini_client_mock.aio = MagicMock()
        gemini_client_mock.aio.models = MagicMock()
        gemini_client_mock.aio.models.generate_content = AsyncMock(return_value=mock_response)
        
        # Mock the open function to avoid writing to the filesystem
        with patch("builtins.open", MagicMock()):
            result = await curate_ignore_file(mock_ctx, user_task, "file_structure")
            
            # Verify the result message is correct
            assert "Successfully created .yellhornignore file" in result
            assert "4 blacklist and 2 whitelist patterns" in result
            
            # Verify the API was called
            assert gemini_client_mock.aio.models.generate_content.called
    
    # Test LSP mode
    with patch("yellhorn_mcp.lsp_utils.get_lsp_snapshot") as mock_lsp_snapshot:
        # Setup mock for LSP mode
        mock_lsp_files = ["src/auth/jwt.py", "src/models/user.py"]
        mock_lsp_contents = {
            "src/auth/jwt.py": "def generate_token(user_id: str) -> str\ndef verify_token(token: str) -> dict",
            "src/models/user.py": "class User\n    username: str\n    password_hash: str\n    def authenticate(self, password: str) -> bool"
        }
        mock_lsp_snapshot.return_value = (mock_lsp_files, mock_lsp_contents)
        
        # Set up a fresh mock context
        mock_ctx = MagicMock()
        mock_ctx.log = AsyncMock()
        mock_ctx.request_context.lifespan_context = {
            "repo_path": Path("/fake/repo/path"),
            "model": "gemini-2.5-pro-preview-03-25",
            "gemini_client": MagicMock(),
        }
        
        # Mock the Gemini API response
        mock_response = MagicMock()
        mock_response.text = """```ignorefile
# BLACKLIST PATTERNS
*.log
node_modules/

# WHITELIST PATTERNS
!src/auth/jwt.py
!src/models/user.py
```"""
        # Set up the async response for gemini client
        gemini_client_mock = mock_ctx.request_context.lifespan_context["gemini_client"]
        gemini_client_mock.aio = MagicMock()
        gemini_client_mock.aio.models = MagicMock()
        gemini_client_mock.aio.models.generate_content = AsyncMock(return_value=mock_response)
        
        # Mock the open function to avoid writing to the filesystem
        with patch("builtins.open", MagicMock()):
            result = await curate_ignore_file(mock_ctx, user_task, "lsp")
            
            # Verify the result message is correct
            assert "Successfully created .yellhornignore file" in result
            assert "2 blacklist and 2 whitelist patterns" in result
            
            # Verify the API was called
            assert gemini_client_mock.aio.models.generate_content.called
            
    # Test error handling for API errors
    mock_ctx = MagicMock()
    mock_ctx.log = AsyncMock()
    mock_ctx.request_context.lifespan_context = {
        "repo_path": Path("/fake/repo/path"),
        "model": "gemini-2.5-pro-preview-03-25",
        "gemini_client": MagicMock(),
    }
    
    # Configure gemini client to raise an exception
    gemini_client_mock = mock_ctx.request_context.lifespan_context["gemini_client"]
    gemini_client_mock.aio = MagicMock()
    gemini_client_mock.aio.models = MagicMock()
    gemini_client_mock.aio.models.generate_content = AsyncMock(side_effect=Exception("API Error"))
    
    with patch("yellhorn_mcp.server.get_codebase_snapshot") as mock_snapshot:
        mock_snapshot.return_value = (["file1.py"], {})
        
        # Test that we still continue processing after chunk errors
        with patch("builtins.open", MagicMock()):
            # Should not raise exception due to try/except that catches chunk errors
            result = await curate_ignore_file(mock_ctx, user_task, "full")
            assert "Successfully created .yellhornignore file" in result
            assert "0 blacklist and 0 whitelist patterns" in result  # No patterns due to API error
    
    # Test invalid codebase_reasoning mode (should default to full)
    mock_ctx = MagicMock()
    mock_ctx.log = AsyncMock()
    mock_ctx.request_context.lifespan_context = {
        "repo_path": Path("/fake/repo/path"),
        "model": "gemini-2.5-pro-preview-03-25",
        "gemini_client": MagicMock(),
    }
    
    with patch("yellhorn_mcp.server.get_codebase_snapshot") as mock_snapshot:
        mock_snapshot.return_value = (["file1.py"], {})
        
        gemini_client_mock = mock_ctx.request_context.lifespan_context["gemini_client"]
        gemini_client_mock.aio = MagicMock()
        gemini_client_mock.aio.models = MagicMock()
        gemini_client_mock.aio.models.generate_content = AsyncMock(return_value=mock_response)
        
        # Mock the open function to avoid writing to the filesystem
        with patch("builtins.open", MagicMock()):
            # Should work and default to full mode
            result = await curate_ignore_file(mock_ctx, user_task, "invalid_mode")
            assert "Successfully created .yellhornignore file" in result


# Helper class for creating async mocks
class AsyncMock(MagicMock):
    """MagicMock subclass that supports async with syntax and awaitable returns."""
    async def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)
    
    def __await__(self):
        yield from []
        return self().__await__()
