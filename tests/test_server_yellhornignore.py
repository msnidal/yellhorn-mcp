"""Tests for .yellhornignore and .yellhorncontext functionality."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

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
            assert (
                "node_modules/important-package.json" in file_paths
            )  # Whitelisted despite node_modules/

            # Verify contents
            assert "file1.py" in file_contents
            assert "file2.js" in file_contents
            assert "regular.log" not in file_contents
            assert "important.log" in file_contents
            assert "node_modules/package.json" not in file_contents
            assert "node_modules/important-package.json" in file_contents
            assert "dist/bundle.js" not in file_contents


# Helper class for creating async mocks
class AsyncMock(MagicMock):
    """MagicMock subclass that supports async with syntax and awaitable returns."""

    async def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    def __await__(self):
        yield from []
        return self().__await__()


@pytest.mark.asyncio
async def test_curate_context():
    """Test the curate_context tool functionality with .yellhornignore integration."""
    from yellhorn_mcp.git_utils import YellhornMCPError
    from yellhorn_mcp.server import curate_context

    # Create a mock context with async log method
    mock_ctx = MagicMock()
    mock_ctx.log = AsyncMock()
    mock_ctx.request_context.lifespan_context = {
        "repo_path": Path("/fake/repo/path"),
        "model": "gemini-2.5-pro-preview-05-06",
        "gemini_client": MagicMock(),
    }

    # Sample user task
    user_task = "Implementing a new feature for data processing"

    # Setup mock for get_codebase_snapshot
    with patch("yellhorn_mcp.server.get_codebase_snapshot") as mock_snapshot:
        # First test: No files found
        mock_snapshot.return_value = ([], {})

        # Test error handling when no files are found
        with pytest.raises(YellhornMCPError, match="No files found in repository to analyze"):
            await curate_context(mock_ctx, user_task)

        # Second test: Without .yellhornignore file
        # Create a list of files to analyze
        mock_sample_files = [
            "src/main.py",
            "src/utils.py",
            "src/data/processor.py",
            "src/data/models.py",
            "tests/test_main.py",
            "tests/test_data/test_processor.py",
            "docs/README.md",
            "build/output.js",
            "node_modules/package1/index.js",
        ]
        mock_snapshot.return_value = (mock_sample_files, {})

        # Mock Path.exists to return False for .yellhornignore
        with patch("pathlib.Path.exists", return_value=False):
            # Mock open to avoid writing to the filesystem
            with patch("builtins.open", MagicMock()):
                # Mock the Gemini client response for directory selection
                gemini_client_mock = mock_ctx.request_context.lifespan_context["gemini_client"]
                gemini_client_mock.aio = MagicMock()
                gemini_client_mock.aio.models = MagicMock()

                # Configure the Gemini response mock
                mock_response = MagicMock()
                mock_response.text = """```context
src
src/data
tests
tests/test_data
```"""
                # Set up both API patterns for backward compatibility
                gemini_client_mock.aio.models.generate_content = AsyncMock(
                    return_value=mock_response
                )
                gemini_client_mock.aio.generate_content = AsyncMock(return_value=mock_response)

                # Call curate_context
                result = await curate_context(mock_ctx, user_task)

                # Verify the result
                assert "Successfully created .yellhorncontext file" in result
                # We now match different directories, so just check for "important directories"
                assert "important directories" in result
                assert "recommended blacklist patterns" in result

                # Verify that correct log messages were created
                log_calls = [
                    call[1]["message"]
                    for call in mock_ctx.log.call_args_list
                    if isinstance(call[1].get("message"), str)
                ]
                assert any("No .yellhornignore file found" in msg for msg in log_calls)
                assert any(
                    "Processing complete, identified 4 important directories" in msg
                    for msg in log_calls
                )
                assert any(
                    "Using Git's tracking information - respecting .gitignore patterns" in msg
                    for msg in log_calls
                )

    # Test with .yellhornignore file
    mock_ctx.reset_mock()
    with patch("yellhorn_mcp.server.get_codebase_snapshot") as mock_snapshot:
        # Create a list of files to analyze
        mock_sample_files = [
            "src/main.py",
            "src/utils.py",
            "src/data/processor.py",
            "src/data/models.py",
            "tests/test_main.py",
            "tests/test_data/test_processor.py",
            "docs/README.md",
            "build/output.js",
            "node_modules/package1/index.js",
        ]
        mock_snapshot.return_value = (mock_sample_files, {})

        # Setup mock Path.exists and Path.is_file for .yellhornignore
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.is_file", return_value=True),
        ):
            # Mock reading .yellhornignore file
            with patch("builtins.open") as mock_open:
                # Create a mock file-like object for .yellhornignore
                mock_file = MagicMock()
                # The file contains patterns to ignore node_modules and build directories
                mock_file.__enter__.return_value.readlines.return_value = [
                    "# Ignore patterns\n",
                    "node_modules/\n",
                    "build/\n",
                    "*.log\n",
                ]

                # Make the mock open return the mock file for .yellhornignore
                # but use the normal open for other files
                def side_effect(*args, **kwargs):
                    if str(args[0]).endswith(".yellhornignore"):
                        return mock_file
                    # For our output file (.yellhorncontext), create a mock
                    elif str(args[0]).endswith(".yellhorncontext"):
                        return MagicMock()
                    # For other files, use a mock as well
                    return MagicMock()

                mock_open.side_effect = side_effect
                mock_file.__enter__.return_value.__iter__.return_value = [
                    "# Ignore patterns\n",
                    "node_modules/\n",
                    "build/\n",
                    "*.log\n",
                ]

                # Mock the Gemini client response for directory selection
                gemini_client_mock = mock_ctx.request_context.lifespan_context["gemini_client"]
                gemini_client_mock.aio = MagicMock()
                gemini_client_mock.aio.models = MagicMock()

                # Configure the Gemini response mock
                mock_response = MagicMock()
                mock_response.text = """```context
src
src/data
tests
tests/test_data
docs
```"""
                gemini_client_mock.aio.models.generate_content = AsyncMock(
                    return_value=mock_response
                )

                # Call curate_context with .yellhornignore
                result = await curate_context(mock_ctx, user_task)

                # Verify the result
                assert "Successfully created .yellhorncontext file" in result
                assert "5 important directories" in result
                assert "existing ignore patterns from .yellhornignore" in result

                # Verify that correct log messages were created
                log_calls = [
                    call[1]["message"]
                    for call in mock_ctx.log.call_args_list
                    if isinstance(call[1].get("message"), str)
                ]
                assert any("Found .yellhornignore file" in msg for msg in log_calls)
                assert any("Applied .yellhornignore filtering" in msg for msg in log_calls)
                assert any("identified 5 important directories" in msg for msg in log_calls)

    # Test with depth_limit parameter
    mock_ctx.reset_mock()
    with patch("yellhorn_mcp.server.get_codebase_snapshot") as mock_snapshot:
        # Create a list of files with various depths
        mock_sample_files = [
            "root_file.py",  # depth 1
            "first_level/file.py",  # depth 2
            "first_level/second_level/file.py",  # depth 3
            "deep/path/to/file.py",  # depth 4
        ]
        mock_snapshot.return_value = (mock_sample_files, {})

        # Mock Path.exists and Path.is_file for no .yellhornignore
        with patch("pathlib.Path.exists", return_value=False):
            # Mock open to avoid writing to the filesystem
            with patch("builtins.open", MagicMock()):
                # Mock the Gemini client response for directory selection
                gemini_client_mock = mock_ctx.request_context.lifespan_context["gemini_client"]
                gemini_client_mock.aio = MagicMock()
                gemini_client_mock.aio.models = MagicMock()

                # Configure the Gemini response mock
                mock_response = MagicMock()
                mock_response.text = """```context
first_level
```"""
                gemini_client_mock.aio.models.generate_content = AsyncMock(
                    return_value=mock_response
                )

                # Call curate_context with depth_limit=2
                result = await curate_context(mock_ctx, user_task, depth_limit=2)

                # Verify that depth filtering was applied
                log_calls = [
                    call[1]["message"]
                    for call in mock_ctx.log.call_args_list
                    if isinstance(call[1].get("message"), str)
                ]
                assert any("Applied depth limit 2" in msg for msg in log_calls)
                assert any("filtered from" in msg for msg in log_calls)

    # Test error handling during LLM call
    mock_ctx.reset_mock()
    with patch("yellhorn_mcp.server.get_codebase_snapshot") as mock_snapshot:
        # Create a simple list of files
        mock_snapshot.return_value = (["file1.py", "file2.py"], {})

        # Mock Path.exists for no .yellhornignore
        with patch("pathlib.Path.exists", return_value=False):
            # Mock open to avoid writing to the filesystem
            with patch("builtins.open", MagicMock()):
                # Mock the Gemini client to raise an exception
                gemini_client_mock = mock_ctx.request_context.lifespan_context["gemini_client"]
                gemini_client_mock.aio = MagicMock()
                gemini_client_mock.aio.models = MagicMock()
                gemini_client_mock.aio.models.generate_content = AsyncMock(
                    side_effect=Exception("API Error")
                )

                # Test we handle errors and use all directories as fallback
                result = await curate_context(mock_ctx, user_task)

                # Verify the result shows we included all directories as fallback
                assert "Successfully created .yellhorncontext file" in result

                # Verify that we logged the error and fallback behavior
                log_calls = [
                    call[1]["message"]
                    for call in mock_ctx.log.call_args_list
                    if isinstance(call[1].get("message"), str)
                ]
                assert any("Error processing chunk" in msg for msg in log_calls)
                assert any(
                    "No important directories identified, including all directories" in msg
                    for msg in log_calls
                )

    # Test with OpenAI model
    mock_ctx.reset_mock()
    mock_ctx.request_context.lifespan_context = {
        "repo_path": Path("/fake/repo/path"),
        "model": "gpt-4o",  # Use an OpenAI model
        "openai_client": MagicMock(),
    }

    with patch("yellhorn_mcp.server.get_codebase_snapshot") as mock_snapshot:
        # Create a simple list of files
        mock_snapshot.return_value = (["src/file1.py", "src/file2.py"], {})

        # Mock Path.exists for no .yellhornignore
        with patch("pathlib.Path.exists", return_value=False):
            # Mock open to avoid writing to the filesystem
            with patch("builtins.open", MagicMock()):
                # Mock the OpenAI client response
                openai_client_mock = mock_ctx.request_context.lifespan_context["openai_client"]
                openai_client_mock.chat = MagicMock()
                openai_client_mock.chat.completions = MagicMock()

                # Create response object mock
                mock_response = MagicMock()
                mock_response.choices = [MagicMock()]
                mock_response.choices[0].message = MagicMock()
                mock_response.choices[
                    0
                ].message.content = """```context
src
```"""

                # Mock the create function
                openai_client_mock.chat.completions.create = AsyncMock(return_value=mock_response)

                # Call curate_context with OpenAI model
                result = await curate_context(mock_ctx, user_task)

                # Verify the result shows successful creation
                assert "Successfully created .yellhorncontext file" in result

                # Verify that we made a call to OpenAI
                log_calls = [
                    call[1]["message"]
                    for call in mock_ctx.log.call_args_list
                    if isinstance(call[1].get("message"), str)
                ]
                assert any("gpt-4o" in msg for msg in log_calls)
