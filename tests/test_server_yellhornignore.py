"""Tests for .yellhornignore and .yellhorncontext functionality."""

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
    """Test the curate_ignore_file tool functionality with parallel processing."""
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
        # Create a larger list of files to trigger parallel processing
        mock_sample_files = [
            "src/main.py",
            "src/utils.py",
            "node_modules/package1/index.js",
            "dist/bundle.js",
            "docs/README.md",
            "tests/test_main.py",
            # Adding more files to ensure we have multiple chunks
            *[f"src/module{i}.py" for i in range(20)],
            *[f"tests/test_module{i}.py" for i in range(20)],
        ]
        mock_snapshot.return_value = (mock_sample_files, {})
        
        # Mock asyncio.Semaphore to ensure we're limiting concurrency
        mock_semaphore = MagicMock()
        mock_semaphore_context = AsyncMock().__aenter__.return_value = None
        mock_semaphore.__aenter__ = AsyncMock(return_value=mock_semaphore_context)
        mock_semaphore.__aexit__ = AsyncMock(return_value=None)
        
        # Mock the Gemini API response - different responses for each chunk
        mock_responses = [
            MagicMock(text="""```ignorefile
# BLACKLIST PATTERNS
node_modules/
dist/
*.pyc

# WHITELIST PATTERNS
!docs/README.md
```"""),
            MagicMock(text="""```ignorefile
# BLACKLIST PATTERNS
__pycache__/
.DS_Store

# WHITELIST PATTERNS
!src/auth/
```"""),
            # Add more responses as needed for multiple chunks
        ]
        
        # Set up the async response for gemini client
        gemini_client_mock = mock_ctx.request_context.lifespan_context["gemini_client"]
        gemini_client_mock.aio = MagicMock()
        gemini_client_mock.aio.models = MagicMock()
        
        # Configure generate_content to return different responses for each call
        generate_content_mock = AsyncMock()
        # Return a different response each time it's called, or last response if we run out
        generate_content_mock.side_effect = lambda **kwargs: mock_responses.pop(0) if mock_responses else mock_responses[-1]
        gemini_client_mock.aio.models.generate_content = generate_content_mock
        
        # Create a patch for asyncio.Semaphore
        with patch("asyncio.Semaphore", return_value=mock_semaphore):
            # Also patch asyncio.create_task to track tasks
            with patch("asyncio.create_task") as mock_create_task:
                # Make mock_create_task pass through the coroutine
                mock_create_task.side_effect = lambda coro: coro
                
                # Patch asyncio.gather with an async mock that returns expected results
                async def mock_gather(*args, **kwargs):
                    return [
                        ({"node_modules/", "dist/", "*.pyc"}, {"!docs/README.md"}),
                        ({"__pycache__/", ".DS_Store"}, {"!src/auth/"}),
                    ]
                with patch("asyncio.gather", mock_gather):
                    
                    # Mock the open function to avoid writing to the filesystem
                    with patch("builtins.open", MagicMock()):
                        result = await curate_ignore_file(mock_ctx, user_task, "file_structure")
                        
                        # With an async function, we can't check .called attribute
                        # Instead, verify the result indicates successful processing
                        
                        # Verify the result message is correct
                        assert "Successfully created .yellhornignore file" in result
                        assert "5 blacklist and 2 whitelist patterns" in result  # Combined from both responses
                        
                        # Verify parallel processing logs were made
                        log_calls = [call[1]['message'] for call in mock_ctx.log.call_args_list if isinstance(call[1].get('message'), str)]
                        assert any("parallel processing" in msg for msg in log_calls)
                        assert any("parallel LLM tasks" in msg for msg in log_calls)
                        assert any("All chunks processed" in msg for msg in log_calls)
    
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
        
        # Mock asyncio functions for parallel processing
        with patch("asyncio.Semaphore"):
            with patch("asyncio.create_task", side_effect=lambda coro: coro):
                async def mock_gather_lsp(*args, **kwargs):
                    return [({"*.log", "node_modules/"}, {"!src/auth/jwt.py", "!src/models/user.py"})]
                with patch("asyncio.gather", mock_gather_lsp):
                    # Mock the open function to avoid writing to the filesystem
                    with patch("builtins.open", MagicMock()):
                        result = await curate_ignore_file(mock_ctx, user_task, "lsp")
                        
                        # Verify the result message is correct
                        assert "Successfully created .yellhornignore file" in result
                        assert "2 blacklist and 2 whitelist patterns" in result
            
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
        
        # Test parallel processing with errors
        with patch("asyncio.Semaphore"):
            with patch("asyncio.create_task", side_effect=lambda coro: coro):
                # Create a mock gather that returns an exception to simulate a task failure
                async def mock_gather_error(*args, **kwargs):
                    return [Exception("API Error")]
                with patch("asyncio.gather", mock_gather_error):
                    # Mock the open function to avoid writing to the filesystem
                    with patch("builtins.open", MagicMock()):
                        # Should not raise exception due to error handling in parallel processing
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
        
        # Mock asyncio functions for parallel processing
        with patch("asyncio.Semaphore"):
            with patch("asyncio.create_task", side_effect=lambda coro: coro):
                async def mock_gather_lsp(*args, **kwargs):
                    return [({"*.log", "node_modules/"}, {"!src/auth/jwt.py", "!src/models/user.py"})]
                with patch("asyncio.gather", mock_gather_lsp):
                    # Mock the open function to avoid writing to the filesystem
                    with patch("builtins.open", MagicMock()):
                        # Should work and default to full mode
                        result = await curate_ignore_file(mock_ctx, user_task, "invalid_mode")
                        assert "Successfully created .yellhornignore file" in result

    # Test depth_limit parameter
    mock_ctx = MagicMock()
    mock_ctx.log = AsyncMock()
    mock_ctx.request_context.lifespan_context = {
        "repo_path": Path("/fake/repo/path"),
        "model": "gemini-2.5-pro-preview-03-25",
        "gemini_client": MagicMock(),
    }
    
    with patch("yellhorn_mcp.server.get_codebase_snapshot") as mock_snapshot:
        # Create a sample file list with different depths
        mock_sample_files = [
            "root_file.py",                  # depth 1
            "first_level/file.py",           # depth 2
            "first_level/second_level/file.py",  # depth 3
            "first_level/second_level/third_level/file.py",  # depth 4
            "another_dir/file.py",           # depth 2
            "another_dir/subdir/file.py",    # depth 3
        ]
        mock_snapshot.return_value = (mock_sample_files, {})
        
        # Set up mock for gemini client
        gemini_client_mock = mock_ctx.request_context.lifespan_context["gemini_client"]
        gemini_client_mock.aio = MagicMock()
        gemini_client_mock.aio.models = MagicMock()
        gemini_client_mock.aio.models.generate_content = AsyncMock(return_value=MagicMock(text="```ignorefile\n# BLACKLIST PATTERNS\n*.log\n\n# WHITELIST PATTERNS\n!important.py\n```"))
        
        # Test with explicit depth_limit = 1 (only root files)
        with patch("asyncio.Semaphore"):
            with patch("asyncio.create_task", side_effect=lambda coro: coro):
                async def mock_gather_depth(*args, **kwargs):
                    return [({"*.log"}, {"!important.py"})]
                with patch("asyncio.gather", mock_gather_depth):
                    with patch("builtins.open", MagicMock()):
                        result = await curate_ignore_file(mock_ctx, user_task, "file_structure", depth_limit=1)
                        assert "Successfully created .yellhornignore file" in result
                        
                        # Verify depth filtering was logged
                        log_messages = [call[1]['message'] for call in mock_ctx.log.call_args_list 
                                      if isinstance(call[1].get('message'), str)]
                        # Look for messages that mention depth limit 1 and filtering
                        depth_logs = [msg for msg in log_messages if "depth limit 1" in msg.lower()]
                        assert any("filtered from" in msg for msg in depth_logs)
                        
        # Test with depth_limit = 2 (root files and first level directories)
        mock_ctx.log.reset_mock()  # Reset log calls
        with patch("asyncio.Semaphore"):
            with patch("asyncio.create_task", side_effect=lambda coro: coro):
                async def mock_gather_depth(*args, **kwargs):
                    return [({"*.log"}, {"!important.py"})]
                with patch("asyncio.gather", mock_gather_depth):
                    with patch("builtins.open", MagicMock()):
                        result = await curate_ignore_file(mock_ctx, user_task, "file_structure", depth_limit=2)
                        assert "Successfully created .yellhornignore file" in result
                        
                        # Verify depth filtering was logged with correct counts in the message
                        log_messages = [call[1]['message'] for call in mock_ctx.log.call_args_list 
                                     if isinstance(call[1].get('message'), str)]
                        # Check for depth limit information in log messages
                        depth_logs = [msg for msg in log_messages if "depth limit 2" in msg.lower()]
                        assert any(depth_logs), "No log message found containing 'depth limit 2'"
                        
        # Test automatic depth limit for file_structure mode
        mock_ctx.log.reset_mock()  # Reset log calls
        with patch("asyncio.Semaphore"):
            with patch("asyncio.create_task", side_effect=lambda coro: coro):
                async def mock_gather_depth(*args, **kwargs):
                    return [({"*.log"}, {"!important.py"})]
                with patch("asyncio.gather", mock_gather_depth):
                    with patch("builtins.open", MagicMock()):
                        # Don't specify depth_limit - should default to 1 for file_structure mode
                        result = await curate_ignore_file(mock_ctx, user_task, "file_structure")
                        assert "Successfully created .yellhornignore file" in result
                        
                        # Verify automatic depth limit was set and logged
                        log_messages = [call[1]['message'] for call in mock_ctx.log.call_args_list 
                                      if isinstance(call[1].get('message'), str)]
                        # Check for the specific message about setting default depth limit
                        assert any("Setting depth_limit to 2" in msg for msg in log_messages)


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
    from yellhorn_mcp.server import curate_context, YellhornMCPError
    
    # Create a mock context with async log method
    mock_ctx = MagicMock()
    mock_ctx.log = AsyncMock()
    mock_ctx.request_context.lifespan_context = {
        "repo_path": Path("/fake/repo/path"),
        "model": "gemini-2.5-pro-preview-03-25",
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
                gemini_client_mock.aio.models.generate_content = AsyncMock(return_value=mock_response)
                
                # Call curate_context
                result = await curate_context(mock_ctx, user_task)
                
                # Verify the result
                assert "Successfully created .yellhorncontext file" in result
                assert "4 important directories" in result
                assert "recommended blacklist patterns" in result
                
                # Verify that correct log messages were created
                log_calls = [call[1]['message'] for call in mock_ctx.log.call_args_list if isinstance(call[1].get('message'), str)]
                assert any("No .yellhornignore file found" in msg for msg in log_calls)
                assert any("Processing complete, identified 4 important directories" in msg for msg in log_calls)
                assert any("Using Git's tracking information - respecting .gitignore patterns" in msg for msg in log_calls)
    
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
        with patch("pathlib.Path.exists", return_value=True), patch("pathlib.Path.is_file", return_value=True):
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
                gemini_client_mock.aio.models.generate_content = AsyncMock(return_value=mock_response)
                
                # Call curate_context with .yellhornignore
                result = await curate_context(mock_ctx, user_task)
                
                # Verify the result
                assert "Successfully created .yellhorncontext file" in result
                assert "5 important directories" in result
                assert "existing ignore patterns from .yellhornignore" in result
                
                # Verify that correct log messages were created
                log_calls = [call[1]['message'] for call in mock_ctx.log.call_args_list if isinstance(call[1].get('message'), str)]
                assert any("Found .yellhornignore file" in msg for msg in log_calls)
                assert any("Applied .yellhornignore filtering" in msg for msg in log_calls)
                assert any("identified 5 important directories" in msg for msg in log_calls)
    
    # Test with depth_limit parameter
    mock_ctx.reset_mock()
    with patch("yellhorn_mcp.server.get_codebase_snapshot") as mock_snapshot:
        # Create a list of files with various depths
        mock_sample_files = [
            "root_file.py",                    # depth 1
            "first_level/file.py",             # depth 2
            "first_level/second_level/file.py", # depth 3
            "deep/path/to/file.py",            # depth 4
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
                gemini_client_mock.aio.models.generate_content = AsyncMock(return_value=mock_response)
                
                # Call curate_context with depth_limit=2
                result = await curate_context(mock_ctx, user_task, depth_limit=2)
                
                # Verify that depth filtering was applied
                log_calls = [call[1]['message'] for call in mock_ctx.log.call_args_list if isinstance(call[1].get('message'), str)]
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
                gemini_client_mock.aio.models.generate_content = AsyncMock(side_effect=Exception("API Error"))
                
                # Test we handle errors and use all directories as fallback
                result = await curate_context(mock_ctx, user_task)
                
                # Verify the result shows we included all directories as fallback
                assert "Successfully created .yellhorncontext file" in result
                
                # Verify that we logged the error and fallback behavior
                log_calls = [call[1]['message'] for call in mock_ctx.log.call_args_list if isinstance(call[1].get('message'), str)]
                assert any("Error processing chunk" in msg for msg in log_calls)
                assert any("No important directories identified, including all directories" in msg for msg in log_calls)
    
    # Test with OpenAI model
    mock_ctx.reset_mock()
    mock_ctx.request_context.lifespan_context = {
        "repo_path": Path("/fake/repo/path"),
        "model": "gpt-4o", # Use an OpenAI model
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
                mock_response.choices[0].message.content = """```context
src
```"""
                
                # Mock the create function
                openai_client_mock.chat.completions.create = AsyncMock(return_value=mock_response)
                
                # Call curate_context with OpenAI model
                result = await curate_context(mock_ctx, user_task)
                
                # Verify the result shows successful creation
                assert "Successfully created .yellhorncontext file" in result
                
                # Verify that we made a call to OpenAI
                log_calls = [call[1]['message'] for call in mock_ctx.log.call_args_list if isinstance(call[1].get('message'), str)]
                assert any("gpt-4o" in msg for msg in log_calls)
