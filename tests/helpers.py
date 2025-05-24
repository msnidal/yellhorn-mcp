"""Helper utilities for test fixtures."""

import contextlib
import os
from unittest.mock import patch


class DummyContext:
    """A simple context object for testing functions that expect a context parameter."""

    def __init__(self, **kwargs):
        """
        Initialize with any attributes needed.

        Args:
            **kwargs: Attributes to set on the context object
        """
        self.__dict__.update(kwargs)

        # Set up a request_context with lifespan_context for improved compatibility
        # This is needed for the codebase_reasoning feature tests
        class DummyRequestContext:
            def __init__(self):
                self.lifespan_context = {"codebase_reasoning": "full"}

        self.request_context = DummyRequestContext()

    async def log(self, level=None, message=None):
        """
        Mock implementation of the log method.

        Args:
            level: Log level
            message: Log message
        """
        # Just silently succeed
        pass


@contextlib.contextmanager
def patch_env(env_vars, clear=False):
    """
    Context manager for patching environment variables.

    Args:
        env_vars: Dictionary of environment variables to set
        clear: Whether to clear all existing environment variables first

    Yields:
        None
    """
    with patch.dict(os.environ, env_vars, clear=clear):
        yield
