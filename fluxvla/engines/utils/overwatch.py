# Origin: Modified from
# Upstream-Repo: openvla/openvla
# Upstream-Path: prismatic/overwatch/overwatch.py
# Upstream-Ref: main
# SPDX-License-Identifier: MIT
# Notes: Attribution normalized; no functional change.
"""
overwatch.py

This module provides utility classes for centralized logging with Rich
formatting and optional integration with distributed training via the
`accelerate` library.

Two main logger classes are provided:
- `DistributedOverwatch`: integrates with `accelerate.PartialState` for
  multi-process awareness.
- `PureOverwatch`: provides standard logging when distributed context is
  not needed.

A convenience factory `initialize_overwatch` is included to automatically
choose based on environment variable `WORLD_SIZE`.
"""

import logging
import logging.config
import os
from contextlib import nullcontext
from logging import LoggerAdapter
from typing import Any, Callable, ClassVar, Dict, MutableMapping, Tuple, Union

# Overwatch Default Format String
RICH_FORMATTER, DATEFMT = '| >> %(message)s', '%m/%d [%H:%M:%S]'

# Set Logging Configuration
LOG_CONFIG = {
    'version': 1,
    'disable_existing_loggers': True,
    'formatters': {
        'simple-console': {
            'format': RICH_FORMATTER,
            'datefmt': DATEFMT
        }
    },
    'handlers': {
        'console': {
            'class': 'rich.logging.RichHandler',
            'formatter': 'simple-console',
            'markup': True,
            'rich_tracebacks': True,
            'show_level': True,
            'show_path': True,
            'show_time': True,
        }
    },
    'root': {
        'level': 'INFO',
        'handlers': ['console']
    },
}
logging.config.dictConfig(LOG_CONFIG)


class ContextAdapter(LoggerAdapter):
    """
    A custom LoggerAdapter to add hierarchical context prefixes to log
    messages, improving the readability and structure of nested logging flows.
    """

    CTX_PREFIXES: ClassVar[Dict[int, str]] = {
        **{
            0: '[*] '
        },
        **{idx: '|=> '.rjust(4 + (idx * 4))
           for idx in [1, 2, 3]}
    }

    def process(
        self, msg: str,
        kwargs: MutableMapping[str,
                               Any]) -> Tuple[str, MutableMapping[str, Any]]:
        """
        Injects context level prefixes into log messages.

        Args:
            msg (str): Original log message.
            kwargs (dict): Keyword arguments for logging call (may include
                           `ctx_level`).

        Returns:
            Tuple[str, dict]: Modified log message with context prefix and
                              remaining kwargs.
        """
        ctx_level = kwargs.pop('ctx_level', 0)
        return f'{self.CTX_PREFIXES[ctx_level]}{msg}', kwargs


class DistributedOverwatch:
    """
    Logging utility with support for distributed training contexts via
    `accelerate.PartialState`. Only the main process logs at INFO level by
    default; others log ERROR level.
    """

    def __init__(self, name: str) -> None:
        """
        Initializes logger and distributed state.

        Args:
            name (str): Logger name (typically module or experiment name).
        """
        from accelerate import PartialState
        self.logger, self.distributed_state = ContextAdapter(
            logging.getLogger(name), extra={}), PartialState()

        self.debug = self.logger.debug
        self.info = self.logger.info
        self.warning = self.logger.warning
        self.error = self.logger.error
        self.critical = self.logger.critical

        self.logger.setLevel(logging.INFO if self.distributed_state.
                             is_main_process else logging.ERROR)

    @property
    def rank_zero_only(self) -> Callable[..., Any]:
        """
        Decorator: run function only on global main process.

        Returns:
            Callable[..., Any]: A decorator to ensure function runs only on
                                the main process.
        """
        return self.distributed_state.on_main_process

    @property
    def local_zero_only(self) -> Callable[..., Any]:
        """
        Decorator: run function only on local main process.

        Returns:
            Callable[..., Any]: A decorator to ensure function runs only on
                                the local main process.
        """
        return self.distributed_state.on_local_main_process

    @property
    def rank_zero_first(self) -> Callable[..., Any]:
        """
        Context: execute block on rank zero first, then all ranks.

        Returns:
            Callable[..., Any]: A decorator for synchronized execution on rank
                                zero first.
        """
        return self.distributed_state.main_process_first

    @property
    def local_zero_first(self) -> Callable[..., Any]:
        """
        Context: execute block on local rank zero first, then others.

        Returns:
            Callable[..., Any]: A decorator for synchronized execution on local
                                rank zero first.
        """
        return self.distributed_state.local_main_process_first

    def is_rank_zero(self) -> bool:
        """
        Returns whether the current process is the global main process.

        Returns:
            bool: Whether the current process is the global main process.
        """
        return self.distributed_state.is_main_process

    def rank(self) -> int:
        """
        Returns global process rank.

        Returns:
            int: The global process rank.
        """
        return self.distributed_state.process_index

    def local_rank(self) -> int:
        """
        Returns local process rank.

        Returns:
            int: The local process rank.
        """
        return self.distributed_state.local_process_index

    def world_size(self) -> int:
        """
        Returns total number of processes.

        Returns:
            int: Total number of processes in the distributed system.
        """
        return self.distributed_state.num_processes


class PureOverwatch:
    """
    Lightweight logger wrapper that does not assume a distributed context.
    All methods return identity behavior or no-op equivalents.
    """

    def __init__(self, name: str) -> None:
        """
        Initializes a pure logger (no accelerate dependency).

        Args:
            name (str): Logger name.
        """
        self.logger = ContextAdapter(logging.getLogger(name), extra={})

        self.debug = self.logger.debug
        self.info = self.logger.info
        self.warning = self.logger.warning
        self.error = self.logger.error
        self.critical = self.logger.critical

        self.logger.setLevel(logging.INFO)

    @staticmethod
    def get_identity_ctx() -> Callable[..., Any]:
        """
        Returns a decorator that acts as an identity function (no-op).

        Returns:
            Callable: Identity decorator.
        """

        def identity(fn: Callable[..., Any]) -> Callable[..., Any]:
            return fn

        return identity

    @property
    def rank_zero_only(self) -> Callable[..., Any]:
        """
        Identity decorator: no-op in pure logger context.

        Returns:
            Callable[..., Any]: No-op decorator for rank zero only.
        """
        return self.get_identity_ctx()

    @property
    def local_zero_only(self) -> Callable[..., Any]:
        """
        Identity decorator: no-op in pure logger context.

        Returns:
            Callable[..., Any]: No-op decorator for local zero only.
        """
        return self.get_identity_ctx()

    @property
    def rank_zero_first(self) -> Callable[..., Any]:
        """
        No-op context: yields control without synchronization.

        Returns:
            Callable[..., Any]: No-op context for rank zero first.
        """
        return nullcontext

    @property
    def local_zero_first(self) -> Callable[..., Any]:
        """
        No-op context: yields control without synchronization.

        Returns:
            Callable[..., Any]: No-op context for local zero first.
        """
        return nullcontext

    @staticmethod
    def is_rank_zero() -> bool:
        """
        Always returns True in non-distributed mode.

        Returns:
            bool: Always True in pure logger context.
        """
        return True

    @staticmethod
    def rank() -> int:
        """
        Returns 0 in non-distributed mode.

        Returns:
            int: Always 0 in pure logger context.
        """
        return 0

    @staticmethod
    def world_size() -> int:
        """
        Returns 1 in non-distributed mode.

        Returns:
            int: Always 1 in pure logger context.
        """
        return 1


def initialize_overwatch(
        name: str) -> Union[DistributedOverwatch, PureOverwatch]:
    """
    Factory function to initialize the appropriate Overwatch logger.

    Args:
        name (str): Logger name.

    Returns:
        Union[DistributedOverwatch, PureOverwatch]: An appropriate logger based
        on `WORLD_SIZE` env.
    """
    return DistributedOverwatch(name) if int(os.environ.get(
        'WORLD_SIZE', -1)) != -1 else PureOverwatch(name)
