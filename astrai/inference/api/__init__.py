"""Inference API: protocol handler, stop checker, tool parsers, and FastAPI server.

``app`` is no longer a module-level global. Use :func:`get_app` to access the
lazy singleton FastAPI instance.
"""

from astrai.inference.api.protocol import GenContext, ProtocolHandler, StopChecker
from astrai.inference.api.server import (
    AnthropicMessage,
    ChatCompletionRequest,
    ChatMessage,
    FunctionDef,
    MessagesRequest,
    ToolDef,
    get_app,
    run_server,
)
from astrai.inference.api.tool_parser import (
    BaseToolParser,
    SimpleJsonToolParser,
    ToolParserFactory,
)

__all__ = [
    "ProtocolHandler",
    "StopChecker",
    "GenContext",
    "BaseToolParser",
    "SimpleJsonToolParser",
    "ToolParserFactory",
    "AnthropicMessage",
    "ChatCompletionRequest",
    "ChatMessage",
    "FunctionDef",
    "ToolDef",
    "MessagesRequest",
    "get_app",
    "run_server",
]
