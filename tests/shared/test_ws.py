from collections.abc import AsyncGenerator, Generator
from urllib.parse import urlparse

import anyio
import pytest
from starlette.applications import Starlette
from starlette.routing import WebSocketRoute
from starlette.websockets import WebSocket

from mcp import MCPError
from mcp.client.session import ClientSession
from mcp.client.websocket import websocket_client
from mcp.server import Server, ServerRequestContext
from mcp.server.websocket import websocket_server
from mcp.types import (
    CallToolRequestParams,
    CallToolResult,
    EmptyResult,
    InitializeResult,
    ListToolsResult,
    PaginatedRequestParams,
    ReadResourceRequestParams,
    ReadResourceResult,
    TextContent,
    TextResourceContents,
    Tool,
)
from tests.test_helpers import run_uvicorn_in_thread

SERVER_NAME = "test_server_for_WS"


async def handle_read_resource(ctx: ServerRequestContext, params: ReadResourceRequestParams) -> ReadResourceResult:
    parsed = urlparse(str(params.uri))
    if parsed.scheme == "foobar":
        return ReadResourceResult(
            contents=[TextResourceContents(uri=str(params.uri), text=f"Read {parsed.netloc}", mime_type="text/plain")]
        )
    elif parsed.scheme == "slow":
        await anyio.sleep(2.0)
        return ReadResourceResult(
            contents=[
                TextResourceContents(
                    uri=str(params.uri), text=f"Slow response from {parsed.netloc}", mime_type="text/plain"
                )
            ]
        )
    raise MCPError(code=404, message="OOPS! no resource with that URI was found")


async def handle_list_tools(  # pragma: no cover
    ctx: ServerRequestContext, params: PaginatedRequestParams | None
) -> ListToolsResult:
    return ListToolsResult(
        tools=[
            Tool(
                name="test_tool",
                description="A test tool",
                input_schema={"type": "object", "properties": {}},
            )
        ]
    )


async def handle_call_tool(  # pragma: no cover
    ctx: ServerRequestContext, params: CallToolRequestParams
) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=f"Called {params.name}")])


def _create_server() -> Server:
    return Server(
        SERVER_NAME,
        on_read_resource=handle_read_resource,
        on_list_tools=handle_list_tools,
        on_call_tool=handle_call_tool,
    )


# Test fixtures
def make_server_app() -> Starlette:
    """Create test Starlette app with WebSocket transport"""
    server = _create_server()

    async def handle_ws(websocket: WebSocket):
        async with websocket_server(websocket.scope, websocket.receive, websocket.send) as streams:
            await server.run(streams[0], streams[1], server.create_initialization_options())

    app = Starlette(routes=[WebSocketRoute("/ws", endpoint=handle_ws)])
    return app


@pytest.fixture()
def server_url() -> Generator[str, None, None]:
    with run_uvicorn_in_thread(make_server_app()) as url:
        yield url.replace("http://", "ws://")


@pytest.fixture()
async def initialized_ws_client_session(server_url: str) -> AsyncGenerator[ClientSession, None]:
    """Create and initialize a WebSocket client session"""
    async with websocket_client(server_url + "/ws") as streams:
        async with ClientSession(*streams) as session:
            # Test initialization
            result = await session.initialize()
            assert isinstance(result, InitializeResult)
            assert result.server_info.name == SERVER_NAME

            # Test ping
            ping_result = await session.send_ping()
            assert isinstance(ping_result, EmptyResult)

            yield session


# Tests
@pytest.mark.anyio
async def test_ws_client_basic_connection(server_url: str) -> None:
    """Test the WebSocket connection establishment"""
    async with websocket_client(server_url + "/ws") as streams:
        async with ClientSession(*streams) as session:
            # Test initialization
            result = await session.initialize()
            assert isinstance(result, InitializeResult)
            assert result.server_info.name == SERVER_NAME

            # Test ping
            ping_result = await session.send_ping()
            assert isinstance(ping_result, EmptyResult)


@pytest.mark.anyio
async def test_ws_client_happy_request_and_response(
    initialized_ws_client_session: ClientSession,
) -> None:
    """Test a successful request and response via WebSocket"""
    result = await initialized_ws_client_session.read_resource("foobar://example")
    assert isinstance(result, ReadResourceResult)
    assert isinstance(result.contents, list)
    assert len(result.contents) > 0
    assert isinstance(result.contents[0], TextResourceContents)
    assert result.contents[0].text == "Read example"


@pytest.mark.anyio
async def test_ws_client_exception_handling(
    initialized_ws_client_session: ClientSession,
) -> None:
    """Test exception handling in WebSocket communication"""
    with pytest.raises(MCPError) as exc_info:
        await initialized_ws_client_session.read_resource("unknown://example")
    assert exc_info.value.error.code == 404


@pytest.mark.anyio
async def test_ws_client_timeout(
    initialized_ws_client_session: ClientSession,
) -> None:
    """Test timeout handling in WebSocket communication"""
    # Set a very short timeout to trigger a timeout exception
    with pytest.raises(TimeoutError):
        with anyio.fail_after(0.1):  # 100ms timeout
            await initialized_ws_client_session.read_resource("slow://example")

    # Now test that we can still use the session after a timeout
    with anyio.fail_after(5):  # Longer timeout to allow completion
        result = await initialized_ws_client_session.read_resource("foobar://example")
        assert isinstance(result, ReadResourceResult)
        assert isinstance(result.contents, list)
        assert len(result.contents) > 0
        assert isinstance(result.contents[0], TextResourceContents)
        assert result.contents[0].text == "Read example"
