from mcp.server.fastmcp import FastMCP
from mcp.server import Server  # Import standard Server
# Removed unused mcp.types import
import chromadb
from sentence_transformers import SentenceTransformer
import argparse

# Removed unused asyncio import
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route, Mount  # Import Mount
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from mcp.server.sse import SseServerTransport
from starlette.requests import Request  # Add Request import back

# Disable telemetry
import os

os.environ["CHROMADB_TELEMETRY"] = "false"

# Keep FastMCP instance, maybe for stdio or future use. Rename it.
mcp_fast = FastMCP("Godot RAG Server (FastMCP)")
# We will access the underlying server from mcp_fast for SSE
# Ensure mcp_sse instance is removed or commented out

client: chromadb.PersistentClient
collection: chromadb.Collection
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


async def homepage(request):
    return JSONResponse({"message": "Welcome to the Godot RAG MCP Server"})


# Apply the decorator to the FastMCP instance
@mcp_fast.tool()
def get_godot_context(query: str) -> list:
    """
    Godot engine has evolved a lot, and a lot of the pretrained knowledge is
    outdated and cannot be relied on.
    This tool retrieves a list of the latest relevant Godot documentation
    snippets based on the provided query.
    If user askes anything related to the Godot engine, including api and
    class references, even you are confident, this function should still be
    called. If there is any conflict between your knowledge and the retrieved
    snippets, the snippets should be considered more reliable, otherwise it's
    okay to rely on your knowledge. Only call this function if you are certain
    it's about the Godot engine.

    Args:
        query: keywords related to Godot engine

    Returns:
        list of relevant Godot documentation/references snippets
    """
    try:
        results = collection.query(
            query_embeddings=model.encode([query]).astype(float).tolist(),
            n_results=20,
        )

        # based on your data, you may include other info such as metadata, etc.
        documents = results["documents"][0][:]

        return documents
    except Exception as e:
        # Ensure the original function returns a list or handles errors
        # consistently. For simplicity in the handler, let's make it return
        # a list even on error.
        return [f"Error: Failed to query ChromaDB: {str(e)}"]


# Define Starlette app structure for SSE later if needed
app: Starlette | None = None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Start the Godot RAG MCP Server"
    )
    parser.add_argument(
        "--chromadb-path",
        "-d",
        type=str,
        required=True,
        help="Path to the ChromaDB database",
    )
    parser.add_argument(
        "--collection-name",
        "-c",
        type=str,
        required=True,
        help="Name of the ChromaDB collection to query",
    )
    parser.add_argument(
        "--transport",
        "-t",
        type=str,
        choices=["stdio", "sse"],
        default="stdio",
        required=False,
        help="Transport protocol to use (stdio or sse)",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=8000,
        help="Port to run the SSE server on",
    )

    args = parser.parse_args()

    client = chromadb.PersistentClient(path=args.chromadb_path)
    collection = client.get_collection(args.collection_name)

    if args.transport == "stdio":
        # Run FastMCP with stdio transport
        # Tool registration for stdio is handled by the
        # @mcp_fast.tool decorator.
        # Run FastMCP with stdio transport.
        mcp_fast.run(transport="stdio")
    elif args.transport == "sse":
        import uvicorn

        # SseServerTransport is already imported at the top level

        # Initialize SSE transport from MCP SDK
        # Use the standard message endpoint path
        sse_transport = SseServerTransport("/messages")

        # Access the underlying standard Server from the FastMCP instance
        actual_mcp_server_for_sse = mcp_fast._mcp_server

        # Define the SSE connection handler using the MCP transport
        async def handle_sse(request: Request):  # Accept Request object
            print("SSE connection attempt received")
            try:
                # Use request._send to get the send callable
                async with sse_transport.connect_sse(
                    request.scope, request.receive, request._send
                ) as streams:
                    print(f"SSE connection established, streams: {streams}")
                    # Run the underlying standard Server instance
                    await actual_mcp_server_for_sse.run(
                        streams[0], streams[1],
                        actual_mcp_server_for_sse.create_initialization_options(),
                    )
                    print(
                        "MCP SSE server finished running for this connection."
                    )
            except Exception as e:
                print(f"Error in handle_sse: {e}")
                # Optionally, re-raise or return an error response
                raise

        # No separate handle_messages function needed, mount directly

        # Define Starlette routes using the MCP SSE transport handlers
        routes = [
            Route("/", homepage),
            # GET endpoint for SSE connection
            Route("/sse", endpoint=handle_sse),
            # Mount the message handler directly from the transport instance
            Mount("/messages", app=sse_transport.handle_post_message),
        ]
        middleware = [Middleware(CORSMiddleware, allow_origins=["*"])]
        app = Starlette(routes=routes, middleware=middleware)

        # Run the Starlette app with uvicorn
        uvicorn.run(app, host="0.0.0.0", port=args.port)
