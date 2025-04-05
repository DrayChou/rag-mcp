# Godot RAG MCP

A Model Context Protocol (MCP) tool that retrieves relevant information about
Godot game engine from a local ChromaDB database.

Although this tool is for Godot in particular, but with small changes to the
tool description, it can be used for any chroma database.

## Usage

- Get the chroma_db vector store ready using the
  [web2embeddings](https://github.com/zivshek/web2embeddings) tool.
- Download or clone this repository to your local disk.
- Configure the MCP server in VSCode Insiders (see below config instructions).

## Configuration with Visual Studio Code Insiders

**_NOTE:_** You may need to download
[**Visual Studio Code Insiders**](https://code.visualstudio.com/insiders/)
version to use MCP or Agent mode.

The system is configured through `settings.json`:

```json
"mcp": {
    "inputs": [],
    "servers": {
        // stdio 协议
        "godot_rag": { // or whatever name you want
            "command": "cmd", // replace this with "uv" on Mac or Linux
            "args": [
                "/c", // remove this on Mac or Linux
                "uv", // remove this on Mac or Linux
                "run",
                "path to the server script 'server.py'", // C:\\dev\\rag-mcp\\server.py
                "-d",
                "path to the chroma_db on your computer", // C:\\dev\\web2embeddings\\artifacts\\vector_stores\\chroma_db
                "-c",
                "name of the collection in the chroma_db" // godotengine_chunks_SZ_400_O_20_all-MiniLM-L6-v2
            ]
        },
        // sse 协议
        "godot_rag_sse": { // or whatever name you want
            "command": "cmd", // replace this with "uv" on Mac or Linux
            "args": [
                "/c", // remove this on Mac or Linux
                "uv", // remove this on Mac or Linux
                "run",
                "path to the server script 'server.py'", // C:\\dev\\rag-mcp\\server.py
                "-d",
                "path to the chroma_db on your computer", // C:\\dev\\web2embeddings\\artifacts\\vector_stores\\chroma_db
                "-c",
                "name of the collection in the chroma_db" // godotengine_chunks_SZ_400_O_20_all-MiniLM-L6-v2
                "-t", 
                "sse" // "sse" for sse protocol
                "-p" // specify the port number
                "8000" // replace with the port number you want to use
            ]
        }
    }
}
```
