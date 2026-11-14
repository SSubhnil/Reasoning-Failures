"""Run the Reasoning-Failures MCP server: stdio by default, Streamable HTTP with --http."""

import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m sst_mcp", description=__doc__)
    parser.add_argument("--http", action="store_true", help="serve Streamable HTTP at /mcp instead of stdio")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP bind address (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=7860, help="HTTP port (default 7860)")
    args = parser.parse_args(argv)

    from .server import build_server, configure_logging

    configure_logging()
    server = build_server(host=args.host, port=args.port, http=args.http)
    server.run(transport="streamable-http" if args.http else "stdio")


if __name__ == "__main__":
    main()
