import os
import subprocess
import sys

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("terminal")


DEFAULT_WORKSPACE = "H:/SOFTWARE_DEVELOPMENT/MACHINE_LEARNING_PROJECT/ai-engineer-projects/mcp_a2a/workspace"

#Ensure the directory actually exists before starting!
os.makedirs(DEFAULT_WORKSPACE, exist_ok=True)


@mcp.tool()
def run_command(command: str):
    """
    Run a terminal command inside the workspace directory.
    """
    try:
        print(f"Executing command: {command}", file=sys.stderr)

        # subprocess will now successfully find the cwd
        result = subprocess.run(
            command,
            shell=True,
            cwd=DEFAULT_WORKSPACE,
            capture_output=True,
            text=True
        )
        return result.stdout or result.stderr
    except Exception as e:
        return str(e)


if __name__ == "__main__":
    print("Server initializing...", file=sys.stderr)
    mcp.run(transport='stdio')