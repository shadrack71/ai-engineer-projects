import os
import sys
import logging


LOG_FILE = "H:/SOFTWARE_DEVELOPMENT/MACHINE_LEARNING_PROJECT/ai-engineer-projects/mcp_a2a/mcp_server_debug.log"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.info("--- SERVER BOOT SEQUENCE INITIATED ---")

try:
    import subprocess
    from mcp.server.fastmcp import FastMCP

    logging.info("Imports successful.")

    mcp = FastMCP("terminal")

    DEFAULT_WORKSPACE = "H:/SOFTWARE_DEVELOPMENT/MACHINE_LEARNING_PROJECT/ai-engineer-projects/mcp_a2a/workspace"
    os.makedirs(DEFAULT_WORKSPACE, exist_ok=True)
    logging.info(f"Workspace verified at {DEFAULT_WORKSPACE}")


    @mcp.tool()
    def run_command(command: str):
        """Run a terminal command inside the workspace directory."""
        try:
            logging.info(f"Executing command: {command}")

            # ADDED TIMEOUT AND DEVNULL STDIN TO PREVENT HANGS
            result = subprocess.run(
                command,
                shell=True,
                cwd=DEFAULT_WORKSPACE,
                capture_output=True,
                text=True,
                timeout=10,  # Kill the command if it takes over 10 seconds
                stdin=subprocess.DEVNULL  # Never wait for human keyboard input
            )

            # Guarantee we always return a string
            output = result.stdout or result.stderr
            return output if output.strip() else "[Command executed successfully with no output]"

        except subprocess.TimeoutExpired:
            error_msg = f"Error: Command '{command}' timed out after 10 seconds. Do not use commands that require interactive user input."
            logging.error(error_msg)
            return error_msg
        except Exception as e:
            logging.error(f"Command execution failed: {str(e)}")
            return str(e)


    if __name__ == "__main__":
        logging.info("Entering MCP stdio event loop...")
        mcp.run(transport='stdio')
        logging.info("Server loop exited cleanly.")

except Exception as e:
    # If ANYTHING goes wrong during boot, it gets written to the file here
    logging.exception("FATAL ERROR IN SERVER BOOT:")
    sys.exit(1)