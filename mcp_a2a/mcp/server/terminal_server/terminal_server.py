# import os
# import subprocess
# import sys
#
# from mcp.server.fastmcp import FastMCP
#
# mcp = FastMCP("terminal")
#
#
# DEFAULT_WORKSPACE = "H:/SOFTWARE_DEVELOPMENT/MACHINE_LEARNING_PROJECT/ai-engineer-projects/mcp_a2a/workspace"
#
# #Ensure the directory actually exists before starting!
# os.makedirs(DEFAULT_WORKSPACE, exist_ok=True)
#
#
# @mcp.tool()
# def run_command(command: str):
#     """
#     Run a terminal command inside the workspace directory.
#     """
#     try:
#         print(f"Executing command: {command}", file=sys.stderr)
#
#         # subprocess will now successfully find the cwd
#         result = subprocess.run(
#             command,
#             shell=True,
#             cwd=DEFAULT_WORKSPACE,
#             capture_output=True,
#             text=True
#         )
#         return result.stdout or result.stderr
#     except Exception as e:
#         return str(e)
#
#
# if __name__ == "__main__":
#     print("Server initializing...", file=sys.stderr)
#     mcp.run(transport='stdio')


import os
import sys
import logging

# 1. IMMEDIATE LOGGING SETUP
# This will create a log file right on your H: drive the millisecond the script boots
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
            result = subprocess.run(
                command,
                shell=True,
                cwd=DEFAULT_WORKSPACE,
                capture_output=True,
                text=True
            )
            return result.stdout or result.stderr
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