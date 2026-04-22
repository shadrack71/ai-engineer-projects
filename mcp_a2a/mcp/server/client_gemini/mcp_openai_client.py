import asyncio
import os
import sys
import json
from typing import Optional
from contextlib import AsyncExitStack

# Import MCP client components
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Import OpenAI SDK
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()


class MCPOpenAIClient:
    def __init__(self):
        """Initialize the MCP client and configure the OpenAI API."""
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

        # Retrieve the OpenAI API key
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not found. Please add it to your .env file.")

        # Configure the Async OpenAI client
        self.openai_client = AsyncOpenAI(api_key=openai_api_key)

    async def connect_to_server(self, server_script_path: str):
        """Connect to the MCP server and list available tools."""
        command = sys.executable if server_script_path.endswith('.py') else "node"

        # Pass environment variables to ensure Windows stdio pipes work correctly
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"

        server_params = StdioServerParameters(command=command, args=[server_script_path], env=env)
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport

        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        await self.session.initialize()

        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

        # Convert MCP tools to OpenAI tool format
        self.openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema
                }
            }
            for tool in tools
        ]

    # async def process_query(self, query: str) -> str:
    #     """Process a query using OpenAI and execute MCP tool calls if needed."""
    #
    #     # OpenAI expects a list of message dictionaries
    #     messages = [{"role": "user", "content": query}]
    #
    #     # 1. Initial Call to OpenAI
    #     response = await self.openai_client.chat.completions.create(
    #         model="gpt-4o-mini",  # Fast, cheap, and excellent at tool calling
    #         messages=messages,
    #         tools=self.openai_tools
    #     )
    #
    #     message = response.choices[0].message
    #
    #     # 2. Check if OpenAI wants to use a tool
    #     if message.tool_calls:
    #         # We must append the assistant's tool call request to the conversation history
    #         messages.append(message)
    #
    #         # Iterate through the requested tools (OpenAI can request multiple tools at once)
    #         for tool_call in message.tool_calls:
    #             tool_name = tool_call.function.name
    #             # OpenAI returns arguments as a JSON string, so we must parse it
    #             tool_args = json.loads(tool_call.function.arguments)
    #
    #             print(f"\n[OpenAI requested tool call: {tool_name} with args {tool_args}]")
    #
    #             try:
    #                 # Execute the tool on your local MCP server
    #                 result = await self.session.call_tool(tool_name, tool_args)
    #
    #                 # MCP returns a list of TextContent objects. We extract the text payload.
    #                 tool_output = result.content[0].text if result.content else "Success"
    #             except Exception as e:
    #                 tool_output = f"Error executing tool: {str(e)}"
    #
    #             # Append the result of the tool execution to the conversation history
    #             messages.append({
    #                 "role": "tool",
    #                 "tool_call_id": tool_call.id,
    #                 "content": tool_output
    #             })
    #
    #         # 3. Second Call to OpenAI with the tool results
    #         second_response = await self.openai_client.chat.completions.create(
    #             model="gpt-4o-mini",
    #             messages=messages,
    #             tools=self.openai_tools
    #         )
    #         return second_response.choices[0].message.content
    #     else:
    #         # If no tool was called, just return the standard text response
    #         return message.content

    async def process_query(self, query: str) -> str:
        """Process a query using OpenAI and execute MCP tool calls if needed."""

        messages = [{"role": "user", "content": query}]

        while True:
            #  Visually track the network call
            print("\n[Thinking...] Sending request to OpenAI...")

            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=self.openai_tools
            )

            message = response.choices[0].message

            if message.tool_calls:
                # Safely convert the assistant message to a dictionary to avoid serialization bugs
                messages.append(message.model_dump(exclude_unset=True))

                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)

                    # 2. Visually track the local tool execution
                    print(f"\n[Executing Tool] {tool_name} -> {tool_args}")

                    try:
                        result = await self.session.call_tool(tool_name, tool_args)

                        #  Handle empty terminal outputs gracefully
                        if result.content and result.content[0].text.strip():
                            tool_output = result.content[0].text
                        else:
                            tool_output = "[Command executed successfully with no output]"

                    except Exception as e:
                        tool_output = f"Error executing tool: {str(e)}"

                    # Print the result so you can see what the AI sees
                    print(f"[Tool Result] {tool_output.strip()}")

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_output
                    })
            else:
                return message.content
    async def chat_loop(self):
        """Run an interactive chat session with the user."""
        print("\nMCP OpenAI Client Started! Type 'quit' to exit.")

        while True:
            query = input("\nQuery: ").strip()
            if query.lower() == 'quit':
                break

            response = await self.process_query(query)
            print("\n" + response)

    async def cleanup(self):
        """Clean up resources before exiting."""
        await self.exit_stack.aclose()


async def main():
    if len(sys.argv) < 2:
        print("Usage: python mcp_openai_client.py <path_to_server_script>")
        sys.exit(1)

    client = MCPOpenAIClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())