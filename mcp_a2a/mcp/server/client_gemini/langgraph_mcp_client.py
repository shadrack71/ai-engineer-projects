import asyncio
import os
import sys
import json
import operator
from typing import Optional, Annotated, TypedDict
from contextlib import AsyncExitStack

# MCP & OpenAI Imports
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import AsyncOpenAI
from dotenv import load_dotenv

# LangGraph Imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()


#Define the Graph State structure
class AgentState(TypedDict):
    # 'operator.add' tells LangGraph to append new messages to the existing list
    # rather than overwriting it, preserving the conversation history.
    messages: Annotated[list, operator.add]


class LangGraphMCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.openai_tools = []

        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file.")

        self.openai_client = AsyncOpenAI(api_key=openai_api_key)
        self.graph = None  # Will hold the compiled graph

    async def connect_to_server(self, server_script_path: str):
        """Connect to the MCP server and fetch tools."""
        command = sys.executable if server_script_path.endswith('.py') else "node"

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

        # Build the graph immediately after tools are loaded
        self.build_graph()

    def build_graph(self):
        """Construct the stateful LangGraph."""

        # NODE 1: The LLM Agent
        async def agent_node(state: AgentState):
            print("\n[Thinking...] Sending request to OpenAI...")
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=state["messages"],
                tools=self.openai_tools
            )
            message = response.choices[0].message

            # Return a dictionary wrapping the new message in a list.
            # LangGraph's state reducer (operator.add) automatically appends it to the history.
            return {"messages": [message.model_dump(exclude_unset=True)]}

        # NODE 2: The Tool Executor
        async def tool_node(state: AgentState):
            last_message = state["messages"][-1]
            tool_messages = []

            for tool_call in last_message.get("tool_calls", []):
                tool_name = tool_call["function"]["name"]
                tool_args = json.loads(tool_call["function"]["arguments"])

                print(f"\n [Executing Tool] {tool_name} -> {tool_args}")

                try:
                    result = await self.session.call_tool(tool_name, tool_args)
                    if result.content and result.content[0].text.strip():
                        tool_output = result.content[0].text
                    else:
                        tool_output = "[Command executed successfully with no output]"
                except Exception as e:
                    tool_output = f"Error executing tool: {str(e)}"

                print(f"[Tool Result] {tool_output.strip()}")

                tool_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": tool_output
                })

            return {"messages": tool_messages}

        # CONDITIONAL ROUTER: Decide where to go after the LLM speaks
        def should_continue(state: AgentState):
            last_message = state["messages"][-1]
            if last_message.get("tool_calls"):
                return "tools"  # Go to tool node
            return END  # Stop execution


        # COMPILE THE GRAPH

        workflow = StateGraph(AgentState)

        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", tool_node)

        workflow.set_entry_point("agent")
        workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
        workflow.add_edge("tools", "agent")

        # MemorySaver is what makes this graph truly "stateful" across different chat queries
        memory = MemorySaver()
        self.graph = workflow.compile(checkpointer=memory)

    async def chat_loop(self):
        """Run an interactive, stateful chat session."""
        print("\nStateful LangGraph + MCP Client Started! Type 'quit' to exit.")

        # A thread ID tells the MemorySaver which conversation history to retrieve
        config = {"configurable": {"thread_id": "my_dev_session"}}

        while True:
            query = input("\nQuery: ").strip()
            if query.lower() == 'quit':
                break

            # Stream the graph execution. We pass only the NEW user message.
            # MemorySaver automatically injects the past conversation history.
            async for event in self.graph.astream({"messages": [{"role": "user", "content": query}]}, config=config):
                # The nodes have internal print statements, so we just iterate through the events silently here.
                pass

            # After the graph hits END, extract the final text message
            final_state = self.graph.get_state(config)
            last_message = final_state.values["messages"][-1]
            if last_message.get("content"):
                print(f"\nAgent: {last_message['content']}")

    async def cleanup(self):
        await self.exit_stack.aclose()


async def main():
    if len(sys.argv) < 2:
        print("Usage: python langgraph_mcp_client.py <path_to_server_script>")
        sys.exit(1)

    client = LangGraphMCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())