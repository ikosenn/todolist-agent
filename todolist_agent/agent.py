import uuid
from datetime import datetime
from langchain.chat_models import init_chat_model
from trustcall import create_extractor
from application import Application
from schemas import TodoAgentState, TodoAgentContext, UpdateMemory, Profile, ToDo
from todolist_agent.utils import Spy, extract_tool_info
from langgraph.prebuilt import RunnableConfig, Runtime
from langchain_core.messages import SystemMessage, merge_message_runs, HumanMessage
from langgraph.graph import StateGraph, START, END


AGENT_SYSTEM_MSG = """You are a helpful chatbot.
You are designed to be a companion to a user, helping them keep track of their Todo list.

You have a long term memory which keeps track of three things:
1. The user's profile information
2. The user's Todo list
3. General instructions for updating the Todo list.

Here is the current User Profile (may be empty if no information  has been collected yet)
<user_profile>
{user_profile}
</user_profile>

Here is the current Todo List (may be empty if no tasks have been added yet)
<todo_list>
{todo_list}
</todo_list>

Here are the current user-specified preference for updating the Todo list (may be empty if no preferences have been specified yet)
<instructions>
{instructions}
</instructions>

Here are your instructions for reasoning about the user's messages:

1. Reason carfully about the user's messages as presented below.

2. Decided whether any of your long-term memories should be updated:
- If personal information was provided about the user, update the user's profile by calling UpdateMemory tool with the tyoe `user`
- If tasks are mentioned, update the Todo list by calling UpdateMemory tool with the type `todo`
- If the user has specified preferences for how to update he Todo list, update the instructions by calling UpdateMemory tool with the tyoe `instructions`

3. Tell the user that you have updated you memory if appropriate:
- Do not tell the user you have update the user's profile.
- Tell the user when you update the Todo list.
- Do not tell the user that you have updated instructions.

4. Err on the side of updating the todo list. No need to ask for explicit permission.

5. Respond naturally to the user after a tool call was made to save memories of if no tool call was made.
"""

TRUSTCALL_EX_SYS_MSG = """Reflect on the following interaction.

Use the provided tools to retain any necessary memories about the user.

Use parallel tool calling to handle updates and inserts simultaneously.

System Time: {system_time}
"""

INSTRUCTIONS_EX_SYS_MSG = """Reflect on the following interaction.

Base on this interaction, update your instructions for how to update Tofo list items.

Use any feedback from the user to update how they like to have items added, e.t.c

Your current instructions are (may be empty if no instructions have been added yet)
<current_instructions>
{current_instructions}
</current_instructions>
"""


class TodoAgent:
    def __init__(self, app: Application, model_name: str):
        self.app = app
        self.model = init_chat_model(model_name)
        self.profile_extractor = create_extractor(
            self.model,
            tools=[Profile],
            tool_choice="Profile",
        )
        self.todo_extractor = create_extractor(
            self.model,
            tools=[ToDo],
            tool_choice="ToDo",
            enable_inserts=True,
        )
        self.graph = self._build_graph()

    def _build_graph(self):
        async def determine_memory_update(
            state: TodoAgentState,
            config: RunnableConfig,
            runtime: Runtime[TodoAgentContext],
        ):
            user_id = runtime.context.user_id
            namespace = ("profile", user_id)
            memories = await runtime.store.asearch(namespace)
            if memories:
                user_profile = memories[0].value
            else:
                user_profile = ""

            namespace = ("todo", user_id)
            memories = await runtime.store.asearch(namespace)
            todo = "\n".join(f"{m.value}" for m in memories)

            namespace = ("instructions", user_id)
            memories = await runtime.store.asearch(namespace)
            if memories:
                instructions = memories[0].value
            else:
                instructions = ""

            system_msg = SystemMessage(
                content=AGENT_SYSTEM_MSG.format(
                    user_profile=user_profile, todo_list=todo, instructions=instructions
                )
            )

            resp = await self.model.bind_tools(
                [UpdateMemory], parallel_tool_calls=False
            ).ainvoke([system_msg, *state.messages])

            return {"messages": [resp]}

        async def update_profile(
            state: TodoAgentState,
            config: RunnableConfig,
            runtime: Runtime[TodoAgentContext],
        ):
            """Reflect on the chat history and update the memory collection"""
            user_id = runtime.context.user_id
            namespace = ("profile", user_id)
            existing_items = await runtime.store.asearch(namespace)
            tool_name = "Profile"
            # TODO: Determine what this existing item looks like
            existing_memories = (
                [
                    (existing_item.key, tool_name, existing_item.value)
                    for existing_item in existing_items
                ]
                if existing_items
                else None
            )
            system_msg = SystemMessage(
                content=TRUSTCALL_EX_SYS_MSG.format(
                    system_time=datetime.now().isoformat()
                )
            )
            updated_msg = list(merge_message_runs([system_msg] + state.messages[:-1]))

            resp = await self.profile_extractor.ainvoke(
                {"messages": updated_msg, "existing": existing_memories}
            )

            for r, rmeta in zip(resp["responses"], resp["response_metadata"]):
                await runtime.store.aput(
                    namespace,
                    rmeta.get("json_doc_id"),
                    str(uuid.uuid4()),
                    r.model_dump(mode="json"),
                )
            tool_calls = state.messages[-1].tool_calls
            return {
                "messages": [
                    {
                        "role": "tool",
                        "content": "updated profile",
                        "tool_call_id": tool_calls[0]["id"],
                    }
                ]
            }

        async def update_todo(
            state: TodoAgentState,
            config: RunnableConfig,
            runtime: Runtime[TodoAgentContext],
        ):
            """Reflect on the chat history and update the memory collection"""
            user_id = runtime.context.user_id
            namespace = ("todo", user_id)
            existing_items = await runtime.store.asearch(namespace)
            tool_name = "ToDo"
            existing_memories = (
                [
                    (existing_item.key, tool_name, existing_item.value)
                    for existing_item in existing_items
                ]
                if existing_items
                else None
            )
            system_msg = SystemMessage(
                content=TRUSTCALL_EX_SYS_MSG.format(
                    system_time=datetime.now().isoformat()
                )
            )
            updated_msg = list(merge_message_runs([system_msg] + state.messages[:-1]))
            spy = Spy()
            todo_extractor = self.todo_extractor.with_listeners(on_end=spy)
            resp = await todo_extractor.ainvoke(
                {"messages": updated_msg, "existing": existing_memories}
            )
            for r, rmeta in zip(resp["responses"], resp["response_metadata"]):
                await runtime.store.aput(
                    namespace,
                    rmeta.get("json_doc_id"),
                    str(uuid.uuid4()),
                    r.model_dump(mode="json"),
                )
            tool_calls = state.messages[-1].tool_calls
            tool_update_msg = extract_tool_info(spy.called_tools, tool_name)

            return {
                "messages": [
                    {
                        "role": "tool",
                        "content": tool_update_msg,
                        "tool_call_id": tool_calls[0]["id"],
                    }
                ]
            }

        async def update_instruction(
            state: TodoAgentState,
            config: RunnableConfig,
            runtime: Runtime[TodoAgentContext],
        ):
            """Reflect on the chat history and update the memory collection"""
            user_id = runtime.context.user_id
            namespace = ("instructions", user_id)
            key = "user_instructions"
            existing_items = await runtime.store.aget(namespace, key)
            system_msg = SystemMessage(
                content=INSTRUCTIONS_EX_SYS_MSG.format(
                    current_instructions=existing_items.value if existing_items else ""
                )
            )
            human_msg = HumanMessage(
                content="Please update the instruction based on the conversation."
            )
            resp = await self.model.ainvoke(
                [system_msg] + state.messages[:-1] + [human_msg]
            )
            await runtime.store.aput(namespace, key, {"memory": resp.content})
            tool_calls = state.messages[-1].tool_calls
            return {
                "messages": [
                    {
                        "role": "tool",
                        "content": "updated instructions",
                        "tool_call_id": tool_calls[0]["id"],
                    }
                ]
            }

        async def route_message(
            state: TodoAgentState,
            config: RunnableConfig,
            runtime: Runtime[TodoAgentContext],
        ):
            """Route the message to the appropriate tool"""
            msg = state.messages[-1]
            if len(msg.tool_calls) == 0:
                return END
            tool_call = msg.tool_calls[0]
            if tool_call["args"]["update_type"] == "user":
                return "update_profile"
            elif tool_call["args"]["update_type"] == "todo":
                return "update_todo"
            elif tool_call["args"]["update_type"] == "instructions":
                return "update_instruction"
            else:
                raise ValueError(
                    f"Invalid update type: {tool_call['args']['update_type']}"
                )

        def build_todo_agent_graph():
            builder = StateGraph(TodoAgentState)
            builder.add_node("determine_memory_update", determine_memory_update)
            builder.add_node("update_profile", update_profile)
            builder.add_node("update_todo", update_todo)
            builder.add_node("update_instruction", update_instruction)
            builder.add_node("route_message", route_message)

            builder.add_edge(START, "determine_memory_update")
            builder.add_conditional_edges("determine_memory_update", route_message)
            builder.add_edge("update_profile", "determine_memory_update")
            builder.add_edge("update_todo", "determine_memory_update")
            builder.add_edge("update_instruction", "determine_memory_update")
            builder.add_edge("route_message", END)
            return builder.compile(
                checkpointer=self.app.checkpointer_db, store=self.app.store_db
            )
