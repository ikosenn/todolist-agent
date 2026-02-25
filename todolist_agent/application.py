import logging
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore
from agent import TodoAgent
from todolist_agent.schemas import TodoAgentContext
from langchain_core.messages import HumanMessage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Application:
    def __init__(self, store_uri: str, checkpointer_uri: str, model_name: str):
        self.store_uri = store_uri
        self.checkpointer_uri = checkpointer_uri
        self.store_db = None
        self.checkpointer_db = None
        self.todo_agent = None
        self.model_name = model_name

    async def stop(self):
        await self.store_db.aclose()
        await self.checkpointer_db.aclose()

    async def run_graph(self):
        user_id = "41c44bcf-1fc7-4c4a-9b5d-1ea24a1d7763"
        thread_id = "596bd914-21f0-42ee-b452-55a5c36c9335"
        config = {"configurable": {"thread_id": thread_id}}
        user_context = TodoAgentContext(**{"user_id": user_id})
        try:
            while True:
                input_text = input("Enter a message: ")
                human_message = [HumanMessage(content=input_text)]
                async for chunk in self.todo_agent.graph.astream({"messages": human_message}, config, context=user_context):
                    logger.info(chunk)
        except KeyboardInterrupt:
            logger.info("Stopping the application...")
            await self.stop()

    async def start(self):
        async with AsyncPostgresSaver.from_conn_string(self.checkpointer_uri) as checkpointer:
            async with AsyncPostgresStore.from_conn_string(self.store_uri) as store:
                self.store_db = store
                self.checkpointer_db = checkpointer
                self.todo_agent = TodoAgent(self, self.model_name)
                await self.run_graph()