import asyncio
import os
import logging
from dotenv import load_dotenv
from application import Application

load_dotenv()


def main():
    store_uri = os.environ.get("POSTGRES_URL")
    checkpointer_uri = os.environ.get("POSTGRES_URL")
    app = Application(store_uri, checkpointer_uri, 'gpt-5-nano')
    asyncio.run(app.start())


if __name__ == "__main__":
    main()
