import uuid
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Literal, Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class Memory(BaseModel):
    content: str = Field(description="The main content of the memory")


class MemoryCollection(BaseModel):
    memories: list[Memory] = Field(description="The collection of memories")


class Profile(BaseModel):
    """User profile information"""

    name: str | None = Field(default=None, description="The name of the user")
    location: str | None = Field(default=None, description="The location of the user")
    job: str | None = Field(default=None, description="The job of the user")
    connections: list[str] = Field(
        default_factory=list,
        description="The connections of the user such as family members, friends or coworkers",
    )
    interests: list[str] = Field(
        default_factory=list,
        description="The interests of the user such as hobbies, sports, music, movies, etc.",
    )


class ToDo(BaseModel):
    """Todo list item"""

    task: str = Field(description="The task to be completed")
    time_to_complete: int | None = Field(
        default=None, description="The time to complete the task in minutes"
    )
    deadline: datetime | None = Field(
        default=None, description="When the task needs to be completed by"
    )
    solutions: list[str] = Field(
        default_factory=list,
        description="List of specific, actionable solutions e.g. specific ideas, service providers, or concrete options relevant to completing the task",
    )
    status: Literal["pending", "in_progress", "completed", "archived"] = Field(
        default="pending", description="Current status of the task"
    )


class UpdateMemory(BaseModel):
    """Determines the type of memory update"""

    update_type: Literal["user", "todo", "instructions"] = Field(
        description="The type of memory update"
    )


class TodoAgentState(BaseModel):
    """State for the Todo Agent"""

    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)


class TodoAgentContext(BaseModel):
    """Context for the Todo Agent"""

    user_id: uuid.UUID = Field(description="The user ID")
