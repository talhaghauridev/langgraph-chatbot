from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
import sqlite3

from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

con = sqlite3.connect(database="chatbot.db", check_same_thread=False)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: dict):
    messages = state.get("messages", []) if isinstance(state, dict) else []
    response = llm.invoke(messages)
    return {"messages": messages + [response]}


checkpointer = SqliteSaver(conn=con)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)


def retrieve_all_threads():
    """Get all thread IDs from database"""
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        cfg = getattr(checkpoint, "config", {}) or {}
        thread_id = cfg.get("configurable", {}).get("thread_id")
        if thread_id is not None:
            all_threads.add(thread_id)
    return list(all_threads)


def retrieve_user_threads(user_id):
    """Get threads that belong to specific user only"""
    all_threads = retrieve_all_threads()
    user_threads = [
        thread for thread in all_threads if thread.startswith(f"{user_id}_")
    ]
    return user_threads
