# ruff: noqa: E402
from typing import Annotated, TypedDict

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from chains import generate_chain, reflect_chain


class MessageGraph(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


REFLCT = "reflect"
GENERATE = "generate"


def generation_node(state: MessageGraph):
    return {"messages": [generate_chain.invoke({"messages": state["messages"]})]}


def reflection_node(state: MessageGraph):
    res = reflect_chain.invoke({"messages": state["messages"]})
    return {"messages": [HumanMessage(content=res.content)]}


builder = StateGraph(state_schema=MessageGraph)
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLCT, reflection_node)
builder.set_entry_point(GENERATE)


def should_continue(state: MessageGraph):
    if len(state["messages"]) > 5:
        return END
    return REFLCT


builder.add_conditional_edges(
    GENERATE,
    should_continue,
    path_map={END: END, REFLCT: REFLCT},
)
builder.add_edge(REFLCT, GENERATE)

graph = builder.compile()
graph.get_graph().draw_mermaid_png(output_file_path="flow.png")


if __name__ == "__main__":
    print("Hello Langgraph")
