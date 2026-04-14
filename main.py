from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

# from tavily import TavilyClient
from langchain_tavily import TavilySearch

load_dotenv()

# tavily = TavilyClient()


# @tool
# def search(query: str) -> str:
#     """
#     Tool that searches over internet.
#     Args:
#         query: The query to search for.
#     Return:
#         The search result.
#     """
#     print(f"Searching for {query}")
#     return tavily.search(query=query)


llm = ChatOpenAI(model="gpt-5")
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)


def main():
    print("Hello from langchain-agent!")
    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Search for 3 job posting for an ai engineer using langchain in the bay area on linked and list their details."
                )
            ]
        }
    )
    print(result)


if __name__ == "__main__":
    main()
