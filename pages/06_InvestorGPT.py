import streamlit as st
import os
import requests
from typing import Type
from langchain_openai import ChatOpenAI
from langchain.tools import Tool, BaseTool
from pydantic import BaseModel, Field
from langchain.agents import initialize_agent, AgentType
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.messages.system import SystemMessage

llm = ChatOpenAI(temperature=0.1, model="gpt-4o")

alpha_vantage_api_key = os.environ.get("ALPHA_VANTAGE_API_KEY")


class StockSymbolSearchToolArgsScheme(BaseModel):
    query: str = Field(description="The query you will search for")


class StockSymbolSearchTool(BaseTool):
    name = "StockSymbolSearchTool"
    description = """
    Use this tool to find the stock symbol for a company. 
    It takes a query as an argument. 
    Example querty: Stock symbol for the Apple Company
    """
    args_schema: Type[StockSymbolSearchToolArgsScheme] = StockSymbolSearchToolArgsScheme

    def _run(self, query):
        ddg = DuckDuckGoSearchAPIWrapper()
        return ddg.run(query)


class CompanyOverviewArgsScheme(BaseModel):
    symbol: str = Field(description="Stock symbol of the compadny. Example:APPL, TSLA")


class CompanyOverviewTool(BaseTool):
    name = "CompanyOverview"
    description = """
    Use this to get an overview of the financials of the company.
    You should enter a stock symbol.
    """
    args_schema: Type[CompanyOverviewArgsScheme] = CompanyOverviewArgsScheme

    def _run(self, symbol):
        r = requests.get(
            f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={alpha_vantage_api_key}"
        )
        return r.json()


class CompanyIncomeStatementTool(BaseTool):
    name = "CompanyIncomeStatement"
    description = """
    Use this to get the income statement of the financials of the company.
    You should enter a stock symbol.
    """
    args_schema: Type[CompanyOverviewArgsScheme] = CompanyOverviewArgsScheme

    def _run(self, symbol):
        r = requests.get(
            f"https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={symbol}&apikey={alpha_vantage_api_key}"
        )
        return r.json()


class CompanyStockHistoryTool(BaseTool):
    name = "CompanyStockHistory"
    description = """
    Use this to get the weekly performance of a company stock.
    You should enter a stock symbol.
    """
    args_schema: Type[CompanyOverviewArgsScheme] = CompanyOverviewArgsScheme

    def _run(self, symbol):
        r = requests.get(
            f"https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&symbol={symbol}&apikey={alpha_vantage_api_key}"
        )
        response = r.json()
        return list(response["Weekly Time Series"].items())[:200]


class CompanyNewsSentimentsTool(BaseTool):
    name = "CompanyNewsSentiments"
    description = """
    Use this to get the recent market news of a company.
    You should enter a stock symbol.
    """
    args_schema: Type[CompanyOverviewArgsScheme] = CompanyOverviewArgsScheme

    def _run(self, symbol):
        r = requests.get(
            f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&symbol={symbol}&apikey={alpha_vantage_api_key}"
        )
        return r.json()


agent = initialize_agent(
    llm=llm,
    verbose=True,
    agent=AgentType.OPENAI_FUNCTIONS,
    handle_parsing_errors=True,
    tools=[
        StockSymbolSearchTool(),
        CompanyOverviewTool(),
        CompanyIncomeStatementTool(),
        CompanyStockHistoryTool(),
        CompanyNewsSentimentsTool(),
    ],
    agent_kwargs={
        "system_message": SystemMessage(
            content="""
        You are a hedge fund manager tasked with evaluating a company's stock. Provide a detailed assessment and your opinion on whether the stock is a buy or not. Consider the stock's performance, recent news, company overview, and income statement in your analysis. Be assertive in your judgment and clearly recommend whether to buy the stock or advise against it, providing reasons for your decision. Include the stock symbol in your response.           
        """
        )
    },
)

prompt = "Provide detailed but concise information on Rivian's stock, including financial data, recent news, income statements, and stock performance. Then analyze whether it is a potentially good stock investment and include the stock symbol in your response."

st.set_page_config(
    page_title="InvestorGPT",
    page_icon="📈",
)

st.title("InvestorGPT")

st.markdown(
    """
    <span style="font-size: 20px;">Enter the name of a company whose stock interests you, and we will conduct the research for you.</span>
    """,
    unsafe_allow_html=True,
)

company = st.text_input("Enter the name of the copmany you are interested.")

if company:
    result = agent.invoke(company)

    st.write(result["output"].replace("$", "\$"))
