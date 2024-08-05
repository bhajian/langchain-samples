from langchain_aws import BedrockLLM
import boto3
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate
from typing import TypedDict, Annotated, List, Union
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
import operator
from langchain.tools import BaseTool, StructuredTool, Tool, tool
import random




bedrock=boto3.client(
    service_name="bedrock-runtime",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET
    )

# "meta.llama3-70b-instruct-v1:0"

llm = BedrockLLM(
    model_id="meta.llama3-8b-instruct-v1:0",
    client=bedrock,
    region_name='us-east-1'
)



