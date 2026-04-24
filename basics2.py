import asyncio
import os

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import MultiModalMessage
from autogen_agentchat.ui import Console
from autogen_core import Image
from autogen_ext.models.openai import OpenAIChatCompletionClient

os.environ[
    "OPENAI_API_KEY"] = "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"


async def main1():
    model_client = OpenAIChatCompletionClient( model="gpt-4o" )
    assistant = AssistantAgent( name="MultiModalAssistant", model_client=model_client )
    image = Image.from_file("/Users/rahulshetty/downloads/2A5A9749.jpg")
    multimodal_message = MultiModalMessage(
        content=["what do you see in this image", image], source="user"
    )
    await Console(assistant.run_stream(task=multimodal_message))
    await model_client.close()


asyncio.run( main1() )
