import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import MultiModalMessage
from autogen_agentchat.ui import Console
from autogen_core import Image 
from autogen_ext.models.ollama import OllamaChatCompletionClient

async def main():
    # 1. Connect to Ollama using your multi-modal Gemma model
    model_client = OllamaChatCompletionClient(
        model="gemma4:e4b",
        model_info={
            "vision": True,
            "family": "gemma",
            "function_calling": False,
            "json_output": False,
            "structured_output": False,
        }
    )

    # 2. Create the agent
    assistant = AssistantAgent(model_client=model_client, name="ImageReader")

    image = Image.from_file("C:\\Users\\Sagar\\Downloads\\W.jpeg")
    multimodal_message = MultiModalMessage(
        source="user",
        content=["What is in this image?", image]
    )
   
    await Console(assistant.run_stream(task=multimodal_message))
    await model_client.close()

asyncio.run(main())