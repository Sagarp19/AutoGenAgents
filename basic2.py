import asyncio  # Allows us to run async (non-blocking) functions

from autogen_core.models import UserMessage  # The message format we send to the model
from autogen_ext.models.ollama import OllamaChatCompletionClient  # Ollama AI client
from autogen_agentchat.agents import AssistantAgent  # A simple agent that can hold conversations
from autogen_agentchat.ui import Console

async def main():

    # Connect to Ollama and tell it which model to use
    # model_info tells the client what features our model supports
    model_client = OllamaChatCompletionClient(
        model="gemma4:e4b",
        model_info={
            "vision": True,            # Cannot process images
            "function_calling": True,  # Cannot call tools/functions
            "json_output": True,        # Can return JSON formatted responses
            "family": "unknown",        # Model family (unknown for custom models)
            "structured_output": True  # Cannot follow strict output schemas
        }
    )
    assistant = AssistantAgent(model_client=model_client, name="Assistant")

    await Console(assistant.run_stream(task="What is 2*9"))

    await model_client.close()


# Entry point — starts the async main() function
asyncio.run(main())
