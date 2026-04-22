import asyncio  # Allows us to run async (non-blocking) functions

from autogen_core.models import UserMessage  # The message format we send to the model
from autogen_ext.models.ollama import OllamaChatCompletionClient  # Ollama AI client


async def main():

    # Connect to Ollama and tell it which model to use
    # model_info tells the client what features our model supports
    model_client = OllamaChatCompletionClient(
        model="gemma4:e4b",
        model_info={
            "vision": False,            # Cannot process images
            "function_calling": False,  # Cannot call tools/functions
            "json_output": True,        # Can return JSON formatted responses
            "family": "unknown",        # Model family (unknown for custom models)
            "structured_output": False  # Cannot follow strict output schemas
        }
    )

    # Create a message — like typing in a chat box
    # content = what you want to ask
    # source  = who is sending it (always "user" for us)
    message = UserMessage(content="What is the capital of India?", source="user")

    # Send the message to Ollama and wait for a response
    response = await model_client.create([message])

    # Print the response from the AI
    print(response.content)

    # Close the connection cleanly when done
    await model_client.close()


# Entry point — starts the async main() function
asyncio.run(main())
