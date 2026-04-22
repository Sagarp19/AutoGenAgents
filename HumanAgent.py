import asyncio

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.ollama import OllamaChatCompletionClient

async def main():
    # 1. Connect to Ollama using your multi-modal Gemma model
    model_client = OllamaChatCompletionClient(
        model="gemma4:e4b",
        model_info={
            "vision": True,
            "family": "gemma",
            "function_calling": True,
            "json_output": True,
            "structured_output": True,
        }
    )

    # 2. Create the agent
    assistant1 = AssistantAgent(model_client=model_client, name="Debate1", 
                                system_message="You are in a Debate and one of the speaker keep the debate intresting."
                                "when the other particpamt says 'Thanks Done' or something similar, acknowledge and say 'Lesson Complete'" \
                                " to end  session")
    #assistant2 = AssistantAgent(model_client=model_client, name="Debate2", system_message="You are in a Debate other speaker keep the debate boring.")
    user_proxy = UserProxyAgent(name="Student")
    
    team = RoundRobinGroupChat(participants=[assistant1, user_proxy], name="Classroom", termination_condition=TextMentionTermination("Lesson Complete"))
  
   
    await Console(team.run_stream(task="Lets discuss about Chicken"))
    await model_client.close()

asyncio.run(main())