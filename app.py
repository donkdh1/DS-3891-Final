#!/usr/bin/env python
# coding: utf-8

# load libraries
import os
import time

from flask import Flask, request

from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain.agents import AgentOutputParser, create_react_agent, AgentExecutor
from langchain.prompts import StringPromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferMemory, MongoDBChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from typing import List

from langchain_core.tools import Tool
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from langchain_mongodb import MongoDBChatMessageHistory

from dotenv import load_dotenv

# load the environment with security details
load_dotenv()

app = Flask(__name__)
MONGO_DB_URI = os.environ.get("MONGO_DB_URI")
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GOOGLE_CSE_ID = os.environ.get("GOOGLE_CSE_ID")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

print("App Initialized.")


contextPrompt = """Answer the following question as best as you can. 
Speak compassionately.
You should try to give some type of diagnosis and suggest remedies.
You have access to the following tools: Search
Your answer should be less than 2500 characters. The question is: 
"""

# Set up the base template
template = """Answer the following questions as best you can, but speaking as compassionate medical professional. 
Ensure you list some possible diagnosis, give advice on remedies, and give a warning as well.
If the user asks about anything other than medical related topics, please politely remind them that you are only there for medical suggestions and advice.
You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin! Remember to answer as a compassionate and nice medical professional when giving your final answer.

Question: {input}
{agent_scratchpad}"""


contextOutput = """
Hello, My name is Felix! I'm here to provide you with information and guidance on general health and wellness. 
Please keep in mind that I am not a medical professional, and the information I provide is for 
informational purposes only. 
If you have any specific medical concerns or conditions, 
it's important to consult with a qualified healthcare professional for personalized advice and treatment. 
How can I assist you today?
"""

# MongoDB test connection
uri = MONGO_DB_URI
# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi("1"))
# Send a ping to confirm a successful connection with MongoDB
try:
    client.admin.command("ping")
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

search = GoogleSearchAPIWrapper()


# specifying the llm model that will be used
llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY, temperature=0)

# tools the langchain agent has access to
tools = [
    Tool(
        name="google_search",
        func=search.run,
        description="Search Google for results"
    )
]

tool_names = [tool.name for tool in tools]  # check if this can be simplified


# setting up a prompt template
class CustomPromptTemplate(StringPromptTemplate):
    # the template to use
    template: str
    # the list of tools available
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        """
        This retrieves the intermediate steps of the LangChain process (AgentAction, Observation) as tuples and formats
        the thought process of the agent.
        :param kwargs:
        :return:
        """
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation} \nThought: "
        # set the agent scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # create tools variable
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # create tool names
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)


# this is the prompt that the LangChain agent will be fed. It includes the thought chain process
prompt = PromptTemplate.from_template(template)


class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> dict:
        # Initialize empty dictionary to store parsed information
        parsed_output = {
            'diagnosis': '',
            'warning': '',
            'symptoms': '',
            'advice': ''
        }

        # Split the LLM output into lines for parsing
        lines = llm_output.strip().split('\n')

        # Iterate through each line to identify and parse relevant information
        for line in lines:
            if line.startswith('Diagnosis:'):
                parsed_output['diagnosis'] = line[len('Diagnosis:'):].strip()
            elif line.startswith('Warning:'):
                parsed_output['warning'] = line[len('Warning:'):].strip()
            elif line.startswith('Symptoms:'):
                parsed_output['symptoms'] = line[len('Symptoms:'):].strip()
            elif line.startswith('Advice:'):
                parsed_output['advice'] = line[len('Advice:'):].strip()

        return parsed_output


parser = CustomOutputParser()

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# LLM chain w LLM model and prompt
llm_chain = LLMChain(llm=llm, prompt=prompt)

agent = create_react_agent(llm, tools, prompt)

# Twilio
account_sid = TWILIO_ACCOUNT_SID  # Twilio Account SID
auth_token = TWILIO_AUTH_TOKEN  # Twilio Account Auth Token
twilio_client = Client(account_sid, auth_token)


def sendMessage(body_message, phone_number):
    message = twilio_client.messages.create(
        from_="whatsapp:+14155238886",  # With country code
        body=body_message,
        to="whatsapp:+" + phone_number,
    )
    print(message)  # debugging purposes


@app.route("/bot", methods=["POST"])
def bot():
    """
    the main function of the app. logs the chat history with MongoDB and sends a message to the user with appropriate
    medical advice.
    :return:
    """
    # incoming message from the user
    incoming_msg = request.values["Body"]

    # gets the phone number of the message sender
    phone_number = request.values["WaId"]

    if incoming_msg:
        message_history = MongoDBChatMessageHistory(
            # collection name can be phone number
            # session id will be the session id
            connection_string=MONGO_DB_URI,
            session_id="1",
            collection_name=phone_number,
            database_name="MedAdvice",
        )
        memory = ConversationBufferMemory(
            ai_prefix="AI Assistant",
            chat_memory=message_history,
            memory_key="chat_history",
            return_messages=True,
        )
        if incoming_msg == "forget everything":
            message_history.clear()
            sendMessage("Your conversation has been reset!", phone_number)
            memory.save_context({"input": contextPrompt}, {"output": contextOutput})
            sendMessage(contextOutput, phone_number)
            return ""
        if not message_history.messages:
            # if this is a new conversation, bring in the context
            memory.save_context({"input": contextPrompt}, {"output": contextOutput})
            sendMessage(contextOutput, phone_number)

        print("Message Received: " + incoming_msg)
        print("number is:" + phone_number)

        agent_executor = AgentExecutor.from_agent_and_tools(agent=agent,
                                                            tools=tools,
                                                            verbose=True,
                                                            handle_parsing_errors=True
                                                            )

        answer = "An error occurred. Please try again."
        start_time = time.time()
        tries = 0
        max_tries = 5
        while answer == "An error occurred. Please try again." or answer == "":
            try:
                if tries > max_tries:
                    answer = "Sorry, I am taking too long to respond. Please try again later."
                    break
                tries += 1
                response_data = agent_executor.invoke(
                    {"input": incoming_msg}
                    )
                if 'output' in response_data:
                    answer = response_data['output']
                else:
                    answer = "Sorry, I couldn't process your request. Please try again."
            except ValueError as e:
                answer = str(e)
                if not answer.startswith("Could not parse LLM output: `"):
                    raise e
                answer = answer.removeprefix(
                    "Could not parse LLM output: `"
                ).removesuffix("`")
                message_history.add_user_message(incoming_msg)
                message_history.add_ai_message(answer)
            except Exception as e:
                answer = str(e)  # Capture and return any error messages

            # Check if time limit has been exceeded
            if time.time() - start_time > 15:  # Set timeout to 15 seconds
                answer = (
                    "Sorry, I am taking too long to respond. Please try again later."
                )
                break
        sendMessage(answer, phone_number)
        print(f"Response from Model: {answer}")
    else:
        sendMessage("Message Cannot Be Empty!", phone_number)
        print("Message Is Empty")
    r = MessagingResponse()
    r.message("")
    return str(r)


# run the app
if __name__ == "__main__":
    app.run(port=5000)
