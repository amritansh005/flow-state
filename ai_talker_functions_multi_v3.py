import asyncio
import aiohttp
import json
import os
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# Fetch variables
aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_REGION")
dynamodb_table_name = os.getenv("DYNAMODB_TABLE")

required_env_vars = [aws_access_key_id, aws_secret_access_key, aws_region, dynamodb_table_name]
if not all(required_env_vars):
    raise EnvironmentError("Missing one or more AWS-related environment variables in .env file.")


# Initialize DynamoDB client using credentials from .env
dynamodb = boto3.resource(
    'dynamodb',
    region_name=aws_region,
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key
)

# Access the table
table = dynamodb.Table(dynamodb_table_name)

org_id = os.getenv("ORG_ID")
use_case = os.getenv("USE_CASE")
bot_name = os.getenv("BOT_NAME")

if not all([org_id, use_case, bot_name]):
    raise ValueError("ORG_ID, USE_CASE, or BOT_NAME not set in .env file")

def get_prompt_from_dynamodb(field_name, org_id, use_case, bot_name):
    try:
        response = table.scan(
            FilterExpression=(
                boto3.dynamodb.conditions.Attr('OrgId').eq(org_id) &
                boto3.dynamodb.conditions.Attr('UseCase').eq(use_case) &
                boto3.dynamodb.conditions.Attr('BotName').eq(bot_name)
            )
        )
        items = response.get('Items', [])
        if items:
            return items[0].get(field_name)
        else:
            print(f"No matching item found for {field_name}.")
            return None
    except ClientError as e:
        print(f"Error retrieving {field_name} from DynamoDB: {e.response['Error']['Message']}")
        return None


# === Fetch prompts for each step ===
greeting_prompt = get_prompt_from_dynamodb('GREETING', org_id, use_case, bot_name)
account_management_prompt = get_prompt_from_dynamodb('ACCOUNT_MANAGEMENT', org_id, use_case, bot_name)
closing_prompt = get_prompt_from_dynamodb('CLOSING', org_id, use_case, bot_name)
common_states_prompt = get_prompt_from_dynamodb('COMMON_STATES', org_id, use_case, bot_name)
connections_prompt = get_prompt_from_dynamodb('Connections', org_id, use_case, bot_name)
error_handling_prompt = get_prompt_from_dynamodb('ERROR_HANDLING', org_id, use_case, bot_name)
feedback_prompt = get_prompt_from_dynamodb('FEEDBACK', org_id, use_case, bot_name)
info_provision_prompt = get_prompt_from_dynamodb('INFO_PROVISION', org_id, use_case, bot_name)
needs_assessment_prompt = get_prompt_from_dynamodb('NEEDS_ASSESSMENT', org_id, use_case, bot_name)
support_prompt = get_prompt_from_dynamodb('SUPPORT', org_id, use_case, bot_name)
transaction_prompt = get_prompt_from_dynamodb('TRANSACTION', org_id, use_case, bot_name)

# Create a directory to store conversation data
os.makedirs("conversation_data", exist_ok=True)

# Define base step prompts (with a placeholder)
step_prompts = {
    "GREETING": (
        "You are a friendly AI assistant. Greet the customer warmly and engage in a brief conversation to make them feel welcome. "
        f"Then, ask the customer for the following details, one by one, in a natural tone: {greeting_prompt}. "
        "Make sure to wait for the user's response to each question before moving to the next. "
        "Do not move to the next step until all of these details are collected. "
        "Keep the conversation engaging and friendly throughout."
    ),
    "NEEDS_ASSESSMENT": (
        "You are a helpful AI assistant. Ask the customer about their interests or needs. "
        "Try to understand what product or service they might be looking for. "
        f"Ask the following follow-up questions to gather detailed insights: {needs_assessment_prompt}. "
        "Make sure to explore preferences, budget, and any specific requirements they might have."
    ),
    "INFO_PROVISION": (
        "You are a knowledgeable AI assistant. Provide detailed information about the product or service the customer is interested in. "
        f"Pull the required information from the following source: {info_provision_prompt}. "
        "Ensure the customer understands the key features and benefits, and ask if they have any questions."
    ),
    "TRANSACTION": (
        "You are a helpful AI sales assistant. Guide the customer through the purchase process. "
        "Collect necessary details for the transaction, such as quantity, shipping address, and payment method. "
        f"Confirm each piece of information clearly and make sure the customer understands the process. {transaction_prompt}"
    ),
    "SUPPORT": (
        "You are a patient AI support assistant. Listen to the customer's issue carefully and ask for any necessary details to understand it fully. "
        "Offer clear and helpful solutions based on the problem described. "
        f"Follow up to ensure the solution has resolved their issue. {support_prompt}"
    ),
    "FEEDBACK": (
        "You are a courteous AI assistant. Politely ask the customer for their feedback on the product, service, or their interaction with you. "
        "Encourage honest and constructive responses. "
        f"Use this prompt to guide the conversation: {feedback_prompt}. "
        "Ask follow-up questions to get more detailed insights."
    ),
    "ACCOUNT_MANAGEMENT": (
        "You are a secure AI account manager. Help the customer with their account-related request. "
        "Ensure to maintain privacy and security protocols while assisting. "
        f"Gather the following account information step by step: {account_management_prompt}."
    ),
    "CLOSING": (
        "You are a grateful AI assistant. Thank the customer sincerely for their time and interaction. "
        "Summarize the key points of your conversation. "
        f"{closing_prompt} "
        "Keep the tone warm and engaging, and make sure everything is handled step by step before closing."
    ),
    "ERROR_HANDLING": (
        "You are an attentive AI troubleshooter. Carefully listen to any issues or errors the customer reports. "
        "Ask for clarification if needed and offer step-by-step instructions to resolve the issue. "
        "Ensure to confirm whether the problem is fully resolved afterward. "
        f"Use this for guidance: {error_handling_prompt}."
    ),
    "COMMON_STATES": (
        "You are a versatile AI assistant. Handle general customer inquiries or common conversation topics in a helpful and engaging way. "
        f"Provide information or guidance based on: {common_states_prompt}. "
        "Always ask follow-up questions to ensure the customer's needs are fully addressed."
    ),
    "Connections": (
        "You are an intelligent assistant managing connections between services, systems, or people. "
        "Ask about connection-specific details such as linked accounts, integrations, or contacts. "
        f"Use this context to assist: {connections_prompt}."
    )
}

class ConversationStep:
    def __init__(self, function):
        self.function = function

available_steps = {
    "GREETING": ConversationStep(function="GREETING"),
    "NEEDS_ASSESSMENT": ConversationStep(function="NEEDS_ASSESSMENT"),
    "INFO_PROVISION": ConversationStep(function="INFO_PROVISION"),
    "TRANSACTION": ConversationStep(function="TRANSACTION"),
    "SUPPORT": ConversationStep(function="SUPPORT"),
    "FEEDBACK": ConversationStep(function="FEEDBACK"),
    "ACCOUNT_MANAGEMENT": ConversationStep(function="ACCOUNT_MANAGEMENT"),
    "CLOSING": ConversationStep(function="CLOSING"),
    "ERROR_HANDLING": ConversationStep(function="ERROR_HANDLING"),
    "COMMON_STATES": ConversationStep(function="COMMON_STATES")
}

async def call_gpt_api(messages):
    url = os.getenv("GPT4_ENDPOINT")
    api_key = os.getenv("GPT4_API_KEY")
    if not api_key or not url:
        return "AI: Error: GPT4_API_KEY or GPT4_ENDPOINT not found in environment variables."

    headers = {
        "Content-Type": "application/json",
        "api-key": api_key
    }
    payload = {
        "messages": messages,
        "max_tokens": 150,
        "temperature": 0.7,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "top_p": 0.95,
        "stop": None
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return data['choices'][0]['message']['content']
                else:
                    error_message = await response.text()
                    return f"AI: I apologize, but I encountered an error. Status code: {response.status}. Error: {error_message}"
    except Exception as e:
        return f"AI: I apologize, but I encountered an error: {str(e)}"

def save_to_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

async def supervisor(conversation, current_step, conversation_sequence):
    supervisor_prompt = f"""
    You are a conversation supervisor. Your task is to determine if the current conversation step ({current_step.function}) has been completed based on the conversation history.
    If the step is complete, decide which step to move to next.
    If the step is not complete, indicate that we should stay on the current step.
    
    Current step: {current_step.function}
    Step requirements: {step_prompts[current_step.function]}
    
    Conversation history (last 5 messages):
    {json.dumps(conversation[-5:], indent=2)}
    
    Please respond with one of the following:
    1. "STAY" if the current step is not complete
    2. The name of the next step (e.g., "NEEDS_ASSESSMENT") if the current step is complete
    3. "COMPLETE" if all steps have been completed
    """

    messages = [{"role": "system", "content": supervisor_prompt}]
    supervisor_response = await call_gpt_api(messages)
    
    supervisor_decision = supervisor_response.strip().upper()
    
    if supervisor_decision == "STAY":
        return current_step
    elif supervisor_decision == "COMPLETE":
        return None
    else:
        next_step_index = conversation_sequence.index(current_step) + 1
        if next_step_index < len(conversation_sequence):
            return conversation_sequence[next_step_index]
        else:
            return None

async def run_conversation(conversation_sequence):
    conversation = []
    current_step = conversation_sequence[0]

    while current_step:
        print(f"\nCurrent step: {current_step.function}")
        system_prompt = step_prompts[current_step.function]
        
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend([{"role": "assistant" if msg["role"] == "AI" else "user", "content": msg["content"]} for msg in conversation[-5:]])
        
        ai_response = await call_gpt_api(messages)
        print(f"AI: {ai_response}")
        conversation.append({"role": "AI", "content": ai_response})
        
        user_input = input("User: ")
        conversation.append({"role": "User", "content": user_input})
        
        # Use the supervisor to determine the next step
        next_step = await supervisor(conversation, current_step, conversation_sequence)
        if next_step != current_step:
            print(f"Moving to next step: {next_step.function if next_step else 'Conversation complete'}")
        current_step = next_step

    return conversation

async def main_multi(conversation_sequence):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Convert the sequence to ConversationStep objects
    conversation_steps = [available_steps[step] for step in conversation_sequence if step in available_steps]

    if not conversation_steps:
        print("Error: No valid steps in the conversation sequence.")
        return

    conversation = await run_conversation(conversation_steps)

    # Save the entire conversation
    conversation_filename = f"conversation_data/{timestamp}_conversation_multi_v3.json"
    save_to_json(conversation, conversation_filename)
    print(f"\nEntire conversation saved to {conversation_filename}")

if __name__ == "__main__":
    asyncio.run(main_multi())
