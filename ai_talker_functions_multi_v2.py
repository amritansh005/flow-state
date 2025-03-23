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
        "Ask for their name and how they're doing today. Be polite and engaging. "
        "Continue the conversation until the customer provides {greeting_prompt}."
        "Also make sure you ask all the required information step by step."
    ),
    "NEEDS_ASSESSMENT": "You are a helpful AI assistant. Ask the customer about their interests or needs. Try to understand what product or service they might be looking for. Ask the following follow-up questions {needs_assessment_prompt} to have a clear understanding of their needs.",
    "INFO_PROVISION": "You are a knowledgeable AI assistant. Provide detailed information about the product or service the customer is interested in. Get information from {info_provision_prompt}.",
    "TRANSACTION": "You are a helpful AI sales assistant. Guide the customer through the purchase process. Collect necessary details for the transaction, such as quantity, shipping address, or payment method. Confirm each piece of information and ask if they have any questions about the process.",
    "SUPPORT": "You are a patient AI support assistant. Listen to the customer's issue and ask for any necessary details to understand the problem fully. Offer clear and helpful solutions. Follow up to ensure the solution works for them.",
    "FEEDBACK": "You are a courteous AI assistant. Politely ask the customer for their feedback on the product, service, or their interaction with you. Encourage honest and constructive feedback. Ask follow-up questions to get more detailed insights.",
    "ACCOUNT_MANAGEMENT": "You are a secure AI account manager. Help the customer with their account-related request. Ensure to maintain privacy and security protocols while assisting them. Ask for necessary information step by step.",
    "CLOSING": "You are a grateful AI assistant. Thank the customer sincerely for their time and interaction. Summarize the key points of your conversation. {closing_prompt}. Keep the conversation engaging and do everything step by step.",
    "ERROR_HANDLING": "You are an attentive AI troubleshooter. Carefully listen to any issues or errors the customer reports. Ask for clarification if needed and offer clear steps to resolve the problem. Confirm if the issue is resolved after providing solutions.",
    "COMMON_STATES": "You are a versatile AI assistant. Address the customer's general inquiry or common conversation topic. Provide helpful and relevant information based on their specific question or comment. Ask follow-up questions to ensure you've fully addressed their needs."
}

# Fallback for insert_statements if not found
#insert_statements = insert_statements or "their name and the reason for reaching out"

# Format dynamic insert only if prompt is not None
#if greeting_prompt:
    #greeting_prompt = greeting_prompt.format(insert_statements=insert_statements)


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

async def run_conversation_step(step, conversation):
    system_prompt = step_prompts[step.function]
    
    while True:
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend([{"role": "assistant" if msg["role"] == "AI" else "user", "content": msg["content"]} for msg in conversation[-5:]])
        
        ai_response = await call_gpt_api(messages)
        print(f"AI: {ai_response}")
        conversation.append({"role": "AI", "content": ai_response})
        
        user_input = input("User: ")
        conversation.append({"role": "User", "content": user_input})
        
        if "NEXT_STEP" in user_input.upper():
            break
    
    return conversation

async def main_multi():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    selected_steps = list(available_steps.keys())  # Use all steps for this example
    conversation_sequence = [available_steps[step] for step in selected_steps]

    conversation = []

    for step in conversation_sequence:
        print(f"\nStarting {step.function}")
        conversation = await run_conversation_step(step, conversation)

    # Save the entire conversation
    conversation_filename = f"conversation_data/{timestamp}_conversation_multi_v2.json"
    save_to_json(conversation, conversation_filename)
    print(f"\nEntire conversation saved to {conversation_filename}")

if __name__ == "__main__":
    asyncio.run(main_multi())
