import asyncio
import aiohttp
import json
import os
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# Create a directory to store conversation data
os.makedirs("conversation_data", exist_ok=True)

step_prompts = {
    "GREETING": "You are a friendly AI assistant. Greet the customer warmly and ask for their name. Be polite and engaging.",
    "NEEDS_ASSESSMENT": "You are a helpful AI assistant. Ask the customer about their interests or needs. Try to understand what product or service they might be looking for.",
    "INFO_PROVISION": "You are a knowledgeable AI assistant. Provide detailed information about the product or service the customer is interested in. Ensure they understand the key features and benefits.",
    "TRANSACTION": "You are a helpful AI sales assistant. Guide the customer through the purchase process. Collect necessary details for the transaction, such as quantity, shipping address, or payment method.",
    "SUPPORT": "You are a patient AI support assistant. Listen to the customer's issue and ask for any necessary details to understand the problem fully. Offer clear and helpful solutions.",
    "FEEDBACK": "You are a courteous AI assistant. Politely ask the customer for their feedback on the product, service, or their interaction with you. Encourage honest and constructive feedback.",
    "ACCOUNT_MANAGEMENT": "You are a secure AI account manager. Help the customer with their account-related request. Ensure to maintain privacy and security protocols while assisting them.",
    "CLOSING": "You are a grateful AI assistant. Thank the customer sincerely for their time and interaction. Offer any final assistance and wish them well.",
    "ERROR_HANDLING": "You are an attentive AI troubleshooter. Carefully listen to any issues or errors the customer reports. Ask for clarification if needed and offer clear steps to resolve the problem.",
    "COMMON_STATES": "You are a versatile AI assistant. Address the customer's general inquiry or common conversation topic. Provide helpful and relevant information based on their specific question or comment."
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

def get_user_selection():
    print("Available conversation steps:")
    for i, step in enumerate(available_steps.keys(), 1):
        print(f"{i}. {step}")
    
    selections = input("Enter the numbers of the steps you want to include (comma-separated): ").split(',')
    return [list(available_steps.keys())[int(s.strip()) - 1] for s in selections]

async def call_gpt_api(prompt):
    url = os.getenv("GPT4_ENDPOINT")
    api_key = os.getenv("GPT4_API_KEY")
    if not api_key or not url:
        return "AI: Error: GPT4_API_KEY or GPT4_ENDPOINT not found in environment variables."

    headers = {
        "Content-Type": "application/json",
        "api-key": api_key
    }
    payload = {
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 150
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return f"AI: {data['choices'][0]['message']['content']}"
                else:
                    return f"AI: I apologize, but I encountered an error. Status code: {response.status}"
    except Exception as e:
        return f"AI: I apologize, but I encountered an error: {str(e)}"

def save_to_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

async def run_conversation_step(step, conversation):
    system_prompt = step_prompts[step.function]
    
    ai_response = await call_gpt_api(system_prompt)
    print(ai_response)
    conversation.append({"role": "AI", "content": ai_response})
    
    user_input = input("User: ")
    conversation.append({"role": "User", "content": user_input})
    
    return user_input

async def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    selected_steps = get_user_selection()
    conversation_sequence = [available_steps[step] for step in selected_steps]

    conversation = []

    for step in conversation_sequence:
        print(f"\nStarting {step.function}")
        await run_conversation_step(step, conversation)

    # Save the entire conversation
    conversation_filename = f"conversation_data/{timestamp}_conversation.json"
    save_to_json(conversation, conversation_filename)
    print(f"\nEntire conversation saved to {conversation_filename}")

if __name__ == "__main__":
    asyncio.run(main())
