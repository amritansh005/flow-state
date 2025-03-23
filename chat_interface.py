import asyncio
from ai_talker_functions_multi_v3 import (
    get_prompt_from_dynamodb, available_steps, step_prompts, call_gpt_api, save_to_json,
    supervisor, ConversationStep
)
import json
from datetime import datetime
import os

async def chat_interface():
    print("Welcome to the AI Chat Interface!")
    
    # Fetch the 'connection' field
    org_id = os.getenv("ORG_ID")
    use_case = os.getenv("USE_CASE")
    bot_name = os.getenv("BOT_NAME")
    connection = get_prompt_from_dynamodb('connection', org_id, use_case, bot_name)

    if not connection:
        print("Error: Unable to fetch 'connection' field from DynamoDB.")
        return

    try:
        connection_dict = json.loads(connection)
        conversation_sequence = connection_dict.get('steps', [])
    except json.JSONDecodeError:
        print("Error: Unable to parse 'connection' field as JSON.")
        return

    conversation_steps = [available_steps[step] for step in conversation_sequence if step in available_steps]

    if not conversation_steps:
        print("Error: No valid steps in the conversation sequence.")
        return

    conversation = []
    current_step = conversation_steps[0]

    while current_step:
        print(f"\nCurrent step: {current_step.function}")
        system_prompt = step_prompts[current_step.function]
        
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend([{"role": "assistant" if msg["role"] == "AI" else "user", "content": msg["content"]} for msg in conversation[-5:]])
        
        ai_response = await call_gpt_api(messages)
        print(f"AI: {ai_response}")
        conversation.append({"role": "AI", "content": ai_response})
        
        user_input = input("You: ")
        conversation.append({"role": "User", "content": user_input})
        
        if user_input.lower() in ['exit', 'quit', 'bye']:
            break

        # Use the supervisor to determine the next step
        next_step = await supervisor(conversation, current_step, conversation_steps)
        if next_step != current_step:
            print(f"Moving to next step: {next_step.function if next_step else 'Conversation complete'}")
        current_step = next_step

    print("\nThank you for chatting!")

    # Save the conversation
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    conversation_filename = f"conversation_data/{timestamp}_chat_interface_conversation.json"
    save_to_json(conversation, conversation_filename)
    print(f"\nConversation saved to {conversation_filename}")

if __name__ == "__main__":
    asyncio.run(chat_interface())
