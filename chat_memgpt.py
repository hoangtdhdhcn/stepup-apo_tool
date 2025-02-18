import os
import time
from openai import OpenAI
import prompt_utils
from long_term_memory_manager import LongTermMemoryManager

api_key = st.secrets["auth_token"]
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", api_key))

# Display current date and time
curr_date, day_of_week, curr_time = prompt_utils.get_current_time()
print(f'-------------------------------\n{curr_date}\n{day_of_week}\n{curr_time}\n-------------------------------')

user_name = '20250215'
chatbot_name = 'Pika'       # default is Assistant
memory_folderename = 'memories_' + user_name

# Memory related params
pre_fetch_from_memory = True
post_fetch_from_memory = False
num_neighbors = 2
min_similarity = 0.2

# Model related params
temperature = 0.7

# Token management
max_num_tries = 4
max_tokens_to_generate_per_message = 320
context_length_hard_limit = 4096
context_length_limit = context_length_hard_limit - max_tokens_to_generate_per_message
system_prompt_tokens_budget = context_length_hard_limit - 768

# Instantiate long term memory manager
os.makedirs(memory_folderename, exist_ok=True)
memory_manager = LongTermMemoryManager(memory_folderename, prompt_utils.get_current_time())

# Initialize conversation history
curr_conversation_history = []

#%% Function to call ChatGPT

def send_query_to_chatgpt(chatgpt_query, max_num_tries=3):
    for i in range(max_num_tries):
        try:
            
            # Ensure that chatgpt_query is correctly formatted
            if not isinstance(chatgpt_query, list) or not all(isinstance(msg, dict) for msg in chatgpt_query):
                raise ValueError("chatgpt_query must be a list of dictionaries")

            completion = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=chatgpt_query, 
                temperature=temperature, 
                max_tokens=max_tokens_to_generate_per_message
            )

            # Corrected response parsing
            response_string = completion.choices[0].message.content  # Correct access
            return response_string  # Return only the response string

        except ValueError as ve:
            print(f"Invalid input format: {ve}")
            return None
        except Exception as e:
            print(f"ChatGPT failed. Retrying... ({e})")
            time.sleep(3 * (i + 1))

    return None  # Return None if all attempts fail

def load_questions_from_txt(txt_path):
    questions_list = []
    try:
        with open(txt_path, "r", encoding="utf-8") as file:
            questions_list = [line.strip() for line in file]
    except Exception as e:
        print(f"Error loading questions from file: {e}")
    return questions_list


#%% Interaction loop

def chat_loop(question_list, output_file="conversation.txt"):
    global curr_conversation_history
    
    with open(output_file, "w", encoding="utf-8") as file:  # Open file in write mode
        for question in question_list:
            user_input = question

            if user_input.lower() in ["exit", "quit"]:
                break

            user_prompt = {"role": "user", "content": user_input}
            chatgpt_query = curr_conversation_history + [user_prompt]  

            response_string = send_query_to_chatgpt(chatgpt_query)

            if response_string:
                # Save conversation to file instead of printing
                file.write(f"User: {user_input}\n")
                file.write(f"{chatbot_name}: {response_string}\n\n")

                # Update conversation history
                curr_conversation_history.append(user_prompt)
                curr_conversation_history.append({"role": "assistant", "content": response_string})
            else:
                file.write(f"User: {user_input}\n")
                file.write("Error: Unable to retrieve response.\n\n")

def chat_loop_v1(question_list):
    global curr_conversation_history
    conversation_log = []  # Store conversation as a list of strings

    for question in question_list:
        user_input = question

        if user_input.lower() in ["exit", "quit"]:
            break

        user_prompt = {"role": "user", "content": user_input}
        chatgpt_query = curr_conversation_history + [user_prompt]  

        response_string = send_query_to_chatgpt(chatgpt_query)

        if response_string:
            # Format conversation entry
            entry = f"User: {user_input}\n{chatbot_name}: {response_string}\n\n"

            # Store conversation in memory
            conversation_log.append(entry)

            # Update conversation history
            curr_conversation_history.append(user_prompt)
            curr_conversation_history.append({"role": "assistant", "content": response_string})
        else:
            error_entry = f"User: {user_input}\nError: Unable to retrieve response.\n\n"
            conversation_log.append(error_entry)

    return "".join(conversation_log) 


# if __name__ == "__main__":
    # questions_txt_path = r"questions.txt"
    # questions_list = load_questions_from_txt(questions_txt_path)
    # curr_conversation_history = [system_prompt]
    # chat_loop(questions_list)
