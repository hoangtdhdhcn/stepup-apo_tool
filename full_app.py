import os
import time
import streamlit as st
from openai import OpenAI
from chat_memgpt import load_questions_from_txt, chat_loop_v1

# Load API key securely
api_key = st.secrets["auth_token"]
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", api_key))

# Function to rephrase the sentence
def rephraser_model(content, model="gpt-4o-mini"):
    try:
        messages = [
            {"role": "system", "content": """
            You are a professional sentence rephraser. Your task is to take a given sentence, and produce a new sentence that is easier for language models to understand.
            Rephrase the sentence to enhance the language model's performance.
            """},
            {"role": "user", "content": content}
        ]

        response = openai_client.chat.completions.create(
            model=model,
            messages=messages,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error in rephrasing: {str(e)}")
        return None

# Function to apply chain-of-thought
def CoT_model(content, model="gpt-4o-mini"):
    try:
        messages = [
            {"role": "system", "content": """
            Your task is to take a given sentence, and produce a new sentence that can help the language models improve their reasoning.
            Keep all the content of the given sentence, apply the chain-of-thought technique to the this sentence.
            The chain-of-thought technique here is very simple, you only need to add this sentence 'Let think step-by-step' to the suitable position to help improve the reasoning of language models.
            """},
            {"role": "user", "content": content}
        ]

        response = openai_client.chat.completions.create(
            model=model,
            messages=messages,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error in CoT processing: {str(e)}")
        return None

# Function to generate questions based on the prompt
def Questioner(content, model="gpt-4o-mini"):
    try:
        messages = [
            {"role": "system", "content": """
            In role of user, imagine you will be in the conversation with the Pika in the provided context. 
            Prepare some questions that you will ask the Pika to test its role-play capability.  
            Put all of your questions in the list.
            Each part only need one question. Questions are in Vietnamese.
            The output must be in list format only, it not be the string, and don't contain any other information or special characters.
            """},
            {"role": "user", "content": content}
        ]

        response = openai_client.chat.completions.create(
            model=model,
            messages=messages,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error in question generation: {str(e)}")
        return None

# Function to evaluate the chat conversation
def Judger(content, model="gpt-4o-mini"):
    try:
        messages = [
            {"role": "system", "content": """
            You are the LLM evaluator/judger. Based on the context below, which is the requirements to access the performance of the LLM. 
            I will provide you the dialogues that contains the responses from LLM, in which LLM is Pika. 
            Please evaluate the dialogues to check if the LLM follow the requirements during the conversation or not. 
            Please also decide the metrics to evaluate and output the score in dict format, the scale for each metric is 5. Only return the dictionary of score, and don't return any other information.
            """},
            {"role": "user", "content": content}
        ]

        response = openai_client.chat.completions.create(
            model=model,
            messages=messages,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error in evaluation: {str(e)}")
        return None

# Streamlit UI
st.title('Prompt Enhancer & Evaluator')

# User input
input_sentence = st.text_area("Enter a prompt for processing:")

if st.button("Start"):
    if input_sentence:
        # Create two tabs
        tab1, tab2 = st.tabs(["APO Result", "Evaluation Score"])

        with tab1:
            st.subheader("Enhanced Prompt")

            with st.spinner("Processing step 1: Optimizing Long Prompt..."):
                rephrased_sentence = rephraser_model(input_sentence)

            if rephrased_sentence:
                with st.spinner("Processing step 2: Applying Chain-of-Thought..."):
                    final_result = CoT_model(rephrased_sentence)

                if final_result:
                    st.write(final_result)

        with tab2:
            st.subheader("Prompt Evaluation Score")
            with st.spinner("Generating test questions..."):
                questions_list_str = Questioner(input_sentence)

            if questions_list_str:
                questions_list = [q.strip() for q in questions_list_str.split("- ") if q.strip()]

                # st.subheader("Generated Questions for Evaluation")
                # st.write(questions_list)

                with st.spinner("Simulating conversation..."):
                    conversation_result = chat_loop_v1(questions_list)

                # st.subheader("Conversation Log")
                # st.write(conversation_result)

                full_context_for_Judger = (
                    "The following is the requirements for the LLM:\n"
                    + input_sentence + "\n\n"
                    + "The following is the content of the conversation between Pika and the user:\n"
                    + conversation_result
                )

                with st.spinner("Evaluating performance..."):
                    eval_score = Judger(full_context_for_Judger)

                if eval_score:
                    st.subheader("Evaluation Score")
                    st.write(f"**Score:** {eval_score}")
    else:
        st.warning("Please enter a prompt to process.")
