import os
import time
import streamlit as st
from openai import OpenAI
from chat_memgpt import load_questions_from_txt, chat_loop_v1, chat_loop_v2

# Load API key securely
api_key = st.secrets["auth_token"]
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", api_key))

# Function to rephrase the sentence
def rephraser_model(content, model="gpt-4o-mini"):
    try:
        messages = [
            {"role": "system", "content": """
            You are a professional paragraph rephraser. Your task is to take a given paragraph, and produce a new paragraph that is easier for language models to understand.
            Rephrase the paragraph to enhance the language model's performance.
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
            Your task is to take a given paragraph, then applying the chain-of-thought technique to that paragraph.
            Return the updated paragraph with chain-of-thought technique.
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
def Questioner(content, model="gpt-4o"):
    try:
        messages = [
            {"role": "system", "content": """
            In role of Student/Child in the provided context.
            Prepare the answers that you will respond to the LLM (Pika) to interact with it, the answers will be based on the provided context.
            Answers are in Vietnamese.
            The output must be in list format only, it not be the string, and don't contain any other information or special characters.
            You must follow the output format below:
                - <put "..." here>
                - <put first answer here>
                - <put second answer here>
                - <put third answer here>
                - <put fourth answer here>
                - <put fifth answer here>
            The following is the provided context:
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
            I will provide you the dialogues that contains the responses from LLM, in which LLM is not User. 
            Please evaluate the dialogues to check if the LLM follow the requirements during the conversation or not. 
            Please also decide the metrics to evaluate and output the score in dict format. The scale of score is 5. 
            Only return the dictionary of score, and don't return any other information.
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
        tab1, tab2, tab3 = st.tabs(["APO Result", "Original Prompt","New Prompt"])

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
            st.subheader("Original Prompt Evaluation Score")
            with st.spinner("Generating test questions..."):
                questions_list_str = Questioner(input_sentence)

            if questions_list_str:
                questions_list = [q.strip() for q in questions_list_str.split("- ") if q.strip()]

                st.subheader("Generated Questions for Evaluation")
                st.write(questions_list)

                with st.spinner("Simulating conversation..."):
                    conversation_result = chat_loop_v2(questions_list, input_sentence)

                st.subheader("Conversation Log")
                st.write(conversation_result)

                full_context_for_Judger = (
                    "The following is the requirements for the LLM:\n"
                    + input_sentence + "\n\n"
                    + "The following is the content of the conversation between LLM and the user:\n"
                    + conversation_result
                )

                with st.spinner("Evaluating performance..."):
                    eval_score = Judger(full_context_for_Judger)

                if eval_score:
                    st.subheader("Evaluation Score")
                    st.write(f"**Score:** {eval_score}")
        with tab3:
            st.subheader("New Prompt Evaluation Score")
            with st.spinner("Generating test questions..."):
                questions_list_str = Questioner(final_result)

            if questions_list_str:
                questions_list = [q.strip() for q in questions_list_str.split("- ") if q.strip()]

                st.subheader("Generated Questions for Evaluation")
                st.write(questions_list)

                with st.spinner("Simulating conversation..."):
                    conversation_result = chat_loop_v2(questions_list, final_result)

                st.subheader("Conversation Log")
                st.write(conversation_result)

                full_context_for_Judger = (
                    "The following is the requirements for the LLM:\n"
                    + input_sentence + "\n\n"
                    + "The following is the content of the conversation between LLM and the user:\n"
                    + conversation_result
                )

                with st.spinner("Evaluating performance..."):
                    eval_score = Judger(full_context_for_Judger)

                if eval_score:
                    st.subheader("Evaluation Score")
                    st.write(f"**Score:** {eval_score}")
    else:
        st.warning("Please enter a prompt to process.")
