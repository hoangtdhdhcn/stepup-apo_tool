import os
import time
import streamlit as st
from openai import OpenAI
import json

os.environ['OPENAI_API_KEY'] = ''

api_key = st.secrets["auth_token"]
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", api_key))


# Function to rephrase the sentence
def rephraser_model(content, model="gpt-4o-mini"):
    openai_client = client

    messages = [
        {
            "role": "system",
            "content": """You are a professional sentence rephraser. Your task is to take a given sentence, and produce a new sentence that is easier for language models to understand.
            Rephrase the sentence to enhance the language model's performance."""
        },
        {"role": "user", "content": content}
    ]

    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message.content
    time.sleep(5)  # Add delay between requests
    return content


# Function to apply the chain-of-thought
def CoT_model(content, model="gpt-4o-mini"):
    openai_client = client

    messages = [
        {
            "role": "system",
            "content": """Your task is to take a given sentence, and produce a new sentence that can help the language models improve their reasoning.
            Keep all the content of the given sentence, apply the chain-of-thought technique to the this sentence.
            The chain-of-thought technique here is very simple, you only need to add this sentence 'Let think step-by-step' to the suitable position to help improve the reasoning of language models."""
        },
        {"role": "user", "content": content}
    ]

    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message.content
    time.sleep(5)  # Add delay between requests
    return content


# Streamlit UI code
st.title('Prompt Enhancer')

# User input section
input_sentence = st.text_area("Enter a prompt for processing:")

if st.button("Start"):
    if input_sentence:
        # Rephrase the input sentence
        with st.spinner("Processing first step..."):
            rephrased_sentence = rephraser_model(input_sentence)

        # Apply chain-of-thought to the rephrased sentence
        with st.spinner("Processing second step..."):
            final_result = CoT_model(rephrased_sentence)

        # Display results in tabs
        tab1, tab2 = st.tabs(["APO Result", ""])

        with tab1:
            st.subheader("APO Result")
            st.write(final_result)

        # Option to save the final result
        # if st.button("Save to File"):
        #     output_file_path = "output.txt"
        #     with open(output_file_path, 'w', encoding='utf-8') as outfile:
        #         outfile.write(final_result)
        #     st.success(f"Result saved to {output_file_path}")
    else:
        st.warning("Please enter a prompt to process.")
