
import streamlit as st
import ollama
import json
import time
from logging_config import llm_logger

# Load configuration
with open('''config.json''') as config_file:
    config = json.load(config_file)

st.title("AI Prompt Builder")

st.write("Enter your initial system prompt below, and the AI will help you refine it.")

# Model selection
selected_model = st.selectbox("Select a model", config['''models'''])

user_prompt = st.text_area("Your System Prompt", height=150)

# Function to interact with Ollama
def get_refined_prompt_from_llm(initial_prompt, model):
    if initial_prompt:
        start_time = time.time()
        try:
            # Assumes Ollama is running on http://localhost:11434
            response = ollama.chat(
                model=model,
                messages=[
                    {
                        '''role''': '''system''',
                        '''content''': config['''system_prompt''']['''content'''],
                    },
                    {
                        '''role''': '''user''',
                        '''content''': initial_prompt,
                    },
                ],
            )
            latency = time.time() - start_time
            llm_logger.info(
                "LLM call successful",
                extra={
                    "prompt_id": config['''system_prompt''']['''id'''],
                    "prompt_version": config['''system_prompt''']['''version'''],
                    "model_used": model,
                    "tokens_used": response.get('''eval_count'''),
                    "latency": latency,
                    "outcome": "success",
                },
            )
            return response['''message''']['''content''']
        except Exception as e:
            latency = time.time() - start_time
            llm_logger.error(
                "LLM call failed",
                extra={
                    "prompt_id": config['''system_prompt''']['''id'''],
                    "prompt_version": config['''system_prompt''']['''version'''],
                    "model_used": model,
                    "latency": latency,
                    "outcome": "error",
                    "error_message": str(e),
                },
            )
            return f"An error occurred while communicating with Ollama: {e}"
    return "Please enter a prompt to get a refined version."

if st.button("Refine Prompt"):
    with st.spinner("Refining prompt..."):
        refined_prompt = get_refined_prompt_from_llm(user_prompt, selected_model)
        st.subheader("Refined Prompt")
        st.markdown(f"'''\n{refined_prompt}\n'''")
