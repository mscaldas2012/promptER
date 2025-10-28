
import streamlit as st
import json
import time
import os
from dotenv import load_dotenv
from logging_config import llm_logger
from llm_factory import LLMProviderFactory

# Load environment variables from .env file
load_dotenv()

# Load configuration
with open('''config.json''') as config_file:
    config = json.load(config_file)

st.title("AI Prompt Builder")

st.write("Enter your initial system prompt below, and the AI will help you refine it.")

# Provider selection
provider_name = st.radio("Select a provider", ["ollama", "azure_openai"], index=["ollama", "azure_openai"].index(config['''provider''']))

# Model selection
if provider_name == "ollama":
    models = config['''ollama''']['''models''']
else:
    models = config['''azure_openai''']['''models''']
selected_model = st.selectbox("Select a model", models)

user_prompt = st.text_area("Your System Prompt", height=150)

if st.button("Refine Prompt"):
    if not user_prompt:
        st.warning("Please enter a prompt to get a refined version.")
    else:
        with st.spinner("Refining prompt..."):
            try:
                factory = LLMProviderFactory()
                llm_provider = factory.get_provider(provider_name)

                log_extra = {
                    "prompt_id": config['''system_prompt''']['''id'''],
                    "prompt_version": config['''system_prompt''']['''version'''],
                }

                refined_prompt = llm_provider.get_refined_prompt(
                    user_prompt,
                    selected_model,
                    config['''system_prompt''']['''content'''],
                    llm_logger,
                    log_extra
                )
                st.subheader("Refined Prompt")
                st.markdown(f"'''\n{refined_prompt}\n'''")
            except Exception as e:
                st.error(f"An error occurred: {e}")
