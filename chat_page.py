
import streamlit as st
import json
from dotenv import load_dotenv
from llm_factory import LLMProviderFactory
from logging_config import llm_logger

# Load environment variables from .env file
load_dotenv()

def chat_page():
    st.title("Playground")


    # Load configurations
    with open('providers.json') as providers_file:
        providers_config = json.load(providers_file)

    # Provider selection
    provider_name = st.radio("Select a provider", ["ollama", "azure_openai"], index=["ollama", "azure_openai"].index(providers_config.get('provider', 'ollama')))

    # Model selection
    if provider_name == "ollama":
        models = providers_config.get('ollama', {}).get('models', [])
    else:
        models = providers_config.get('azure_openai', {}).get('models', [])
    selected_model = st.selectbox("Select a model", models)

    if 'system_prompt' not in st.session_state:
        st.session_state.system_prompt = ""

    if 'chat_system_prompt_prefill' in st.session_state and st.session_state.chat_system_prompt_prefill:
        st.session_state.system_prompt = st.session_state.chat_system_prompt_prefill
        st.session_state.chat_system_prompt_prefill = ""

    system_prompt = st.text_area("Enter your System Prompt here", key="system_prompt", height=150)

    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What is up?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Thinking..."):
            try:
                factory = LLMProviderFactory()
                llm_provider = factory.get_provider(provider_name)

                log_extra = {
                    "prompt_id": "chat_test",
                    "prompt_version": "1.0",
                }

                roles = {"system": st.session_state.system_prompt}

                # Get response from the LLM
                response = llm_provider.get_llm_response(
                    prompt, # In chat mode, the user input is the prompt
                    selected_model,
                    roles,
                    llm_logger,
                    log_extra
                )

                # Display assistant response in chat message container
                with st.chat_message("assistant"):
                    st.markdown(response)
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})

            except Exception as e:
                st.error(f"An error occurred: {e}")