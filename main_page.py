
import streamlit as st
import json
import yaml
from dotenv import load_dotenv
from llm_factory import LLMProviderFactory
from logging_config import llm_logger

# Load environment variables from .env file
load_dotenv()

PROMPT_USE_CASE = "system_prompt_generator"
PROMPT_MODEL_NAME = "default"

def main_page():
    st.title("AI Prompt Builder")

    st.write("Enter your initial system prompt below, and the AI will help you refine it.")

    # Load configurations
    with open('providers.json') as providers_file:
        providers_config = json.load(providers_file)
    with open('config.yaml') as config_file:
        config_data = yaml.safe_load(config_file)

    prompt_config = config_data.get('prompts', {}).get(PROMPT_USE_CASE)
    if not prompt_config:
        st.error(f"Prompt configuration '{PROMPT_USE_CASE}' not found in config.yaml.")
        st.stop()

    prompt_model_config = prompt_config.get('models', {}).get(PROMPT_MODEL_NAME)
    if not prompt_model_config:
        st.error(f"Model configuration '{PROMPT_MODEL_NAME}' not found for prompt '{PROMPT_USE_CASE}'.")
        st.stop()

    # Provider selection
    provider_name = st.radio("Select a provider", ["ollama", "azure_openai"], index=["ollama", "azure_openai"].index(providers_config['provider']))

    # Model selection
    if provider_name == "ollama":
        models = providers_config['ollama']['models']
    else:
        models = providers_config['azure_openai']['models']
    selected_model = st.selectbox("Select a model", models)

    if 'refiner_user_prompt' not in st.session_state:
        st.session_state.refiner_user_prompt = ""

    user_prompt = st.text_area("Your System Prompt", key="refiner_user_prompt", height=150)

    if 'llm_response' not in st.session_state:
        st.session_state.llm_response = None

    if st.button("Refine Prompt"):
        if not st.session_state.refiner_user_prompt:
            st.warning("Please enter a prompt to get a refined version.")
        else:
            with st.spinner("Refining prompt..."):
                try:
                    factory = LLMProviderFactory()
                    llm_provider = factory.get_provider(provider_name)

                    log_extra = {
                        "prompt_id": prompt_config.get('id', PROMPT_USE_CASE),
                        "prompt_version": prompt_config.get('version', 'unknown'),
                    }

                    llm_response_str = llm_provider.get_llm_response(
                        st.session_state.refiner_user_prompt,
                        selected_model,
                        prompt_model_config.get('prompt_roles', {}),
                        llm_logger,
                        log_extra
                    )
                    
                    import re
                    json_match = re.search(r'\{.*\}', llm_response_str, re.DOTALL)

                    if json_match:
                        json_str = json_match.group()
                        try:
                            st.session_state.llm_response = json.loads(json_str)
                        except json.JSONDecodeError:
                            st.session_state.llm_response = {"error": "The LLM returned a string that looks like JSON, but it is not valid.", "raw_response": json_str}
                    else:
                        st.session_state.llm_response = {"error": "The LLM did not return a valid JSON.", "raw_response": llm_response_str}
                except Exception as e:
                    st.session_state.llm_response = {"error": f"An error occurred: {e}", "raw_response": ""}

    if st.session_state.llm_response:
        response_json = st.session_state.llm_response
        if "error" in response_json:
            st.error(response_json["error"])
            st.markdown(response_json["raw_response"])
        else:
            expected_keys = ['review_comments', 'suggested_improvements', 'revised_prompt']
            if all(key in response_json for key in expected_keys):
                st.subheader("Review Comments")
                st.markdown(response_json.get('review_comments', 'N/A'))

                st.subheader("Suggested Improvements")
                st.markdown(response_json.get('suggested_improvements', 'N/A'))

                st.subheader("Revised Prompt")
                st.markdown(response_json.get('revised_prompt', 'N/A'))

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Test it"):
                        st.session_state.chat_system_prompt_prefill = response_json.get('revised_prompt', '')
                        st.session_state.messages = [] # Reset chat history
                        st.session_state.page = "Playground"
                        st.rerun()
                with col2:
                    def refine_again():
                        st.session_state.refiner_user_prompt = response_json.get('revised_prompt', '')
                    
                    st.button("Refine Again", on_click=refine_again)

            else:
                st.warning("The LLM returned a valid JSON, but it does not contain the expected keys. The keys found are: " + ", ".join(response_json.keys()))
                st.json(response_json)
