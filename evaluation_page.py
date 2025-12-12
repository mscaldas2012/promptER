import json
import yaml
import streamlit as st
from dotenv import load_dotenv
from llm_factory import LLMProviderFactory
from logging_config import llm_logger

# Load environment variables from .env file
load_dotenv()

PROMPT_USE_CASE = "llm_evaluation"
PROMPT_MODEL_NAME = "default"


def evaluation_page():
    st.title("LLM-as-a-Judge")

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
    provider_name = st.radio(
        "Select a provider",
        ["ollama", "azure_openai"],
        index=["ollama", "azure_openai"].index(providers_config.get('provider', 'ollama'))
    )

    # Model selection
    if provider_name == "ollama":
        models = providers_config.get('ollama', {}).get('models', [])
    else:
        models = providers_config.get('azure_openai', {}).get('models', [])
    selected_model = st.selectbox("Select a model", models)

    # Prefill prompts from config, but allow edits (order: System -> Assistant -> User)
    default_system_prompt = prompt_model_config.get('prompt_roles', {}).get('system', '')
    if 'eval_system_prompt' not in st.session_state:
        st.session_state.eval_system_prompt = default_system_prompt

    system_prompt = st.text_area(
        "System Prompt (LLM judge instructions)",
        key="eval_system_prompt",
        height=180,
        help="Loaded from config.yaml; you can edit for this session."
    )

    default_assistant_prompt = prompt_model_config.get('prompt_roles', {}).get('assistant', '')
    if 'eval_assistant_prompt' not in st.session_state:
        st.session_state.eval_assistant_prompt = default_assistant_prompt

    assistant_prompt = st.text_area(
        "Assistant Prompt (output format / scoring response)",
        key="eval_assistant_prompt",
        height=180,
        help="Loaded from config.yaml; you can edit for this session."
    )

    if 'eval_user_prompt' not in st.session_state:
        st.session_state.eval_user_prompt = ""
    if 'evaluation_result' not in st.session_state:
        st.session_state.evaluation_result = None

    # User prompt (placed before the Run button)
    st.session_state.eval_user_prompt = st.text_area(
        "User Prompt to Evaluate",
        value=st.session_state.eval_user_prompt,
        height=140
    )

    run_clicked = st.button("Run Evaluation")
    result_area = st.container()

    if run_clicked:
        user_prompt_val = st.session_state.eval_user_prompt
        if not user_prompt_val:
            st.warning("Please provide a user prompt to evaluate.")
        else:
            with result_area:
                with st.spinner("Scoring..."):
                    try:
                        factory = LLMProviderFactory()
                        llm_provider = factory.get_provider(provider_name)

                        log_extra = {
                            "prompt_id": prompt_config.get('id', PROMPT_USE_CASE),
                            "prompt_version": prompt_config.get('version', 'unknown'),
                        }

                        roles = {"system": system_prompt, "assistant": assistant_prompt}

                        evaluation_input = f"User prompt:\n{user_prompt_val}"

                        evaluation = llm_provider.get_llm_response(
                            evaluation_input,
                            selected_model,
                            roles,
                            llm_logger,
                            log_extra
                        )

                        st.session_state.evaluation_result = evaluation
                    except Exception as e:
                        st.session_state.evaluation_result = f"An error occurred: {e}"

    if st.session_state.evaluation_result and not run_clicked:
        with result_area:
            st.subheader("Evaluation Result")
            st.markdown(st.session_state.evaluation_result)
    elif st.session_state.evaluation_result and run_clicked:
        with result_area:
            st.subheader("Evaluation Result")
            st.markdown(st.session_state.evaluation_result)
