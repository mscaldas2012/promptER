import json
import streamlit as st
from llm_factory import LLMProviderFactory
from logging_config import llm_logger
from prompt_loader import load_prompt_config
from utils import extract_thinking


@st.dialog("💭 Model Thinking", width="large")
def _show_thinking_dialog(content: str) -> None:
    st.markdown(content)

PROMPT_USE_CASE = "dfe-llm-evaluator"
PROMPT_MODEL_NAME = "default"


def evaluation_page():
    st.title("LLM-as-a-Judge")

    # Load configurations
    with open('providers.json') as providers_file:
        providers_config = json.load(providers_file)

    prompt_config = load_prompt_config(PROMPT_USE_CASE)
    if not prompt_config:
        st.error(f"Prompt configuration '{PROMPT_USE_CASE}' not found in Langfuse or config.yaml.")
        st.stop()

    prompt_model_config = prompt_config.get('models', {}).get(PROMPT_MODEL_NAME)
    if not prompt_model_config:
        st.error(f"Model configuration '{PROMPT_MODEL_NAME}' not found for prompt '{PROMPT_USE_CASE}'.")
        st.stop()

    # Provider selection
    providers = [k for k in providers_config if k != 'provider']
    default_provider = providers_config.get('provider', providers[0])
    provider_name = st.radio(
        "Select a provider",
        providers,
        index=providers.index(default_provider)
    )
    models = providers_config.get(provider_name, {}).get('models', [])
    selected_model = st.selectbox("Select a model", models)

    # Prefill prompts from config, but allow edits (order: System -> Assistant -> User)
    default_system_prompt = prompt_model_config.get('prompt_roles', {}).get('system', '')
    if 'eval_system_prompt' not in st.session_state:
        st.session_state.eval_system_prompt = default_system_prompt

    system_prompt = st.text_area(
        "System Prompt (LLM judge instructions)",
        key="eval_system_prompt",
        height=180,
        help="Loaded from Langfuse (or config.yaml fallback); you can edit for this session."
    )

    default_assistant_prompt = prompt_model_config.get('prompt_roles', {}).get('assistant', '')
    if 'eval_assistant_prompt' not in st.session_state:
        st.session_state.eval_assistant_prompt = default_assistant_prompt

    assistant_prompt = st.text_area(
        "Assistant Prompt (output format / scoring response)",
        key="eval_assistant_prompt",
        height=180,
        help="Loaded from Langfuse (or config.yaml fallback); you can edit for this session."
    )

    if 'eval_user_prompt' not in st.session_state:
        st.session_state.eval_user_prompt = ""
    if 'evaluation_result' not in st.session_state:
        st.session_state.evaluation_result = None
    if 'evaluation_thinking' not in st.session_state:
        st.session_state.evaluation_thinking = None

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
                            "use_case": PROMPT_USE_CASE,
                            "prompt_id": prompt_config.get('id', PROMPT_USE_CASE),
                            "prompt_version": prompt_config.get('version', 'unknown'),
                            "model_params": prompt_model_config.get('model_params', {}),
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

                        clean_eval, thinking = extract_thinking(evaluation)
                        st.session_state.evaluation_thinking = thinking
                        st.session_state.evaluation_result = clean_eval
                    except Exception as e:
                        st.session_state.evaluation_thinking = None
                        st.session_state.evaluation_result = f"An error occurred: {e}"

    if st.session_state.evaluation_result:
        with result_area:
            header_col, btn_col = st.columns([4, 1])
            with header_col:
                st.subheader("Evaluation Result")
            with btn_col:
                if st.session_state.get("evaluation_thinking"):
                    if st.button("💭 View Thinking", key="view_thinking_eval"):
                        _show_thinking_dialog(st.session_state.evaluation_thinking)
            st.markdown(st.session_state.evaluation_result)
