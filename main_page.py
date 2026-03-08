
import streamlit as st
import json
from llm_factory import LLMProviderFactory
from logging_config import llm_logger
from prompt_loader import load_prompt_config

PROMPT_USE_CASE = "meta-prompt-generator"
PROMPT_MODEL_NAME = "default"
FRAMEWORK_OPTIONS_PATH = "framework_options.json"


def load_framework_options():
    try:
        with open(FRAMEWORK_OPTIONS_PATH) as framework_options_file:
            framework_options = json.load(framework_options_file)
    except (OSError, json.JSONDecodeError) as exc:
        st.error(f"Failed to load framework options from {FRAMEWORK_OPTIONS_PATH}: {exc}")
        st.stop()

    if not isinstance(framework_options, list):
        st.error(f"Framework options in {FRAMEWORK_OPTIONS_PATH} must be a list.")
        st.stop()

    return framework_options


def apply_framework_template(prompt_roles, replacements):
    updated_roles = {}
    for role, content in prompt_roles.items():
        if isinstance(content, str):
            updated_content = content
            for placeholder, value in replacements.items():
                updated_content = updated_content.replace(placeholder, value)
            updated_roles[role] = updated_content
        else:
            updated_roles[role] = content
    return updated_roles


def inject_framework_instructions(prompt_roles, framework_instructions):
    if not framework_instructions:
        return prompt_roles

    updated_roles = dict(prompt_roles)
    system_prompt = updated_roles.get("system")
    if isinstance(system_prompt, str):
        updated_roles["system"] = (
            f"{system_prompt}\n\nFramework instructions:\n{framework_instructions}"
        )
    return updated_roles


def main_page():
    st.title("AI Prompt Builder")

    st.write("Enter your initial system prompt below, and the AI will help you refine it.")

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

    framework_options = load_framework_options()

    # Provider selection
    provider_name = st.radio("Select a provider", ["ollama", "azure_openai", "edav_openai"], index=["ollama", "azure_openai", "edav_openai"].index(providers_config['provider']))
    models = providers_config[provider_name]['models']
    # # Model selection
    # if provider_name == "ollama":
    #     models = providers_config['ollama']['models']
    # elif provider_name == "edav_openai":
    #     models = providers_config['edav_openai']['models']
    # else:
    #     models = providers_config['azure_openai']['models']
    selected_model = st.selectbox("Select a model", models)

    if 'refiner_user_prompt' not in st.session_state:
        st.session_state.refiner_user_prompt = ""

    if not isinstance(st.session_state.refiner_user_prompt, str):
        st.session_state.refiner_user_prompt = json.dumps(st.session_state.refiner_user_prompt, indent=2)

    user_prompt = st.text_area("Your System Prompt", key="refiner_user_prompt", height=150)

    framework_labels = [option["label"] for option in framework_options]
    selected_framework_label = st.selectbox("Prompt Framework (optional)", framework_labels, index=0)
    selected_framework_option = next(option for option in framework_options if option["label"] == selected_framework_label)
    custom_framework = ""
    if selected_framework_option["key"] == "OTHER":
        custom_framework = st.text_input("Enter a custom framework name or description", key="custom_framework_text")
    selected_framework_text = custom_framework.strip() if selected_framework_option["key"] == "OTHER" else selected_framework_option["label"]
    framework_note = selected_framework_text or "Free Form / No Specific Framework"
    framework_description = selected_framework_option.get("description", "")
    framework_instructions = selected_framework_option.get("instructions", "")
    framework_key = selected_framework_option.get("key", "")
    if framework_key == "OTHER":
        framework_key = "custom"
        framework_description = custom_framework.strip()
        framework_instructions = custom_framework.strip()
    if framework_key == "free_form":
        framework_description = selected_framework_option.get("description", "")

    if 'llm_response' not in st.session_state:
        st.session_state.llm_response = None

    if st.button("Refine Prompt"):
        if not st.session_state.refiner_user_prompt:
            st.warning("Please enter a prompt to get a refined version.")
        elif selected_framework_option["key"] == "OTHER" and not selected_framework_text:
            st.warning("Please enter your custom framework or choose another option.")
        else:
            with st.spinner("Refining prompt..."):
                try:
                    factory = LLMProviderFactory()
                    llm_provider = factory.get_provider(provider_name)

                    log_extra = {
                        "use_case": PROMPT_USE_CASE,
                        "prompt_id": prompt_config.get('id', PROMPT_USE_CASE),
                        "prompt_version": prompt_config.get('version', 'unknown'),
                        "prompt_framework": framework_note,
                        "model_params": prompt_model_config.get('model_params', {}),
                    }
                    prompt_roles = prompt_model_config.get('prompt_roles', {})
                    replacements = {
                        "{{framework}}": framework_note,
                        "{{framework_label}}": framework_note,
                        "{{framework_description}}": framework_description,
                        "{{framework_instructions}}": framework_instructions,
                        "{{framework_key}}": framework_key,
                    }
                    prompt_roles_with_framework = apply_framework_template(prompt_roles, replacements)
                    prompt_roles_with_framework = inject_framework_instructions(
                        prompt_roles_with_framework,
                        framework_instructions
                    )

                    llm_response_str = llm_provider.get_llm_response(
                        st.session_state.refiner_user_prompt,
                        selected_model,
                        prompt_roles_with_framework,
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
                        revised_prompt = response_json.get('revised_prompt', '')
                        if not isinstance(revised_prompt, str):
                            revised_prompt = json.dumps(revised_prompt, indent=2)
                        st.session_state.refiner_user_prompt = revised_prompt
                    
                    st.button("Refine Again", on_click=refine_again)

            else:
                st.warning("The LLM returned a valid JSON, but it does not contain the expected keys. The keys found are: " + ", ".join(response_json.keys()))
                st.json(response_json)
