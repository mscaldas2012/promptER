import streamlit as st
from main_page import main_page
from chat_page import chat_page
from evaluation_page import evaluation_page

PAGES = {
    "Prompt Refiner": main_page,
    "Playground": chat_page,
    "Evaluation": evaluation_page
}

st.sidebar.title('Navigation')

# Initialize session state for the page
if 'page' not in st.session_state:
    st.session_state.page = "Prompt Refiner"

# Function to set the page
def set_page(page_name):
    st.session_state.page = page_name

# Create buttons for each page
for page_name in PAGES.keys():
    st.sidebar.button(page_name, on_click=set_page, args=(page_name,))

# Get the page from session state and call it
page = PAGES[st.session_state.page]
page()
