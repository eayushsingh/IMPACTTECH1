"""
# DineMate Main Application

This module sets up the Streamlit UI for the DineMate food ordering chatbot with an enhanced dark theme.

Dependencies:
- streamlit: For UI rendering.
- scripts.utils: For chatbot and session handling.
- scripts.streaming: For real-time streaming responses.
- app modules: For specific pages (home, kitchen, analysis, etc.).
- time: For UI delays.
"""

import streamlit as st, time, traceback
import scripts.utils as utils
from scripts.config import STATIC_CSS_PATH
from scripts.streaming import StreamHandler, stream_graph_updates
from app import kitchen, update_prices, login, order_management, home, add_remove_items, track_order, analysis
from scripts.logger import get_logger

logger = get_logger(__name__)

# Set up Streamlit UI with dark theme
st.set_page_config(page_title="SaaS Voice Architecture - Food Ordering Bot", page_icon="S", layout="wide")

# Load centralized CSS
try:
    with open(STATIC_CSS_PATH, "r", encoding="utf-8") as f:
        css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    logger.error({"message": "styles.css not found"})
    st.error("CSS file not found. Please ensure static/styles.css exists.")

# Initialize session state for authentication
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
    st.session_state["username"] = None
    st.session_state["role"] = None

# Show login/register page if not authenticated
if not st.session_state["authenticated"]:
    login.login()
    st.stop()

# Sidebar with enhanced navigation
st.sidebar.markdown(
    "<div class='header'><h2 style='color: #E8ECEF;'>SaaS Voice Architecture</h2><p style='color: #FFA500;'>Order Smarter with AI</p></div>",
    unsafe_allow_html=True
)
st.sidebar.markdown(
    f"<h3 style='text-align: center;'><span style='color: #FFA500;'>{st.session_state['username'].title()}</span> ({st.session_state['role'].title()})</h3>",
    unsafe_allow_html=True
)

# Define Role-Based Page Access
ROLE_PAGES = {
    "admin": [
        {"label": "Home", "tooltip": "View SaaS Voice Architecture overview"},
        {"label": "Update Prices", "tooltip": "Manage menu prices"},
        {"label": "Kitchen Orders", "tooltip": "Handle kitchen tasks"},
        {"label": "Add/Remove Items", "tooltip": "Update menu items"},
        {"label": "Analysis", "tooltip": "Explore business insights"}
    ],
    "kitchen_staff": [
        {"label": "Home", "tooltip": "View SaaS Voice Architecture overview"},
        {"label": "Kitchen Orders", "tooltip": "Handle kitchen tasks"}
    ],
    "customer_support": [
        {"label": "Home", "tooltip": "View SaaS Voice Architecture overview"},
        {"label": "Order Management", "tooltip": "Manage customer orders"}
    ],
    "customer": [
        {"label": "Home", "tooltip": "View SaaS Voice Architecture overview"},
        {"label": "SaaS Voice AI", "tooltip": "Order with AI chatbot"},
        {"label": "Voice Chat", "tooltip": "Order with voice"},
        {"label": "Track Order", "tooltip": "Check order status"}
    ]
}

# Get allowed pages for the logged-in role
available_pages = [page["label"] for page in ROLE_PAGES.get(st.session_state["role"], [])]
tooltips = {page["label"]: page["tooltip"] for page in ROLE_PAGES.get(st.session_state["role"], [])}

# If no assigned pages, show warning
if not available_pages:
    st.markdown(
        "<div class='warning-container'><h3 style='color: #EF0606;'>No Access</h3><p>You do not have access to any pages.</p></div>",
        unsafe_allow_html=True
    )
    st.stop()

# Sidebar Navigation Menu with tooltips
page = st.sidebar.radio(
    "Navigation",
    available_pages,
    format_func=lambda x: x,
    label_visibility="collapsed",
    width="stretch"
)
for label in available_pages:
    st.markdown(f"<style>.stRadio label[data-label='{label}']::after {{ content: '{tooltips[label]}'; }}</style>", unsafe_allow_html=True)

# Load Selected Page
if page == "Home":
    home.home()

elif page == "SaaS Voice AI":
    st.markdown(
        "<div class='header'><h1>SaaS Voice AI Chatbot</h1><p style='color: #E8ECEF;'>Order food with our intelligent AI agent</p></div>",
        unsafe_allow_html=True
    )
    st.divider()

    @utils.enable_chat_history
    def chatbot_main():
        utils.sync_st_session()
        user_query = st.chat_input(placeholder="Type your order (e.g., '2 burgers and a coke')...")

        if user_query:
            with st.chat_message("user"):
                st.markdown(f"**You**: {user_query}")
                logger.info({"user": st.session_state["username"], "query": user_query, "message": "User submitted chatbot query"})

            with st.chat_message("assistant"):
                try:
                    with st.spinner("Processing your order..."):
                        response = st.write_stream(stream_graph_updates(user_query))
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        utils.print_qa(chatbot_main, user_query, response)
                        logger.info({"user": st.session_state["username"], "response": response, "message": "Chatbot response generated"})
                except Exception as e:
                    error_msg = f"Error processing request: {str(e)} (Type: {type(e).__name__})"
                    st.markdown(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    logger.error({
                        "user": st.session_state["username"],
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "traceback": traceback.format_exc(),
                        "message": "Chatbot error"
                    })
    chatbot_main()

elif page == "Voice Chat":
    st.markdown(
        "<div class='header'><h1>Voice Chat with SaaS Voice Architecture</h1><p style='color: #E8ECEF;'>Speak to our AI to order food</p></div>",
        unsafe_allow_html=True
    )
    st.divider()
    try:
        from app import voice_chat
        record_audio, transcribe_audio, get_llm_response, text_to_speech = voice_chat.ai_voice_assistance()
        st.info("Voice assistant is scoped to: order, replace/modify, cancel, and track order.")
        st.caption(f"Whisper model: {voice_chat.WHISPER_MODEL}")

        mic_options = voice_chat.list_input_devices()
        mic_labels = [label for label, _ in mic_options]
        selected_mic_label = st.selectbox(
            "Microphone device",
            options=mic_labels,
            index=0,
            help="Choose your active microphone. If speech is not detected, select a different device.",
        )
        selected_mic_device = dict(mic_options).get(selected_mic_label)

        source_language = st.selectbox(
            "Speech language",
            options=list(voice_chat.LANGUAGE_HINTS.keys()),
            index=0,
            help="Auto Detect is default and recommended. Select a language only if auto-detection struggles.",
        )
        translate_to_english = st.checkbox(
            "Translate speech to English before intent processing",
            value=True,
            help="Useful when speaking Telugu/Hindi/other languages to improve downstream understanding.",
        )

        if st.button("Start Voice Assistant", width="stretch"):
            with st.spinner("Listening..."):
                audio_file = record_audio(input_device=selected_mic_device)

            with st.spinner("Transcribing..."):
                user_text = transcribe_audio(
                    audio_file,
                    selected_language=source_language,
                    translate_to_english=translate_to_english,
                )

            if not user_text.strip():
                st.warning("No voice input detected. Please try again.")
            else:
                st.markdown(f"**You said:** {user_text}")

                response_source = get_llm_response(user_text)
                if isinstance(response_source, str):
                    response_text = response_source
                    st.markdown(f"**SaaS Voice Architecture:** {response_text}")
                else:
                    with st.spinner("Processing your request..."):
                        response_text = st.write_stream(response_source)

                with st.spinner("Speaking response..."):
                    text_to_speech(response_text)

    except Exception as e:
        logger.error({"error": str(e), "message": "Voice assistant initialization failed"})
        st.markdown(
            "<div class='warning-container'><h3 style='color: #EF0606;'>Voice Setup Error</h3><p>Install voice dependencies and use local execution for microphone support.</p></div>",
            unsafe_allow_html=True
        )
        st.caption(str(e))

elif page == "Kitchen Orders":
    kitchen.show_kitchen_orders()

elif page == "Update Prices":
    update_prices.show_price_update_page()

elif page == "Order Management":
    order_management.show_order_management()

elif page == "Add/Remove Items":
    add_remove_items.show_add_remove_items_page()

elif page == "Track Order":
    track_order.show_order_tracking()

elif page == "Analysis":
    analysis.show_analysis_page()

# Logout Button in Sidebar
st.sidebar.divider()
if st.sidebar.button("Logout", width="stretch"):
    st.success("Logging out...")
    logger.info({"user": st.session_state["username"], "message": "User logged out"})
    time.sleep(1.2)
    login.logout()