import streamlit as st

st.set_page_config(
    page_title="Leah",
    page_icon="✨",
)

st.title("Meet Leah, Your Faithful AI✨")

st.markdown(
    """
    <span style="font-size: 23px;">
    Leah is currently in training! Feel free to try out these apps.</span>
                
    - <span style="font-size: 20px;">[DocumentGPT](/DocumentGPT)</span>
    - <span style="font-size: 20px;">[PrivateGPT](/PrivateGPT)</span>
    - <span style="font-size: 20px;">[QuizGPT](/QuizGPT)</span>
    - <span style="font-size: 20px;">[SiteGPT](/SiteGPT)</span>
    - <span style="font-size: 20px;">[MeetingGPT](/MeetingGPT)</span>
    - <span style="font-size: 20px;">[ChatPluginGPT](/ManuelGPT)</span>
    
    """,
    unsafe_allow_html=True,
)
