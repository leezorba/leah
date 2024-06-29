import streamlit as st

st.set_page_config(
    page_title="Leah",
    page_icon="✨",
)

st.title("Welcome to Hwa's GPT Portfolio")

st.markdown(
    """
    <span style="font-size: 20px;"> Hi there 👋, try out these AI tools I built! 🔥 You will need an OpenAI API key to use most of them; contact me if you need one. Thanks! 🤘</span>
                
    - <span style="font-size: 20px;">📃 [DocumentGPT](/DocumentGPT)</span>
    - <span style="font-size: 20px;">🔒 [PrivateGPT](/PrivateGPT)</span>
    - <span style="font-size: 20px;">❓ [QuizGPT](/QuizGPT)</span>
    - <span style="font-size: 20px;">🌐 [SiteGPT](/SiteGPT)</span>
    - <span style="font-size: 20px;">🖥️ [MeetingGPT](/MeetingGPT)</span>
    - <span style="font-size: 20px;">📈 [InvestorGPT](/InvestorGPT)</span>
    
    """,
    unsafe_allow_html=True,
)
