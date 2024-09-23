import streamlit as st
import openai

# Set up the title of the app
st.title("ChatGPT-like Clone")

# Initialize OpenAI client with the API key from Streamlit secrets
# openai.api_key = st.secrets["OPENAI_API_KEY"]

# Initialize the chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display the chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input from the chat input widget
if prompt := st.chat_input("Type your message..."):
    # Add the user message to the chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate a response using the OpenAI API
    with st.chat_message("assistant"):
        # Stream the assistant's response
        # response_stream = openai.ChatCompletion.create(
        #     model="gpt-3.5-turbo",  # Use the desired model
        #     messages=[{"role": msg["role"], "content": msg["content"]}
        #               for msg in st.session_state.messages],
        #     stream=True,
        # )

        # Display streamed response
        response_content = "d"
        # for chunk in response_stream:
        #     chunk_message = chunk["choices"][0].get(
        #         "delta", {}).get("content", "")
        #     response_content += chunk_message
        st.markdown(response_content)

    # Add the assistant's response to the chat history
    st.session_state.messages.append(
        {"role": "assistant", "content": response_content})
