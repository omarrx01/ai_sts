
import streamlit as st
from rag_engine import retrieve_chunks, generate_answer

# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="NovaTech AI Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 NovaTech AI Assistant")
st.caption("Your smart assistant for NovaTech products and support")
st.caption("Demo project | All products, prices, and policies are fictional.")

st.divider()

# --------------------------------------------------
# 2. Initialize chat history
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# Welcome message
if not st.session_state.messages:
    st.info(
        "👋 Welcome to NovaTech! Ask about products, prices, "
        "specifications, delivery, or return policies."
    )

# --------------------------------------------------
# 3. Display previous messages
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Display saved sources for previous AI answers
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("📚 View Sources"):
                for index, source in enumerate(
                    message["sources"], start=1
                ):
                    st.markdown(
                        f"**Source {index} — Page {source['page']}**"
                    )
                    st.write(source["text"])
                    st.divider()

# --------------------------------------------------
# 4. Chat input
# --------------------------------------------------

question = st.chat_input("Ask anything about NovaTech...")

if question:

    # Save and display user's question
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    # --------------------------------------------------
    # 5. Retrieve information and generate AI answer
    # --------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            results = []

            try:
                # Step 1: Retrieve relevant chunks
                results = retrieve_chunks(question)

                # Step 2: Generate answer using retrieved context
                response = generate_answer(question, results)

                # Step 3: Display generated answer
                st.markdown(response)

                # Step 4: Display supporting sources
                if results:
                    with st.expander("📚 View Sources"):

                        for index, result in enumerate(
                            results, start=1
                        ):
                            st.markdown(
                                f"**Source {index} — Page {result['page']}**"
                            )

                            st.write(result["text"])

                            st.caption(
                                f"Cosine similarity: "
                                f"{result['similarity']:.4f}"
                            )

                            st.divider()

            except Exception as error:

                response = (
                    "Sorry, something went wrong while generating "
                    "your answer. Please try again."
                )

                st.error(response)
                st.caption(f"Technical details: {error}")

                results = []

    # --------------------------------------------------
    # 6. Save AI answer and sources in chat history
    # --------------------------------------------------

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "sources": results
    })
