import streamlit as st
import time
from qna_system.agent import orchestrate


def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF 💁")
                

    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question:
        workflow = orchestrate()

        app = workflow.compile()
        start = time.process_time()
        
        # Invoke the agent app with the user's prompt
        response = app.invoke({"messages":[f"{user_question}"]})

        end = time.process_time()
        
        st.write(f"Response generated in {end - start:.2f} seconds.")
        st.code(response['messages'][-1].content, language='markdown')

        # Agentic Workflow
        with st.expander("Agentic Workflow"):
            st.write(response)
            st.write("--------------------------------")
        

if __name__ == "__main__":
    main()

