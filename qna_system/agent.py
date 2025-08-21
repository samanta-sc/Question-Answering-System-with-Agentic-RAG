from typing import Annotated, Literal, Sequence
from typing_extensions import TypedDict # Changed import here
from langchain import hub
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.graph.message import add_messages
from langgraph.prebuilt import tools_condition
from langgraph.graph import START, StateGraph, END
from langgraph.prebuilt import ToolNode
from qna_system.retrieval_generation import retrieval_chain


llm, chain = retrieval_chain()


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

class grade(BaseModel):
    binary_score:str=Field(description="Relevance score 'yes' or 'no'")

def retriever_tool(query: str):
    """Search the blog posts to answer a question."""
    print("---VECTOR RETRIEVER---")
    print(f"question: {query}")

    response = chain.invoke({'input': query})

    # return the relevant documents, not the full response object
    retrieved_docs = response.get("context", [])
    return "\n\n".join([doc.page_content for doc in retrieved_docs])

tools = [retriever_tool]

# Re-define the ToolNode with the updated retriever_tool
retrieve = ToolNode([retriever_tool])

def ai_assistant(state: AgentState):
    print("---CALL AGENT---")
    messages = state["messages"]
    llm_with_tool = llm.bind_tools(tools)
    response = llm_with_tool.invoke(messages)
    return {"messages":[response]}

def grade_documents(state: AgentState)->Literal["output_generator", "query_rewriter"]:
    llm_with_structure_op = llm.with_structured_output(grade)

    prompt = PromptTemplate(
        template="""You are a grader deciding if a document is relevant to a user’s question.
                    Here is the document: {context}
                    Here is the user’s question: {question}
                    If the document talks about or contains information related to the user’s question, mark it as relevant.
                    Give a 'yes' or 'no' answer to show if the document is relevant to the question.""",
                    input_variables=["context", "question"]
                    )
    chain = prompt | llm_with_structure_op
    messages = state["messages"]
    print(f"message from the grader: {messages}")
    last_message = messages[-1]
    question = messages[0].content
    docs = last_message.content
    scored_result = chain.invoke({"context": docs, "question": question})
    score = scored_result.binary_score


    if score == "yes":
        print("---DECISION: DOCS RELEVANT---")
        return "output_generator"
    else:
        print("---DECISION: DOCS NOT RELEVANT---")
        return "query_rewriter"
    
def generate(state: AgentState):
    print("---GENERATOR---")
    messages = state["messages"]

    print(f"message from the generator: {messages}")

    question = messages[0].content
    docs = messages[-1].content

    prompt = hub.pull("rlm/rag-prompt")

    rag_chain = prompt | llm

    response = rag_chain.invoke({"context": docs, "question": question})
    print(f"response from the generator: {response}")

    return {"messages":[response]}

def rewrite(state: AgentState):
    print("---TRANSFORM QUERY---")
    messages = state["messages"]
    question = messages[0].content
    print(f"initial question: {question}")

    message = [HumanMessage(content=f"""Look at the input and try to reason about the underlying semantic intent or meaning.
                    Here is the initial question: {question}
                    Formulate an improved question: """)
      ]

    response = llm.invoke(message)
    print(f"improved question: {response}")

    return {"messages":[response]}

def orchestrate():
    
    # Define the workflow
    workflow = StateGraph(AgentState)
    workflow.add_node("ai_assistant", ai_assistant)
    workflow.add_node("vector_retriever", retrieve)
    workflow.add_node("output_generator", generate)
    workflow.add_node("query_rewriter", rewrite)

    workflow.add_edge(START, "ai_assistant")
    workflow.add_conditional_edges("ai_assistant",
                               tools_condition,
                               {"tools": "vector_retriever",
                                END: END,})
    workflow.add_conditional_edges("vector_retriever",
                               grade_documents,
                               {"output_generator": "output_generator",
                                "query_rewriter": "query_rewriter",})
    
    workflow.add_edge("output_generator", END)
    workflow.add_edge("query_rewriter", "ai_assistant")

    # app = workflow.compile()

    return workflow
