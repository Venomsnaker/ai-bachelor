# Import Libraries
import os
import warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv

from typing import List, TypedDict
import gradio as gr

from langchain_upstage import ChatUpstage, UpstageLayoutAnalysisLoader, UpstageEmbeddings, GroundednessCheck
from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph

# Set up Upstage API
upstage_api_key_env_name = "UPSTAGE_API_KEY"
def load_env():
    load_dotenv()
    return os.environ.get(upstage_api_key_env_name)
UPSTAGE_API_KEY = load_env()

# Set up the database
def layout_analysis(filenames: str) -> List[Document]:
    layout_analysis_loader = UpstageLayoutAnalysisLoader(filenames, split="element")
    return layout_analysis_loader.load()

filenames = [
    'data/map.pdf',
]
docs = layout_analysis(filenames)
db: VectorStore = Chroma(
    embedding_function=UpstageEmbeddings(model="solar-embedding-1-large-passage")
)
retriever = db.as_retriever()
db.add_documents(docs)

# Set up the model and prompt
template = """Ensure the itinerary follows the context, do not repeat the locations:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
model = ChatUpstage(model="solar-pro")
model_chain = prompt | model | StrOutputParser()

# Set up the RAG system
class RagState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        context: retrieved context
        question: question asked by the user
        answer: generated answer to the question
        groundedness: groundedness of the assistant's response
    """

    context: str
    question: str
    answer: str
    groundedness: str

def format_documents(docs: List[Document]) -> str:
    return "\n".join([doc.page_content for doc in docs])

def retrieve(state: RagState) -> RagState:
    docs = retriever.invoke(state["question"])
    context = format_documents(docs)
    return RagState(context=context)

def model_answer(state: RagState) -> RagState:
    response = model_chain.invoke(state)
    return {**state, "answer": response}

gc = GroundednessCheck()

def groundedness_check(state: RagState) -> RagState:
    response = gc.run({"context": state["context"], "answer": state["answer"]})
    return RagState(groundedness=response)

def groundedness_condition(state: RagState) -> RagState:
    return state["groundedness"]

# Set up the workflow
workflow = StateGraph(RagState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("model", model_answer)
workflow.add_node("groundedness_check", groundedness_check)

workflow.add_edge("retrieve", "model")
workflow.add_edge("model", "groundedness_check")
workflow.add_conditional_edges(
    "groundedness_check",
    groundedness_condition,
    {
        "grounded": END,
        "notGrounded": "model",
        "notSure": "model",
    },
)
workflow.set_entry_point("retrieve")
app = workflow.compile()

def chat(inputs_user):
    inputs = {"question": inputs_user}
    respond = ""

    for output in app.stream(inputs):
        for key, value in output.items():
            if key == 'model' and 'answer' in value:
                respond = value['answer']
    return respond

interface = gr.Interface(
    fn=chat,
    inputs=[gr.Textbox(label="text", lines=2)],
    outputs='text',
    title="Solar Pro Itinerary Creator",
)

if __name__ == "__main__":
    interface.launch()