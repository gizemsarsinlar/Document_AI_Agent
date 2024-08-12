from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.passthrough import RunnableAssign
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import gradio as gr
from functools import partial
from operator import itemgetter

from faiss import IndexFlatL2
from langchain_community.docstore.in_memory import InMemoryDocstore
import json
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings

from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ArxivLoader
from langchain_community.document_transformers import LongContextReorder

from langchain_community.document_loaders import PyPDFLoader


import os

api_key = "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"


# # Mevcut modelleri kontrol edin
# available_embeddings = NVIDIAEmbeddings.get_available_models(api_key=api_key)
# print("Available NVIDIA Embedding Models:", available_embeddings)

# available_llms = ChatNVIDIA.get_available_models(api_key=api_key)
# print("Available NVIDIA Language Models:", available_llms)

# NVIDIAEmbeddings.get_available_models()
embedder = NVIDIAEmbeddings(model="NV-Embed-QA", api_key=api_key, truncate="END")
# ChatNVIDIA.get_available_models()
instruct_llm = ChatNVIDIA(model="mistralai/mixtral-8x7b-instruct-v0.1", api_key=api_key)

embed_dims = len(embedder.embed_query("test"))
def default_FAISS():
    '''Useful utility for making an empty FAISS vectorstore'''
    return FAISS(
        embedding_function=embedder,
        index=IndexFlatL2(embed_dims),
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
        normalize_L2=False
    )

def aggregate_vstores(vectorstores):
    ## Initialize an empty FAISS Index and merge others into it
    ## We'll use default_faiss for simplicity, though it's tied to your embedder by reference
    agg_vstore = default_FAISS()
    for vstore in vectorstores:
        agg_vstore.merge_from(vstore)
    return agg_vstore

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100,
    separators=["\n\n", "\n", ".", ";", ",", " "],
)

docs = [
    ArxivLoader(query="1706.03762").load(),  ## Attention Is All You Need Paper
    ArxivLoader(query="1810.04805").load(),  ## BERT Paper
    ArxivLoader(query="2005.11401").load(),  ## RAG Paper
    ArxivLoader(query="2205.00445").load(),  ## MRKL Paper
    ArxivLoader(query="2310.06825").load(),  ## Mistral Paper
    ArxivLoader(query="2306.05685").load(),  ## LLM-as-a-Judge
    ## Some longer papers
    ArxivLoader(query="2210.03629").load(),  ## ReAct Paper
    ArxivLoader(query="2112.10752").load(),  ## Latent Stable Diffusion Paper
    ArxivLoader(query="2103.00020").load(),  ## CLIP Paper
    ## TODO: Feel free to add more
]


## Cut the paper short if references is included.
## This is a standard string in papers.
for doc in docs:
    content = json.dumps(doc[0].page_content)
    if "References" in content:
        doc[0].page_content = content[:content.index("References")]

## Split the documents and also filter out stubs (overly short chunks)
print("Chunking Documents")
docs_chunks = [text_splitter.split_documents(doc) for doc in docs]
docs_chunks = [[c for c in dchunks if len(c.page_content) > 200] for dchunks in docs_chunks]

## Make some custom Chunks to give big-picture details
doc_string = "Available Documents:"
doc_metadata = []
for chunks in docs_chunks:
    metadata = getattr(chunks[0], 'metadata', {})
    doc_string += "\n - " + metadata.get('Title')
    doc_metadata += [str(metadata)]

extra_chunks = [doc_string] + doc_metadata

vecstores = [FAISS.from_texts(extra_chunks, embedder)]
vecstores += [FAISS.from_documents(doc_chunks, embedder) for doc_chunks in docs_chunks]

## Unintuitive optimization; merge_from seems to optimize constituent vector stores away
docstore = aggregate_vstores(vecstores)

print(f"Constructed aggregate docstore with {len(docstore.docstore._dict)} chunks")

convstore = default_FAISS()

# Fonksiyon tanımları
def long_reorder(chunks):
    """Belgeleri uzunluklarına göre yeniden sıralar."""
    return sorted(chunks, key=lambda x: len(x.page_content), reverse=True)

def docs2str(docs):
    """Belgeleri string formatına dönüştürür."""
    return "\n\n".join([doc.page_content for doc in docs])

def save_memory_and_get_output(d, vstore):
    """Accepts 'input'/'output' dictionary and saves to convstore"""
    vstore.add_texts([
        f"User previously responded with {d.get('input')}",
        f"Agent previously responded with {d.get('output')}"
    ])
    return d.get('output')

initial_msg = (
    "Hello! I am a document chat agent here to help the user!"
    f" I have access to the following documents: {doc_string}\n\nHow can I help you?"
)

chat_prompt = ChatPromptTemplate.from_messages([("system",
    "You are a document chatbot. Help the user as they ask questions about documents."
    " User messaged just asked: {input}\n\n"
    " From this, we have retrieved the following potentially-useful info: "
    " Conversation History Retrieval:\n{history}\n\n"
    " Document Retrieval:\n{context}\n\n"
    " (Answer only from retrieval. Only cite sources that are used. Make your response conversational.)"
), ('user', '{input}')])


def RPrint(preface=""):
    """Simple passthrough "prints, then returns" chain"""
    def print_and_return(x, preface):
        if preface: print(preface, end="")
        return x
    return RunnableLambda(partial(print_and_return, preface=preface))

retrieval_chain = (
    {'input' : (lambda x: x)}
    ## TODO: Make sure to retrieve history & context from convstore & docstore, respectively.
    ## HINT: Our solution uses RunnableAssign, itemgetter, long_reorder, and docs2str
    | RunnableAssign({'history' : itemgetter('input') | convstore.as_retriever() | long_reorder | docs2str})
    | RunnableAssign({'context' : itemgetter('input') | docstore.as_retriever()  | long_reorder | docs2str})
    | RPrint()
)

stream_chain = chat_prompt| RPrint() | instruct_llm | StrOutputParser()

def chat_gen(message, history=[], return_buffer=True):
    buffer = ""
    ## First perform the retrieval based on the input message
    retrieval = retrieval_chain.invoke(message)
    line_buffer = ""

    ## Then, stream the results of the stream_chain
    for token in stream_chain.stream(retrieval):
        buffer += token
        ## If you're using standard print, keep line from getting too long
        yield buffer if return_buffer else token

    ## Lastly, save the chat exchange to the conversation memory buffer
    save_memory_and_get_output({'input':  message, 'output': buffer}, convstore)


# ## Start of Agent Event Loop
# test_question = "Tell me about RAG!"  ## <- modify as desired

# ## Before you launch your gradio interface, make sure your thing works
# for response in chat_gen(test_question, return_buffer=False):
#     print(response, end='')

chatbot = gr.Chatbot(value = [[None, initial_msg]])
demo = gr.ChatInterface(chat_gen, chatbot=chatbot).queue()

try:
    demo.launch(debug=True, share=False, show_api=False)
    demo.close()
except Exception as e:
    demo.close()
    print(e)
    raise e
