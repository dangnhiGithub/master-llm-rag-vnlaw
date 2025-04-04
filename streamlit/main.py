from transformers import AutoTokenizer, AutoModel
import torch
from collections import Counter
import regex as re
from nltk.tokenize import word_tokenize
from langchain.schema import Document
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import BM25Retriever
from chromadb import Settings, EmbeddingFunction, Embeddings
import chromadb
from langchain.vectorstores import Chroma
from ollama import AsyncClient
import asyncio
import ollama
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="CHATBOT RAG app", layout="wide")
st.title("CHATBOT RAG app")
st.write('Welcome to the CHATBOT RAG app!')


base_model_id = "FacebookAI/xlm-roberta-base"
trained_model_path = "./models/trained_embedding_small_data/"
data_path = "data/"
MAX_LEN = 512
OVERLAP = 50

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
model = AutoModel.from_pretrained(trained_model_path)
model.eval()

device = torch.device('cpu')
model.to(device)


class MyEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model, tokenizer, max_length=512):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length

    def embed_documents(self, texts):
        inputs = self.tokenizer(texts,
                                padding='max_length',
                                max_length=self.max_length,
                                return_tensors="pt")
        with torch.no_grad():
            inputs = {key: value.to(device) for key, value in inputs.items()}
            embeddings = self.model(**inputs).pooler_output

        return embeddings.cpu().numpy().tolist()

    def embed_query(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            inputs = {key: value.to(device) for key, value in inputs.items()}
            embeddings = self.model(**inputs).pooler_output
        return embeddings.cpu().numpy().tolist()[0]


myembed = MyEmbeddingFunction(model, tokenizer)

db = Chroma(
    embedding_function=myembed,  # lấy embedding model đã train được
    persist_directory='./chroma_db',
    collection_name="VNLaws",
)
vectorstore_retriever = db.as_retriever(search_kwargs={"k": 10})

stored_data = db._collection.get()
zip_doc = list(zip(stored_data["documents"], stored_data["metadatas"]))
docs = [Document(page_content=doc, metadata=metadata)
        for doc, metadata in zip_doc]
keyword_retriever = BM25Retriever.from_documents(docs, search_kwargs={"k": 10})
ensemble_retriever = EnsembleRetriever(
    retrievers=[vectorstore_retriever, keyword_retriever], weights=[0.5, 0.5])

ollama_api_url = "https://87ea-35-201-213-122.ngrok-free.app/"  # Thay đổi mỗi lần host

ollama_client = ollama.Client(
    host=ollama_api_url,
    headers={'Header': 'application/json'}
)


system_prompt = "Xây dựng chương trình RAG với chatbot trung thực.\n" \
    "Và không bao giờ trả lời các câu hỏi liên quan đến chính trị, tôn giáo, tình dục và các vấn đề nhạy cảm khác.\n" \
    "Tôi có bộ dữ liệu gồm các Lĩnh vực pháp luật Việt Nam bao gồm Bảo Hiểm, Lao Động, Nhà Đất và Tài Chính Ngân Hàng. " \
    "Trong đó bao bồm các văn bản hành chính của Chính phủ.\n"


def judment1_answerable(question, retry=2):
    if retry == 0:
        return None

    agent_judgment1 = 'Bạn là một agent với nhiệm vụ đánh giá câu trả lời. ' \
        'Với câu hỏi "{question}", bạn có thể trả lời câu hỏi bằng cơ sở dữ liệu của tôi được hay không (trả lời 1 nếu có thể, trả lời 0 nếu không thế). \n' \
        'Trả lời:'

    response = ollama_client.chat(
        model="gemma3:12b",
        messages=[{"role": "user", "content": system_prompt +
                   agent_judgment1.format(question=question)}],
        stream=False,
        options={"temperature": 0.8}
    )
    # print("response.message.content = ", response.message.content)
    if "0" in response.message.content:
        judment1_result = 0
    elif "1" in response.message.content:
        judment1_result = 1
    else:
        return judment1_answerable(question, retry - 1)

    return judment1_result


def planer(question,):
    agent_search_plan = 'Bạn là một agent để dự đoán thông tin cần truy vấn.\n' \
        'Theo bạn để trả lời câu hỏi "{question}", thì cần những thông tin gì ?\n' \
        'Gợi ý cho tôi một câu ngắn gọn bắt đầu sau từ "Trả lời" sau đây.\n' \
        'Trả lời:\n'

    response = ollama_client.chat(
        model="gemma3:12b",
        messages=[{"role": "user", "content": system_prompt +
                   agent_search_plan.format(question=question)}],
        stream=False,
    )
    plan = re.sub("^Trả lời:*\s*", "", response.message.content)
    return plan


def retriever(question, plan, top_k=10):
    text = f"Để trả lời câu hỏi {question} cần {plan}"
    relevants = ensemble_retriever.get_relevant_documents(text)
    name = [i.metadata["name"] for i in relevants]
    name_counts = Counter(name)
    ranked_names = sorted(name_counts.items(),
                          key=lambda x: x[1], reverse=True)

    sorted_documents = []
    for name, count in ranked_names:
        for doc in relevants:
            if doc.metadata["name"] == name:
                sorted_documents.append(doc)

    top_documents = sorted_documents[:10]
    return top_documents


PROMP_TEMPLATE = """
    Bạn là một chuyên gia trả lời câu hỏi về luật về các lĩnh vực bao gồm Bảo Hiểm, Lao Động, Nhà Đất và Tài Chính Ngân Hàng và câu trả lời dựa trên thông tin được cung cấp. \n
    Hãy trả lời câu hỏi ngắn gọn và dễ hiểu. Nếu không có thông tin để trả lời câu hỏi, hãy gợi ý câu hỏi khác phù hợp hơn. \n
    Đây là một số thông tin bạn có được:\n
    --------\n
    {context} \n
    --------\n
    Dựa vào thông tin trên hãy trả lời câu hỏi sau đây: {question} \n
    Trả lời: 
    """


messages = st.container()
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Tôi là trở lý ảo, tôi có thể giúp gì cho bạn?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

st.sidebar.button("Clear Chat History", on_click=lambda: st.session_state.update(
    messages=[{"role": "assistant", "content": "Tôi là trở lý ảo, tôi có thể giúp gì cho bạn?"}]))


def generate_stream_response(prompt):
    response = ollama_client.chat(
        model="gemma3:12b",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        options={"temperature": 0.9}
    )
    for partial_rep in response:
        token = partial_rep.message.content
        st.session_state['full_message'] += token
        yield token


if prompt := st.chat_input("Prompt của bạn"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    messages.chat_message('user').write(prompt)

    if judment1_answerable(prompt) == 0:
        st.chat_message("assistant", avatar="🤖").write("Xin lỗi, tôi không thể trả lời câu hỏi này.")
        st.session_state.messages.append({"role": "assistant", "content": "Xin lỗi, tôi không thể trả lời câu hỏi này."})
    if judment1_answerable(prompt) == 1:
        plan = planer(prompt)
        st.chat_message("agent planner", avatar="🧠").write(plan)
        st.session_state.messages.append({"role": "agent planner", "content": plan})

        top_documents = retriever(prompt, plan)
        messages.chat_message("agent retriever", avatar="🔍").write("Một số nguồn tham khảo")
        references = {doc.metadata["name"]: doc.metadata['url'] for doc in top_documents}
        for item in references.items():
            messages.chat_message("agent reader", avatar="📚").write(item)

        context = "\n--------\n".join([doc.page_content for doc in top_documents])

        st.session_state['full_message'] = ""
        st.chat_message("assistant", avatar="🤖").write_stream(generate_stream_response(PROMP_TEMPLATE.format(context=context, question=prompt)))
        st.session_state.messages.append(
            {"role": "assistant", "content": st.session_state['full_message']})

    # messages.chat_message("agent retriever", avatar="🔍").write("Đây là câu trả lời từ agent retriever")
    #
    # messages.chat_message("agent generator", avatar="🧠").write("Đây là câu trả lời từ agent generator")
    # messages.chat_message("ai").write("Đây là câu trả lời cuối cùng")
    # Add a button to clear chat history
