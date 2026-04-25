"""
RAG 问答系统（Streamlit 交互版）
前端对话界面 + 本地向量检索 + 大模型生成回答，支持多轮对话记忆
"""
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
import streamlit as st

# ===================== 页面基础配置 =====================
st.set_page_config(
    page_title="物流RAG问答",
    layout="wide",
    page_icon="🚛"  # 浏览器标签图标
)

# 侧边栏加个简单说明（让界面更专业）
with st.sidebar:
    st.title("🚛 物流行业信息咨询系统")
    st.divider()
    st.subheader("功能说明")
    st.write("基于本地物流知识库，为您提供精准的物流信息咨询服务。")
    st.divider()
    st.caption("版本：v1.0 | 本地部署，数据安全")

# ===================== 加载向量库（缓存提速） =====================
@st.cache_resource
def load_vector_db():
    """加载本地 FAISS 向量库，使用缓存避免重复加载"""
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    db = FAISS.load_local(
        ".faiss/wuliu",
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )
    return db

db = load_vector_db()

# ===================== 初始化带记忆的检索链 =====================
def new_retrival():
    chain = ConversationalRetrievalChain.from_llm(
        llm=Ollama(model="qwen2.5:7b"),
        retriever=db.as_retriever(),
        return_source_documents=False
    )
    return chain

# ===================== 聊天历史管理 =====================
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息（给消息加背景色区分）
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "🤖"):
        st.markdown(msg["content"])

# ===================== 用户输入 + AI回答 =====================
if prompt := st.chat_input("请输入你的问题，比如：我的快递预计几天到达？"):
    # 保存并显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    # AI回答（加了加载动画）
    with st.chat_message("assistant", avatar="🤖"):
        placeholder = st.empty()
        chat_history = []
        chain = new_retrival()
        with st.spinner("AI正在查询物流信息，请稍等..."):
            result = chain.invoke({
                "question": prompt,
                "chat_history": chat_history
            })
            answer = result["answer"]
        placeholder.markdown(answer)

    # 保存AI回答
    st.session_state.messages.append({"role": "assistant", "content": answer})
