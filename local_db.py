"""
构建本地向量知识库
加载物流文档 → 文本切分 → 向量嵌入 → 存储到本地 FAISS
"""

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader

def get_vector():
    # 1. 加载物流数据文档
    loader = TextLoader("物流数据.txt", encoding="utf-8")
    data = loader.load()
    print(f"加载了 {len(data)} 个文档")
    # loader = PyMuPDFLoader("物流信息.pdf")
    # data = loader.load()
    # print(len(data))

    # 2. 文本切分：每块500字，重叠50字，保证上下文连贯
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    split_docs = text_splitter.split_documents(data)
    print(f"切分后文档块数：{len(split_docs)}")

    # 3. 向量嵌入并存储到本地
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    db = FAISS.from_documents(split_docs, embeddings)
    db.save_local(".faiss/wuliu")
    print("向量库构建完成！")


if __name__ == '__main__':
    get_vector()