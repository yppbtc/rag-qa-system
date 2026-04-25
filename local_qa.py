"""
RAG 问答模块（命令行版）
加载本地向量库 → 检索相关文档 → 拼接提示词 → 大模型生成回答
"""
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
import time
from langchain_community.embeddings import OllamaEmbeddings

# 1. 加载本地向量库
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
db = FAISS.load_local(
    ".faiss/wuliu",
    embeddings=embeddings,
    allow_dangerous_deserialization=True
)

# 工具函数：拼接检索到的文档
def get_related_content(related_docs):
    related_content = []
    for doc in related_docs:
        related_content.append(doc.page_content.replace("\n\n", "\n"))
    return "\n".join(related_content)

# 2. 构建提示词
def define_prompt():
    # question = "我的快递出发地是哪?预计几天到达?"
    question = "北京到上海用什么运输方式？"
    docs = db.similarity_search(question, k=2)
    related_content = get_related_content(docs)

    PROMPT_TEMPLATE = """
    基于以下已知信息，简洁专业地回答问题，不允许编造。
    已知内容:
    {context}
    问题:
    {question}
    """
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=PROMPT_TEMPLATE
    )
    my_pmt = prompt.format(context=related_content, question=question)
    return my_pmt

# 3. 问答主函数
def qa():
    """执行问答：加载模型 → 构建提示词 → 生成回答"""
    model = Ollama(model="qwen2.5:7b")
    my_pmt = define_prompt()
    result = model.invoke(my_pmt)
    return result

if __name__ == '__main__':
    start_time = time.time()
    result = qa()
    print("AI回答：", result)
    print(f"耗时：{time.time() - start_time}秒")
