# 基于RAG的智能知识库问答系统

使用LangChain搭建检索增强生成系统，基于本地物流文档实现智能问答，覆盖知识库构建、向量检索到答案生成的完整链路。

## 功能
- 加载本地文档构建向量知识库（FAISS）
- 用户提问 → 向量检索 → 召回相关文档片段 → 拼接提示词 → 大模型生成回答
- Streamlit 搭建前端交互界面，支持多轮对话

## 文件结构
- local_db.py：构建本地向量知识库
- local_qa.py：命令行问答模块
- web_rag.py：Streamlit 交互界面
- 物流数据.txt：物流知识库文档

## 运行方式
1. 安装依赖：pip install langchain streamlit pymupdf
2. 安装并启动 Ollama，拉取模型：ollama pull mxbai-embed-large && ollama pull qwen2.5:7b
3. 构建向量库：python local_db.py
4. 启动 Web 界面：streamlit run web_rag.py
