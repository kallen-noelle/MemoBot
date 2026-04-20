from typing import List, Literal
from langchain_chroma import Chroma

from langchain.tools import tool
from langchain_core.documents import Document
from backend.config import get_app_config
from backend.entity.paths import get_paths
from langgraph.runtime import Runtime

import logging
logger = logging.getLogger(__name__)

def reciprocal_rank_fusion(results_list, k=60):
    """Reciprocal Rank Fusion 融合多种检索结果"""
    fused_scores = {}
    doc_id_map = {}
    
    for results in results_list:
        for rank, doc in enumerate(results):
            doc_id = doc.metadata.get('id', str(doc))
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0
            fused_scores[doc_id] += 1 / (k + rank)
            if doc_id not in doc_id_map:
                doc_id_map[doc_id] = doc
    
    sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id_map[doc_id] for doc_id, score in sorted_docs[:3]]


    
def _bm25_rerank(query, documents: List[Document], top_k=5)-> List[Document]:
    """
    根据查询检索相关文档
    query: 查询字符串
    top_k: 返回前k个结果
    """
    try:
        from rank_bm25 import BM25Okapi
        import jieba
    except ImportError:
        logger.error("BM25或结巴分词器未安装，无法进行BM25关键词检索。")
        return []
    bm25 = BM25Okapi([list(jieba.cut(doc.page_content)) for doc in documents])
    # 对查询进行分词
    tokenized_query = list(jieba.cut(query))
    
    # 获取得分
    scores = bm25.get_scores(tokenized_query)
    
    # 获取top-k结果
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    
    # 返回结果
    results = []
    for idx in top_indices:
        if scores[idx] > 0:  # 只返回有得分的文档
            results.append(documents[idx])
    
    return results

    
@tool("retrieve_documents", parse_docstring=True, return_direct=True)
def retrieve_documents(query: str,file_ids: List[str],runtime: Runtime) -> str:
    """
    当用户的问题与上传的资料内容相关或者任何联系时，该工具可以帮助你获取最相关的资料。
    例如当：文件列表：国家基层高血压防治管理指南.pdf:id xxx ...。queston；老年人要怎么防治高血压？
    你需要生成相关的query查询文件中的内容。
    请注意:只要问题与任何上传的资料内容相关或者任何联系时，你都需要进行查询，并作为回答用户问题的关键参考信息
    Args:
        query: 查询字符串
        file_ids: 用户上传的文件ID列表
        
    Returns:
        str: 检索到的相关信息，格式为文本字符串
    """

    
    try:
        # 1. 向量检索
        uid = runtime.context.get("uid")
        if not uid:
            return "未找到用户ID，无法进行检索。"

        config = get_app_config().vector_db
        embeddings = config.embedding_model()
        vectorstore = Chroma(
            collection_name="vector_db",
            embedding_function=embeddings,
            persist_directory=get_paths().base_dir(uid)
            
            
        )
        knowledge_graph =  Chroma(
            collection_name="knowledge_graph",
            embedding_function=embeddings,
            persist_directory=get_paths().base_dir(uid)
            
        )
        vector_results = vectorstore.as_retriever(
            search_kwargs={"k": config.top_k_documents}
        ).invoke(query)
        
        #
        bm25_results = _bm25_rerank(query, vector_results)
        # 4. 融合向量和BM25检索结果
        fused_docs = reciprocal_rank_fusion([vector_results, bm25_results])


        # 3. 知识图谱检索
        graph_results = knowledge_graph.query_related_entities(query)
        
        # 5. 构建返回结果
        if fused_docs:
            documents_text = "\n\n".join([doc.page_content for doc in fused_docs])
            graph_context = "\n".join([f"  - {entity.page_content}" for entity in graph_results])
            
            result = f"【检索到的资料】\n{documents_text}\n\n【知识图谱关系】\n{graph_context}"
            return result
        else:
            return "未检索到相关医疗信息，请尝试使用更具体的关键词。"
            
    except Exception as e:
        return f"检索过程中出现错误: {str(e)}"
