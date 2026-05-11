
import json
import uuid
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document

from backend.utils.text_splitter import TextSplitter
from backend.config import get_app_config
from backend.entity.paths import get_paths
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
import logging

logger = logging.getLogger(__name__)
class RagBuilder:
    def __init__(self):
        config = get_app_config()
        self.embeddings = DashScopeEmbeddings(model=config.vector_db.embedding_model, dashscope_api_key=config.vector_db.api_key)
        self.llm = ChatOpenAI(
            model=config.models[0].name,
            api_key=config.models[0].api_key,
            base_url=config.models[0].base_url,
        )
        self.text_splitter = TextSplitter()

    def build_from_file(self, content: str, source: str, uid: str):
        persist_dir = str(get_paths().base_dir(uid))

        vectorstore = Chroma(
            collection_name="vector_db",
            embedding_function=self.embeddings,
            persist_directory=persist_dir,
        )

        graphstore = Chroma(

            collection_name="graph_db",
            embedding_function=self.embeddings,
            persist_directory=persist_dir
        )

        chunks = self.text_splitter.split_text_into_chunks(content, source)

        logger.info(f"将文本分块为{len(chunks)}个块")
        print(chunks)

        texts = []
        metadatas = []
        for i, chunk in enumerate(chunks):
            entities, relations = self.extract_entities_relations(chunk['text'])
            for entity in entities:
                if isinstance(entity, dict) and "name" in entity and "type" in entity:
                    texts.append(f"实体：{entity['name']}:类型：{entity['type']}")
                    metadatas.append({"type": entity["type"], "source": f"{chunk['source']}"})
            for relation in relations:
                if isinstance(relation, dict) and "source" in relation and "target" in relation and "type" in relation:
                    texts.append(f"关系：{relation['source']} -> {relation['target']}:类型：{relation['type']}")
                    metadatas.append({"type": relation["type"], "source": f"{chunk['source']}"})
        len_texts = len(texts)
        if len_texts !=0:
            documents = [Document(page_content=texts[i], metadata=metadatas[i]) for i in range(len_texts)]
            graphstore.add_documents(documents,ids=[str(uuid.uuid4()) for _ in range(len_texts)])
        
        # len_chunks = len(chunks)
        # if len_chunks !=0:
        #     documents = [Document(page_content=chunks[i]['text'], metadata={"source": chunks[i]['source']}) for i in range(len_chunks)]
        #     vectorstore.add_documents(documents,ids=[str(uuid.uuid4()) for _ in range(len_chunks)])

    def extract_entities_relations(self, text):
        """使用大模型提取实体和关系"""
        prompt = (f"从以下文本中提取医学实体和关系，实体类型包括药物、疾病、指南、患者。关系类型包括INDICATED_FOR（药物-适应症）"
                  f"、TREATED_WITH（疾病-治疗方案）、RECOMMENDS（指南-推荐）等。返回格式为JSON，包含entities和relations两个字段。"
                  f"例如：{{entities: [{{name: '药物A', type: '药物'}}, {{name: '疾病B', type: '疾病'}}, {{name: '指南C', type: '指南'}}],"
                  f" relations: [{{source: '药物A', target: '疾病B', type: 'INDICATED_FOR'}}, {{source: '疾病B', target: '指南C', type: 'RECOMMENDS'}}]}}"
                  f"请严格保证按照给定格式返回，不能包含其他内容。"
                  f"\n\n文本：{text}")
        try:
            response = self.llm.invoke(prompt)
            result = json.loads(response.content)
            return result.get('entities', []), result.get('relations', [])
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析错误: {e}")
            return [], []
        except Exception as e:
            logger.error(f"提取实体和关系时出错: {e}")
            return [], []
