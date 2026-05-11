from langchain_text_splitters import RecursiveCharacterTextSplitter
from backend.config import get_app_config

class TextSplitter:
    def __init__(self):
        config = get_app_config().vector_db
        self.chunk_size = config.chunk_size
        self.chunk_overlap = config.chunk_overlap
    
    def split_text(self, text, source):
        """将文本按指定大小分块，返回带有来源信息的分块列表"""
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            chunk = text[start:end]
            chunks.append({
                'text': chunk,
                'source': source,
                'start': start,
                'end': end
            })
            start += self.chunk_size - self.chunk_overlap
        
        return chunks
    
    def split_text_into_chunks(self, text, source):
        """使用 RecursiveCharacterTextSplitter 分块，返回带编号的分块列表"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,  # 每个chunk的最大字符数（可根据模型上下文调整）
            chunk_overlap=self.chunk_overlap,  # 相邻chunk重叠100字符，避免拆分切断语义
            separators=["\n\n", "\n", ".", "！", "？", ".", "!", "?", ",","，"]  # 优先按段落/句子拆分
        )
        chunks = text_splitter.split_text(text)
        # 给每个chunk加编号，方便后续整合
        chunks=[{
                'text': chunk,
                'source': source,
            } for chunk in chunks]
        return chunks