import os
import pdfplumber
import logging

logger = logging.getLogger(__name__)

class FileProcessor:
    def process_file(self, file_path):
        """处理文件，提取内容和来源信息"""
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.txt':
            content = self._process_txt(file_path)
        elif file_extension == '.pdf':
            content = self._process_pdf(file_path)
        else:
            logger.warning(f"不支持的文件类型: {file_extension}")
            return {}
        
        return {
            'content': content,
            'source': file_path
        }
    
    def _process_txt(self, file_path):
        """处理TXT文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    
    def _process_pdf(self, file_path):
        """处理PDF文件"""
        content = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    content += page_text
        return content