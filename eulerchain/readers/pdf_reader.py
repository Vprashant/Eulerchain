from typing import Optional, Any, Dict, List
from pathlib import Path
from eulerchain.eulerchain_root.documents.document import Document
from eulerchain.readers.base_reader import BaseReader

class PDFReader(BaseReader):
    def load(self, file: Path, 
                  extra_info: Optional[Dict] = None) -> List[Document]:
        
        try:
            import pypdf
        except ImportError as e:
            raise ImportError("pypdf is not module not found: 'pip install pypdf' ") from e.error()
        
        with open(file, "rb") as f_read:
            pdf = pypdf.PdfReader(f_read)
            no_pages = len(pdf.pages)

            docs = []
            for page in range(no_pages):
                page_text = pdf.pages[page].extract_text()
                page_label = pdf.page_labels[page]

                metadata = {"page_label": page_label, "file_name": file.name}

                if extra_info is not None:
                    metadata.update(extra_info)
                docs.append(Document(text = page_text, ))

        return docs