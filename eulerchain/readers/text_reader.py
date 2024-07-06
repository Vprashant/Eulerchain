from typing import Optional, Any, Dict, List
from pathlib import Path
from eulerchain.eulerchain_root.documents.document import Document
from eulerchain.readers.base_reader import BaseReader
from eulerchain.eulerchain_root.pydantic_v2 import BaseModel, Field
from eulerchain.eulerchain_root.documents.document_encoding import file_encoding_detection

import logging

logger = logging.getLogger(__name__)

class TextReader(BaseReader):
    """Read text file.
    Args:
        file_path: Path to the file to load.
        encoding: File encoding to use. If `None`, the file will be loaded
        with the default system encoding.
        autodetect_encoding: Whether to try to autodetect the file encoding
            if the specified encoding fails.
    """   
    file_path: str = None
    encoding: Optional[str] = None,
    autodetect_encoding: bool = Field(default=None , init= False, description=''),

    def load(self) -> List[Document]:
        """Load from file path."""
        text = ""
        try:
            with open(self.file_path, encoding=self.encoding) as f:
                text = f.read()
        except UnicodeDecodeError as e:
            if self.autodetect_encoding:
                detected_encodings = file_encoding_detection(self.file_path)
                for encoding in detected_encodings:
                    logger.debug(f"Trying encoding: {encoding.encoding}")
                    try:
                        with open(self.file_path, encoding=encoding.encoding) as f:
                            text = f.read()
                        break
                    except UnicodeDecodeError:
                        continue
            else:
                raise RuntimeError(f"Error loading {self.file_path}") from e
        except Exception as e:
            raise RuntimeError(f"Error loading {self.file_path}") from e

        metadata = {"source": self.file_path}
        return [Document(page_content=text, metadata=metadata)]
