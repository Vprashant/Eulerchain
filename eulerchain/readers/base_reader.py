from typing import Any, Optional
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod
from langchain.docstore.document import Document


class BaseReader(BaseModel, ABC):
    filepath : str

    @abstractmethod
    def load(filepath: str)-> list[Document]:
        """ """
        pass

