from src.app.python.constant.global_data import GlobalData 
from src.app.python.constant.project_constant import Constant as constant
from typing import Dict, type, Optional, List, Any
from pydantic import Field
from langchain_community.llms.vllm import VLLM
import time
import asyncio



class QueryAsyncEngine(object):
    """"Responsible to handle multi query request. """
    def __init__(self,
                 query: str,
                 no_gpu_request: Optional[List[str]] = None,
                 start_engine_loop: bool = True,
                 max_log_len: int | None = None,
                 engine_use_ray: bool = True,
                 **field_kwarg: Any,
                 ) -> None:
        
        self.query = query
        self._queue = asyncio.Queue()
        
        ...

    def vllm_initialization(self):
        try:
            llm = VLLM(model= "", 
                                            trust_remote_code= constant.SUCCESS_STATUS,
                                            max_new_tokens=128,
                                            top_k=10,
                                            top_p=0.95,
                                            temperature=0)
            
            llm.invoke("What is the future of AI?")
        except ModuleNotFoundError as e:
            raise ValueError(f" VLLM Modlue Not Found !! {e}")
        except Exception as e:
            raise ValueError(f' Exception occured in vllm initializtion.  {e}')
        
        

        

    
        


