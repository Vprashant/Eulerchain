"""
filename: table_handler.py
Author: Prashant Verma
email: 
"""
from typing import Dict, Any, List , Optional
from pydantic import BaseModel
import json
import os

class TableHandler:
    """_summary_
    Args:
        BaseModel (_type_): _description_
    """

    def __init__(self,
                 filename: str,
                 stream: Optional[bool] = False, 
                 pages: Optional[str] = "all", 
                 encoding: str = "utf-8", 
                 multiple_tables: Optional[bool] = True,
                 guess: Optional[bool] = False,
                 pandas_options: Optional[Dict[str, Any]] = None,
                  )->None:
        """_summary_

        Args:
            filename (Optional[str], optional): _description_. Defaults to None.
            stream (Optional[bool], optional): _description_. Defaults to True.
            pages (Optional[str], optional): _description_. Defaults to "all".
            encoding (Optional[str], optional): _description_. Defaults to None.
            multiple_tables (Optional[bool], optional): _description_. Defaults to True.
            guess (Optional[bool], optional): _description_. Defaults to False.
            pandas_options (_type_, optional): _description_. Defaults to {'header': None}.

        Raises:
            ModuleNotFoundError: _description_
        """
        ...
        try:
            from tabula.io import read_pdf
            # self.tabular_data = tabula.read_pdf(filename, stream=stream, pages=pages, \
            #                                             multiple_tables=multiple_tables, encoding=encoding, guess=guess, pandas_options=pandas_options)
            self.tabular_data = read_pdf(filename, pages='all', pandas_options={'header': None})
        except ModuleNotFoundError:
            raise ModuleNotFoundError(f"tabula is not found. please 'pip install tabula-py'")
        
        self.tbl_status = False if not self.tabular_data else True
        

    def table_header_generator(self, table_data)-> str:
        """_summary_
        Args:
            table_data (_type_): _description_
        Returns:
            str: createing header string for comparison 
        """
        contextual_info_columns = table_data.columns.values
        contextual_info = " ".join(str(cell) for column in table_data[contextual_info_columns].values for cell in column)
        return contextual_info
    
    def tbl_context_filter(self, table_content) -> str:
        """_summary_
        Args:
            table_content (_type_): _description_
        Returns:
            str: _description_
        """
        clr_content = table_content.replace('nan', '')
        text = clr_content.replace(", ", '')
        text = text.split(' ')
        return ' '.join(text)

    
    def json_generator(self):
        """_summary_
        Args:
            tables (_type_): create table structure from unstratcuture table context and stroring
            it into json files.
        """
        i = 0
        tables = self.tabular_data
        print(len(tables))

        current_directory = os.path.dirname(os.path.abspath(__file__))
        table_json_path = os.path.join(current_directory, "resource")

        pass
        if not self.tbl_status:
            print(f'[INFO]: No tables found in the document')
            return False
        
        while i < len(tables):
            table_data = tables[i].to_dict()
            for col in range(len(table_data)):
                table_data['column_'+str(col)] = table_data.pop(col)
                for row in range(len(table_data['column_'+str(col)])):
                    table_data['column_'+str(col)]['row_'+str(row)] = table_data['column_'+str(col)].pop(row)
            table_data['header'] = self.tbl_context_filter(self.table_header_generator(tables[i]))
            
            print(table_data)
            with open(str(table_json_path)+'\\tbl_to_json_'+str(i)+'.json', 'a',encoding="utf-8") as file:
                json.dump(table_data, file)
            i+=1
        return True                


        


        
    