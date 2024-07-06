from __future__ import annotations

import csv
import json
import pandas as pd
from typing import Any

from eulerchain.agents.llm_agent_base import LLMBaseAgent

class CSVAgent(LLMBaseAgent):
    """
    Example usage for CSV agent:
        csv_config = LLMConfig(agent_type="csv", config={"file_path": "data.csv"})
        csv_agent = AgentFactory.create_agent(csv_config)
        print(csv_agent.run("some query", None))
    """
    def __init__(self, file_path: str):
        self.file_path = file_path

    def run(self, prompt: str, input_data: Any) -> Any:
        try:
            with open(self.file_path, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                results = [row for row in reader if prompt in row.values()]
            return results
        except FileNotFoundError:
            raise FileNotFoundError(f"File {self.file_path} not found")

class SQLAgent(LLMBaseAgent):
    """
    Example usage for SQL agent:
        sql_config = LLMConfig(agent_type="sql", config={"db_path": "database.db"})
        sql_agent = AgentFactory.create_agent(sql_config)
        print(sql_agent.run("SELECT * FROM users WHERE name='John'", None))
    """
    def __init__(self, db_path: str):
        try:
            import sqlite3
            self.connection = sqlite3.connect(db_path)
        except ModuleNotFoundError:
            raise ModuleNotFoundError("sqlite3 module not found. Please install it using: `pip install sqlite3`")

    def run(self, prompt: str, input_data: Any) -> Any:
        cursor = self.connection.cursor()
        cursor.execute(prompt)
        return cursor.fetchall()

class JSONAgent(LLMBaseAgent):
    """
    Example usage for JSON agent:
        json_config = LLMConfig(agent_type="json", config={"file_path": "data.json"})
        json_agent = AgentFactory.create_agent(json_config)
        print(json_agent.run("John", None))
    """
    def __init__(self, file_path: str):
        try:
            with open(file_path) as jsonfile:
                self.data = json.load(jsonfile)
        except FileNotFoundError:
            raise FileNotFoundError(f"File {file_path} not found")

    def run(self, prompt: str, input_data: Any) -> Any:
        results = [item for item in self.data if prompt in json.dumps(item)]
        return results

class DataFrameAgent(LLMBaseAgent):
    """
    Example DataFrame Agent:
        data = {"name": ["Alice", "Bob", "John"], "age": [25, 30, 45]}
        df = pd.DataFrame(data)
        dataframe_config = LLMConfig(agent_type="dataframe", config={"dataframe": df})
        dataframe_agent = AgentFactory.create_agent(dataframe_config)
        print(dataframe_agent.run("name == 'John'", None))
    """
    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe

    def run(self, prompt: str, input_data: Any) -> Any:
        if isinstance(self.dataframe, pd.DataFrame):
            return self.dataframe.query(prompt)
        else:
            raise ValueError("dataframe is not a correct instance of DataFrame")

class PythonAgent(LLMBaseAgent):
    """
    Example of python agent:
        python_config = LLMConfig(agent_type="python", config={})
        python_agent = AgentFactory.create_agent(python_config)
        print(python_agent.run("result = 1 + 2", None))
    """
    def run(self, prompt: str, input_data: Any) -> Any:
        exec_globals = {}
        exec(prompt, exec_globals)
        return exec_globals
