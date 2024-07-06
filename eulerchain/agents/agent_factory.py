from eulerchain.agents.llm_agent_base import LLMBaseAgent, LLMConfig
from typing import Type, Dict

class AgentFactory:
    """
        ````Register agents````
        AgentFactory.register_agent("openai", OpenAIAgent)
        AgentFactory.register_agent("csv", CSVAgent)
        AgentFactory.register_agent("sql", SQLAgent)
        AgentFactory.register_agent("json", JSONAgent)
        AgentFactory.register_agent("dataframe", DataFrameAgent)
        AgentFactory.register_agent("python", PythonAgent)
    """
    _agents: Dict[str, Type[LLMBaseAgent]] = {}

    @classmethod
    def register_agent(cls, agent_type: str, agent_class: Type[LLMBaseAgent]):
        cls._agents[agent_type] = agent_class

    @classmethod
    def create_agent(cls, config: LLMConfig) -> LLMBaseAgent:
        agent_class = cls._agents.get(config.agent_type)
        if not agent_class:
            raise ValueError(f"Agent type {config.agent_type} not registered.")
        return agent_class(**config.config)


