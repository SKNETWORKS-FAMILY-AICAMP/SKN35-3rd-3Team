from src.graph.state import State
from src.const.model import create_llm , create_llm_gmn
from src.prompt.general_prompt import GENERAL_PROMPT
from langchain_core.messages import SystemMessage , HumanMessage

model = create_llm_gmn(temperature=0.3)

def general_agent(state : State):
    question = HumanMessage(content=state["origin_query"])

    response = model.invoke([
        SystemMessage(content=GENERAL_PROMPT),
        *state["messages" , []],
        question
    ])

    return {"messages": [question, response] , "answer" : response.txt}