# agent_state_utils.py

from typing import Annotated, Any, Dict, Sequence, TypedDict
import operator
import json
import logging

from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)

def combine_dictionaries(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merges two dictionaries into a new dictionary.

    Args:
        a (Dict[str, Any]): The first dictionary.
        b (Dict[str, Any]): The second dictionary.

    Returns:
        Dict[str, Any]: A dictionary containing the union of both input dictionaries.
                        If there are conflicting keys, values from b will overwrite those in a.
    """
    return {**a, **b}


class TradingAgentState(TypedDict):
    """
    Represents the shared state for trading agents in the multi-agent pipeline.

    Attributes:
        messages (Annotated[Sequence[BaseMessage], operator.add]):
            A list of agent-generated messages.
        data (Annotated[Dict[str, Any], combine_dictionaries]):
            A dictionary of relevant data (e.g. prices, metrics, portfolio).
        metadata (Annotated[Dict[str, Any], combine_dictionaries]):
            A dictionary of additional state info (e.g., show_reasoning, configs).
    """
    messages: Annotated[Sequence[BaseMessage], operator.add]
    data: Annotated[Dict[str, Any], combine_dictionaries]
    metadata: Annotated[Dict[str, Any], combine_dictionaries]


def show_agent_reasoning(output: Any, agent_name: str) -> None:
    """
    Pretty-prints the agent's reasoning or outputs in JSON format to the console.

    This function attempts to parse and format the output as JSON. 
    If the output is a string containing valid JSON, it will be parsed; otherwise, 
    the raw string/object will be printed.

    Args:
        output (Any): The data or message to be displayed.
        agent_name (str): A descriptive name of the agent for console labeling.
    """
    print(f"\n{'=' * 10} {agent_name.center(28)} {'=' * 10}")

    def convert_to_serializable(obj: Any) -> Any:
        """
        Recursively converts various objects (Pandas, custom classes, etc.)
        into JSON-serializable structures.

        Args:
            obj (Any): The object to serialize.

        Returns:
            Any: JSON-serializable representation of the object.
        """
        if hasattr(obj, 'to_dict'):  # Handle Pandas Series/DataFrame
            return obj.to_dict()
        elif hasattr(obj, '__dict__'):  # Handle custom objects
            return obj.__dict__
        elif isinstance(obj, (int, float, bool, str)):
            return obj
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        else:
            return str(obj)  # Fallback to string representation

    if isinstance(output, (dict, list)):
        # Convert the output to a JSON-serializable format
        serializable_output = convert_to_serializable(output)
        print(json.dumps(serializable_output, indent=2))
    else:
        try:
            # If it's a JSON string, parse and pretty-print it
            parsed_output = json.loads(output)
            print(json.dumps(parsed_output, indent=2))
        except json.JSONDecodeError:
            # Fallback to the raw string if not valid JSON
            print(output)

    print("=" * 48)
