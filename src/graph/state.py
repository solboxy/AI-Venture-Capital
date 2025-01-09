# agent_state_utils.py

from typing import Annotated, Any, Dict, Sequence, TypedDict
import operator
import json

from langchain_core.messages import BaseMessage


def combine_dictionaries(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Combine two dictionaries into a single dictionary."""
    return {**a, **b}


class TradingAgentState(TypedDict):
    """
    Represents the shared state for the trading agents.
    - messages: A list of agent-generated messages
    - data: A dictionary of relevant data (prices, metrics, portfolio, etc.)
    - metadata: A dictionary of additional state info (e.g., show_reasoning, config)
    """
    messages: Annotated[Sequence[BaseMessage], operator.add]
    data: Annotated[Dict[str, Any], combine_dictionaries]
    metadata: Annotated[Dict[str, Any], combine_dictionaries]


def show_agent_reasoning(output: Any, agent_name: str) -> None:
    """
    Pretty-print the agent's reasoning or outputs in JSON format to the console.
    If `output` is already JSON-serializable, it will be formatted nicely.
    If `output` is a string containing valid JSON, it will be parsed and formatted.
    Otherwise, it prints the string or object representation.
    """
    print(f"\n{'=' * 10} {agent_name.center(28)} {'=' * 10}")

    def convert_to_serializable(obj: Any) -> Any:
        """
        Recursively convert various objects (Pandas, custom classes, etc.)
        into JSON-serializable structures.
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
