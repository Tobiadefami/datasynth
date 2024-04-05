import random
from datasynth import EXAMPLE_DIR
import os
import json
from typing import Union
from settings import SEPARATOR

separator = SEPARATOR

def generate_population(datatype: str) -> Union[list[str], list[dict[str, str]]]:
    population: Union[list[str], list[dict[str, str]]] = []
    example_file: str = os.path.join(EXAMPLE_DIR, f"{datatype}.json")

    if os.path.exists(example_file):
        with open(example_file, "r") as file:
            data = json.load(file)
        # Check if the JSON structure is a dictionary containing a list (e.g., under a key like "examples")
        if isinstance(data, dict) and "examples" in data:
            population = data["examples"]
        elif isinstance(data, list):
            population = data
    return population


def convert_to_string(example: Union[str, dict[str, str]]) -> str:
    if isinstance(example, str):
        return example
    elif isinstance(example, dict):
        new_examples = "\n".join(f"{key}: {value}" for key, value in example.items())
        return new_examples
    elif isinstance(example, list):
        # Process each dictionary in the list and join them with a separator
        dict_strings = ["\n".join(f"{key}: {value}" for key, value in d.items()) for d in example]
        return "\n---\n".join(dict_strings)


def populate_few_shot(
    population: Union[list[str], list[dict[str, str]]], sample_size: int
) -> str:
    sample = random.sample(population, k=sample_size)
    sample_text = separator.join([convert_to_string(example) for example in sample])
    few_shot = f"Reference Format:\n{sample_text}\n"
    return few_shot
