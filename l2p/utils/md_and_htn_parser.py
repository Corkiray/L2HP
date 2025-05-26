"""
This file contains collection of functions for extracting/parsing markdown datastructures from LLM output
"""
import re
from .pddl_parser import *
from .pddl_types import *

def extract_section_by_name(markdown_text: str, title: str, level : int = 1) -> str:
    """
    Extracts the content of a specific section in markdown text based on the title and level.

    Args:
        markdown_text (str): The raw markdown text.
        title (str): The title of the section to extract.
        level (str): The markdown level of the title (e.g., 1="#", 2="##", 3="###").

    Returns:
        str: The content of the section, or None if the section is not found.
    """
    pattern = rf"(?:^|\n){level*'#'} {re.escape(title)}\n(.*?)(?=\n{level*'#'} |\Z)"
    match = re.search(pattern, markdown_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return ""

def split_sections(markdown_text: str, level = 1) -> list[str]:
    """
    Splits the markdown text into sections based on the headings.

    Args:
        markdown_text (str): The raw markdown text.
        level (str): The markdown level of the headings to split by (e.g., 1="#", 2="##", 3="###").

    Returns:
        list: A list of sections, each section is a string.
    """
    pattern = rf"(?:^|\n){level*'#'} (.+?)(?=\n{level*'#'} |\Z)"
    sections = re.findall(pattern, markdown_text, re.DOTALL)
    return [section.strip() for section in sections]

def extract_list(markdown_text: str) -> list[str]:
    """
    Parses a markdown list from the given text.

    Args:
        markdown_text (str): The raw markdown text containing a list.

    Returns:
        list: A list of items extracted from the markdown list.
    """
    items = re.findall(r"^\s*[-*] (.+)$", markdown_text, re.MULTILINE)
    return [item.strip() for item in items]

def prune_unsupported_keywords(dictionary: dict, unsupported_keywords: list = ["object", "pddl", "lisp"]) -> dict:
    """
    Prune from a dict keywords that are not supported
    Args:
        types_hierarchy (dict): A dictionary of types.
    Returns:
        dict: The pruned dictionary of types.
    """
    return {
        name: description
        for name, description in dictionary.items()
        if name not in unsupported_keywords
    }
    
def parse_tasks(llm_response: str) -> dict[str, HPDLTask]:
    """
    Extracts HTN Tasks from LLM response and returns it as a list of dict strings

    Args:
        llm_response (str): The LLM output.

    Returns:
        list[task]): list of tasks
    """
    new_tasks = dict()

    for p_line in llm_response.split("\n"):
        if ("." not in p_line or not p_line.split(".")[0].strip().isdigit()) and not (
            p_line.startswith("-") or p_line.startswith("(") or p_line.startswith("*")
        ):
            if len(p_line.strip()) > 0:
                print(f'[WARNING] unable to parse the line: "{p_line}"')
            continue
        task_info = p_line.split(": ")[0].strip(" 1234567890.(-*)`").split(" ")
        task_name = task_info[0]
        task_desc = p_line.split(": ")[1].strip() if ": " in p_line else ""

        # get the predicate type info
        if len(task_info) > 1:
            task_params_info = task_info[1:]
            task_params_info = [
                l.strip(" ()`") for l in task_params_info if l.strip(" ()`")
            ]
        else:
            task_params_info = []
        params = OrderedDict()
        next_is_type = False
        upcoming_params = []

        for p in task_params_info:
            if next_is_type:
                if p.startswith("?"):
                    print(
                        f"[WARNING] `{p}` is not a valid type for a variable, but it is being treated as one. Should be checked by syntax check later."
                    )
                for up in upcoming_params:
                    params[up] = p
                next_is_type = False
                upcoming_params = []
            elif p == "-":
                next_is_type = True
            elif p.startswith("?"):
                upcoming_params.append(p)  # the next type will be for this variable
            else:
                print(
                    f"[WARNING] `{p}` is not corrrectly formatted. Assuming it's a variable name."
                )
                upcoming_params.append(f"?{p}")
        if next_is_type:
            print(
                f"[WARNING] The last type is not specified for `{p_line}`. Undefined are discarded."
            )
        if len(upcoming_params) > 0:
            print(
                f"[WARNING] The last {len(upcoming_params)} is not followed by a type name for {upcoming_params}. These are discarded"
            )

        # generate a clean version of the predicate
        clean = f"({task_name} {' '.join([f'{k} - {v}' for k, v in params.items()])}): {task_desc}"

        # drop the index/dot
        p_line = p_line.strip(" 1234567890.-`")
        new_tasks[task_name] = {
            "name": task_name,
            "desc": task_desc,
            "raw": p_line,
            "params": params,
            "clean": clean,
        }
        
     
    return new_tasks

def parse_method(llm_response: str, method_name: str) -> HPDLMethod:
    """
    Parse a method from a given LLM output.

    Args:
        llm_response (str): The LLM output.
        method_name (str): The name of the method.

    Returns:
        Method: The parsed method.
    """
    from .pddl_parser import parse_params
    parameters, _ = parse_params(llm_response)
    try:
        task = (
            llm_response.split("Method Task\n")[1]
            .split("###")[0]
            .strip(" `\n")
        )
    except:
        raise Exception(
            "Could not find the 'Method Task' section in the output. Provide the entire response, including all headings even if some are unchanged."
        )
    try:
        subtasks = (
            llm_response.split("Method Ordered Subtasks\n")[1]
            .split("###")[0]
            .strip(" `\n")
        )
        subtasks = substract_logical_expression(subtasks)
    except:
        raise Exception(
            "Could not find the 'Method Ordered Subtasks' section in the output. Provide the entire response, including all headings even if some are unchanged."
        )
    return {
        "name": method_name,
        "params": parameters,
        "task": task,
        "tasks": subtasks,
    }

def parse_methods(raw_methods: str) -> list[HPDLMethod | HDDLMethod]:
    """
    Parses methods from LLM response and returns them as a list of dictionaries.

    Args:
        llm_response (str): The LLM output.

    Returns:
        list[dict[str, str]]: List of methods in dictionaries.
    """
    methods = list()
    raw_methods_list = split_sections(raw_methods, level=2)
    for j in raw_methods_list:
        method_name, rest_of_string = j.split("\n", 1)
        method = parse_method(llm_response=rest_of_string, method_name=method_name)
        methods.append(method)
    return methods


def substract_logical_expression(llm_response: str) -> str:
    """
    Substracts logical expression from LLM response and returns it as a string

    Args:
        llm_response (str): The LLM output.

    Returns:
        states (list[dict[str,str]]): list of initial states in dictionaries
    """
    # Find all substrings enclosed by parentheses
    matches = re.findall(r'\((.*)\)', llm_response, re.DOTALL)

    # Return the largest match or an empty string if no matches
    if matches:
        return f"({max(matches, key=len)})"
    else:
        raise ValueError("Could not find the logical expression in the LLM output. Provide the entire response, including all headings even if some are unchanged.")

def parse_list_of_predicates(llm_output) -> list[Predicate]:
    """
    Parses new predicates from LLM into Python format.

    The LLM Output provided has to contain a structured list of items.
    """
    new_predicates = list()

    for p_line in llm_output.split("\n"):
        if ("." not in p_line or not p_line.split(".")[0].strip().isdigit()) and not (
            p_line.startswith("-") or p_line.startswith("(") or p_line.startswith("*")
        ):
            if len(p_line.strip()) > 0:
                print(f'[WARNING] unable to parse the line: "{p_line}"')
            continue
        predicate_info = p_line.split(": ")[0].strip(" 1234567890.(-*)`").split(" ")
        predicate_name = predicate_info[0]
        predicate_desc = p_line.split(": ")[1].strip() if ": " in p_line else ""

        # get the predicate type info
        if len(predicate_info) > 1:
            predicate_type_info = predicate_info[1:]
            predicate_type_info = [
                l.strip(" ()`") for l in predicate_type_info if l.strip(" ()`")
            ]
        else:
            predicate_type_info = []
        params = OrderedDict()
        next_is_type = False
        upcoming_params = []

        for p in predicate_type_info:
            if next_is_type:
                if p.startswith("?"):
                    print(
                        f"[WARNING] `{p}` is not a valid type for a variable, but it is being treated as one. Should be checked by syntax check later."
                    )
                for up in upcoming_params:
                    params[up] = p
                next_is_type = False
                upcoming_params = []
            elif p == "-":
                next_is_type = True
            elif p.startswith("?"):
                upcoming_params.append(p)  # the next type will be for this variable
            else:
                print(
                    f"[WARNING] `{p}` is not corrrectly formatted. Assuming it's a variable name."
                )
                upcoming_params.append(f"?{p}")
        if next_is_type:
            print(
                f"[WARNING] The last type is not specified for `{p_line}`. Undefined are discarded."
            )
        if len(upcoming_params) > 0:
            print(
                f"[WARNING] The last {len(upcoming_params)} is not followed by a type name for {upcoming_params}. These are discarded"
            )

        # generate a clean version of the predicate
        clean = f"({predicate_name} {' '.join([f'{k} - {v}' for k, v in params.items()])}): {predicate_desc}"

        # drop the index/dot
        p_line = p_line.strip(" 1234567890.-`")
        new_predicates.append(
            {
                "name": predicate_name,
                "desc": predicate_desc,
                "raw": p_line,
                "params": params,
                "clean": clean,
            }
        )

    return new_predicates

def parse_actions_list(raw_actions_list: list[str]) -> list[Action]:
    """
    Parses actions from a list of strings and returns them as a list of Action objects.

    Args:
        raw_actions_list (list[str]): List of action strings.

    Returns:
        list[Action]: List of Action objects.
    """
    actions = []
    for action_str in raw_actions_list:
        action_name, rest_of_string = action_str.split("\n", 1)
        action = parse_md_action(rest_of_string, action_name)
        actions.append(action)
    return actions

def parse_md_action(markdown_text: str, action_name: str) -> Action:
    """
    Parses a single action from markdown text and returns it as a dictionary.

    Args:
        markdown_text (str): The markdown text containing the action details.
        action_name (str): The name of the action.

    Returns:
        Action: The parsed action as a dictionary.
    """
    from .pddl_parser import parse_params
    parameters, _ = parse_params(markdown_text)
    
    preconditions = (
        markdown_text.split("Action Preconditions\n")[1]
        .split("###")[0]
        .strip(" `\n")
    )
    preconditions = substract_logical_expression(preconditions)
   
    effects = (
            markdown_text.split("Action Effects\n")[1]
            .split("###")[0]
            .strip(" `\n")
        )
    effects = substract_logical_expression(effects)
    
    return {
        "name": action_name,
        "params": parameters,
        "preconditions": preconditions,
        "effects": effects,
    }