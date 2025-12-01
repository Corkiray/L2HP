"""
This file contains collection of functions for extracting/parsing HTN information from text.
"""
from .pddl_parser import *
from .pddl_types import *
from .md_parser import *

def parse_tasks(text: str) -> dict[str, HPDLTask]:
    """
    Extracts HTN Tasks from LLM response and returns it as a dictionary of tasks

    Args:
        text (str): The text containing task definitions in list format.

    Returns:
        dict[str, HPDLTask]: Dictionary of tasks with their names as keys
    """
    new_tasks = dict()

    for p_line in text.split("\n"):
        # Skip empty lines
        if not p_line.strip():
            continue
            
        # Clean the line from list markers and extra spaces
        p_line = p_line.replace("`", "")
        p_line = p_line.strip(" -.*")
        
        # Skip if line is empty after cleaning
        if not p_line:
            continue
            
        # Split into task definition and description
        if ": " not in p_line:
            print(f'[WARNING] unable to parse the line: "{p_line}"')
            continue
            
        task_def, task_desc = p_line.split(": ", 1)
        
        # Remove surrounding parentheses if present
        task_def = task_def.strip("()")
        
        # Split task definition into name and parameters
        parts = task_def.split()
        if not parts:
            continue
            
        task_name = parts[0]
        task_params_info = parts[1:]
        
        # Parse parameters
        params = OrderedDict()
        next_is_type = False
        upcoming_params = []

        for p in task_params_info:
            if next_is_type:
                if p.startswith("?"):
                    print(f"[WARNING] `{p}` is not a valid type for a variable")
                for up in upcoming_params:
                    params[up] = p
                next_is_type = False
                upcoming_params = []
            elif p == "-":
                next_is_type = True
            elif p.startswith("?"):
                upcoming_params.append(p)
            else:
                print(f"[WARNING] `{p}` is not correctly formatted")
                upcoming_params.append(f"?{p}")

        # Generate clean version
        clean = f"({task_name} {' '.join([f'{k} - {v}' for k, v in params.items()])}): {task_desc}"

        new_tasks[task_name] = {
            "name": task_name,
            "desc": task_desc,
            "raw": p_line,
            "params": params,
            "clean": clean,
        }
        
    return new_tasks

def parse_method(text: str, method_name: str) -> HPDLMethod:
    """
    Parse a method from a given LLM output.

    Args:
        llm_response (str): The LLM output.
        method_name (str): The name of the method.

    Returns:
        Method: The parsed method.
    """
    
    clean_text = clean_markdown_comments(text)
    parameters, _ = parse_params(clean_text)
    try:
        task = (
            clean_text.split("Method Task")[1].split("###")[0].split(";")[0]
        )
        task = substract_logical_expression(task)
    except:
        raise Exception(
            "Could not find the 'Method Task' section in the output. Provide the entire response, including all headings even if some are unchanged."
        )
    try:
        subtasks = (
            text.split("Method Ordered Subtasks")[1]
            .split("###")[0]
            .strip(" `\n")
        )
        subtasks = substract_logical_expression(subtasks)
    except:
        raise Exception(
            "Could not find the 'Method Ordered Subtasks' section in the output. Provide the entire response, including all headings even if some are unchanged."
        )
        
    return {
        'name': method_name,
        'params': parameters,
        'task': task,
        'ordered_subtasks': subtasks,
        'raw': text,
        'desc': None
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
        method = parse_method(text=rest_of_string, method_name=method_name)
        methods.append(method)
    return methods

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
    parameters, _ = parse_params(markdown_text)
    
    preconditions = (
        markdown_text.split("Action Preconditions")[1]
        .split("###")[0]
        .strip(" `\n")
    )
    preconditions = substract_logical_expression(preconditions)
   
    effects = (
            markdown_text.split("Action Effects")[1]
            .split("###")[0]
            .strip(" `\n")
        )
    effects = substract_logical_expression(effects)
    
    return {
        "name": action_name,
        "params": parameters,
        "preconditions": preconditions,
        "effects": effects,
        "raw": markdown_text
        }
    
def parse_list_of_predicates(text: str) -> list[Predicate]:
    """
    Parses new predicates from LLM into Python format.

    The LLM Output provided has to contain a structured list of items.
    """
    new_predicates = list()

    for p_line in text.split("\n"):
        if ("." not in p_line or not p_line.split(".")[0].strip().isdigit()) and not (
            p_line.startswith("-") or p_line.startswith("(") or p_line.startswith("*")
        ):
            if len(p_line.strip()) > 0:
                print(f'[WARNING] unable to parse the line: "{p_line}"')
            continue
        predicate_info = p_line.split(": ")[0].strip(" 1234567890.(-*)`").split(" ")
        predicate_name = predicate_info[0]
        predicate_desc = p_line.split(": ")[1].strip() if ": " in p_line else ""
        
        if len(predicate_name) == 0 or len(predicate_info) == 0:
            print(f'[WARNING] unable to parse the line: "{p_line}"')
            continue

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
        if len(params) == 0:
            clean = f"({predicate_name}): {predicate_desc}"
        else:
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


def parse_goal_htn(llm_output: str) -> list[dict[str, str]]:
    """
    Extracts goal (PDDL-goal) from markdown text and returns it as a string

    Parameters:
        llm_output (str): raw LLM output

    Returns:
        states (list[dict[str,str]]): list of goal states in dictionaries
    """
    
    goal_section = extract_section_by_name(llm_output, "GOAL")
    goal_output = extract_section_by_name(goal_section, "OUTPUT", level=2)
    goal_combined = combine_blocks(goal_output)
    goal_clean = remove_comments(goal_combined)
    goal_parsed = parse_pddl(f"({goal_clean})")
    goal_formatted = parse_task_states(goal_parsed[0])
      
    return goal_formatted

def parse_list_of_methods(text: str) -> list[HPDLMethod]:
    """
    Parses a list of method names from LLM response.

    Args:
        llm_response (str): The LLM output.
    Returns:
        list[HPDLMethod]: List of methods.
    """
    methods = list()
    for line in text.split("\n"):
        line = line.strip(" -`")
        if len(line) == 0:
            continue
        methods.append({
            'name': line,
            'params': OrderedDict(),
            'task': None,
            'ordered_subtasks': None,
            'raw': line,
            'desc': None
        })
    return methods


def parse_ordered_subtasks(llm_output: str) -> str:
    """Parses precondition string from LLM output"""
    try:
        subtasks = (
            llm_output.split("Subtasks\n")[1]
            .split("###")[0]
            .split("```")[1]
            .strip(" `\n")
        )

        return subtasks
    except:
        raise Exception(
            "Could not find the 'Preconditions' section in the output. Provide the entire response, including all headings even if some are unchanged."
        )
        
        
def parse_new_nl_actions(llm_output) -> dict[str,str]:
    """
    Parses new NL actions from LLM into Python format.
    The LLM Output provided has to contain a structured list of items.
    Args:
        llm_output (str): The LLM output.
    Returns:
        dict[str,str]: Dictionary of new NL actions with action signature as key and description as value.
    """
    new_actions = {}
    try:
        heading = (
            llm_output.split("New Actions\n")[1].strip().split("###")[0]
        )
    except:
        raise Exception(
            "Could not find the 'New Actions' section in the output. Provide the entire response, including all headings even if some are unchanged."
        )
    output = combine_blocks(heading)

    for p_line in output.split("\n"):
        p_line = p_line.strip()
        if not p_line or p_line.startswith("```"):
            continue  # skip empty lines and code block markers

        # skip lines that do not look like definitions
        if not (p_line.startswith("-") or p_line.startswith("(")):
            if len(p_line) > 0:
                print(f'[WARNING] unable to parse the line: "{p_line}"')
            continue

        # extract signature and description
        if ":" in p_line:
            action, desc = p_line.split(":", 1)
            desc = desc.strip().strip("'\"")
        elif ";" in p_line:
            action, desc = p_line.split(";", 1)
            desc = desc.strip().strip("'\"")
        else:
            action = p_line
            desc = ""

        # clean the signature
        action = action.strip("- ()").strip()

        new_actions[action] = desc

    return new_actions
