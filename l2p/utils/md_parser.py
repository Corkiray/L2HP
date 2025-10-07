"""
This file contains collection of functions for extracting/parsing markdown datastructures from LLM output
"""
import re, ast
from .pddl_parser import combine_blocks, parse_pddl, remove_comments, parse_task_states

def extract_bracket_block(text: str, block_name: str) -> str:
    """
    Extracts a block of text headed by a specific name in square brackets.
    Args:
        text (str): The raw text containing the block.
        block_name (str): The name of the block to extract (e.g., "[ROLE]", "[TEMPLATE]").
    Returns:
        str: The content of the block.
    """
    # Regex to match [BLOCK_NAME] ... [NEXT_BLOCK] or end of string
    pattern = re.compile(
        rf"\[{re.escape(block_name)}\]\s*(.*?)(?=\n\[\w+\]|\Z)", re.DOTALL | re.IGNORECASE
    )
    match = pattern.search(text)
    if match:
        return match.group(1).strip()
    return ""

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
    
def substract_logical_expression(text: str) -> str:
    """
    Substracts logical expression from LLM response and returns it as a string

    Args:
        llm_response (str): The LLM output.

    Returns:
        states (list[dict[str,str]]): list of initial states in dictionaries
    """
    # Find all substrings enclosed by parentheses
    matches = re.findall(r'\((.*)\)', text, re.DOTALL)

    # Return the largest match or an empty string if no matches
    if matches:
        return f"({max(matches, key=len)})"
    else:
        raise ValueError("Could not find the logical expression in the LLM output. Provide the entire response, including all headings even if some are unchanged.")


def convert_to_dict(llm_response: str) -> dict[str, str]:

    """Converts string into Python dictionary format."""

    dict_pattern = re.compile(

        r"{.*}", re.DOTALL

    )  # regular expression to find the JSON-like dictionary structure

    match = dict_pattern.search(

        llm_response

    )  # search for the pattern in the llm_response



    # safely evaluate the string to convert it into a Python dictionary

    if match:

        dict_str = match.group(0)

        try:

            dict = ast.literal_eval(dict_str)

            return dict

        except Exception as e:

            print(f"Error parsing dictionary: {e}")

            return None

    else:

        print("No dictionary found in the llm_response.")

        return None
    
    
    
def parse_objects_md(llm_output: str) -> dict[str, str]:
    """
    Extract objects from markdown text and returns dictionary string pairs object(name, type)

    Args:
        llm_output (str): raw LLM output
        types (dict[str,str]): WILL BE USED FOR CHECK ERROR RAISES
        predicates (list[Predicate]): WILL BE USED FOR CHECK ERROR RAISES

    Returns:
        objects_parsed (dict[str,str]): PDDL task objects
    """

    objects_section = extract_section_by_name(llm_output, "OBJECTS")
    objects_output = extract_section_by_name(objects_section, "OUTPUT", level=2)
    objects_combined = combine_blocks(objects_output)
    objects_clean = remove_comments(objects_combined)

    objects_parsed = {
        obj.split(" - ")[0].strip(" `"): obj.split(" - ")[1].strip(" `")
        for obj in objects_clean.split("\n")
        if obj.strip()
    }

    return objects_parsed


def parse_initial_md(llm_output: str) -> list[dict[str, str]]:
    """
    Extracts state (PDDL-init) from markdown text and returns it as a list of dict strings

    Parameters:
        llm_output (str): raw LLM output

    Returns:
        states (list[dict[str,str]]): list of initial states in dictionaries
    """
    initial_section = extract_section_by_name(llm_output, "INITIAL")
    initial_output = extract_section_by_name(initial_section, "OUTPUT", level=2)
    initial_combined = combine_blocks(initial_output)
    initial_clean = remove_comments(initial_combined)
    initial_parsed = parse_pddl(f"({initial_clean})")
    initial_formatted = parse_task_states(initial_parsed)
    
    return initial_formatted


def clean_markdown_comments(text: str) -> str:
    """
    Removes HTML-style comments (<!-- ... -->) from markdown text.
    """
    return re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)