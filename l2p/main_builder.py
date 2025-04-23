"""
This file contains collection of functions for HDDL domain and problem generation purposes
"""

import re, time
from collections import OrderedDict
from .utils import *
from .llm_builder import LLM, require_llm
from .domain_builder import DomainBuilder
from .task_builder import TaskBuilder

class MainBuilder(DomainBuilder, TaskBuilder):
    """
    Class to build a planning model, including the domain and problem specifications and the HTN capabilities.
    """
    isHTN: bool
    
    def __init__(self, isHTN: bool = True):
        """
        Initializes the MainBuilder class.

        Args:
            isHTN (bool): Flag to indicate if the model is HTN or not.
        """
        super().__init__()
        self.isHTN = isHTN
    
    @require_llm
    def extract_domain_and_problem(
        self,
        model: LLM,
        task_desc: str,
        prompt_template: str,
        max_retries: int = 1,
    ) -> tuple[dict[str, str], list[dict[str, str]], list[dict[str, str]], str]:
        """
        Extracts the domain and problem from a given task descrption, via One-shot LLM

        Args:
            model (LLM): LLM
            task_desc (str): problem description
            prompt_template (str): prompt template class
            max_retries (int): max # of retries if failure occurs

        Returns:
            type_dict (dict[str,str]): dictionary of types with (name:description) pair
            type_hierarchy (dict[str,str]): dictionary of type hierarchy
            tasks ([Task]): list of constructed tasks class (only if isHTN=True)
            methods ([Method]): list of constructed methods class (only if isHTN=True)
            actions (Action): list of constructed action class
            predicates (list[Predicate]): a list of new predicates
            objects (dict[str,str]): dictionary of object types {name:description}
            initial (list[dict[str,str]]): list of dictionary of initial states [{predicate,params,neg}]
            goal (list[dict[str,str]]): list of dictionary of goal states [{predicate,params,neg}]
            llm_response (str): the raw string LLM response
        """

        model.reset_tokens()

        prompt_template = prompt_template.replace("{task_desc}", task_desc)

        # iterate through attempts in case of extraction failure
        for attempt in range(max_retries):
            try:
                model.reset_tokens()

                llm_response = model.query(prompt=prompt_template)

                # extract respective types from response
                types_fragment = llm_response.split("## TYPES")[1].split("##")[0]
                types = convert_to_dict(llm_response=types_fragment)
                
                types_hierarchy_fragment = llm_response.split("## TYPES HIERARCHY")[1].split("##")[0]
                type_hierarchy = convert_to_dict(llm_response=types_hierarchy_fragment)
                
                # extract respective types predicates and tasks from response
                predicates = parse_new_predicates(llm_response)
                if self.isHTN: 
                    tasks = parse_tasks(llm_response)
                        
                # extract respective actions from response
                raw_actions = llm_response.split("## NEXT ACTION")
                actions = []
                for i in raw_actions:
                    # define the regex patterns
                    action_pattern = re.compile(r"\[([^\]]+)\]")
                    rest_of_string_pattern = re.compile(r"\[([^\]]+)\](.*)", re.DOTALL)

                    # search for the action name
                    action_match = action_pattern.search(i)
                    action_name = action_match.group(1) if action_match else None

                    # extract the rest of the string
                    rest_match = rest_of_string_pattern.search(i)
                    rest_of_string = rest_match.group(2).strip() if rest_match else None

                    actions.append(
                        parse_action(llm_response=rest_of_string, action_name=action_name)
                    )
                    
                if self.isHTN:
                    # extract respective methods from response
                    raw_methods = llm_response.split("## NEXT METHOD")
                    methods = []
                    for i in raw_methods:
                        # define the regex patterns
                        pattern = re.compile(r"\[([^\]]+)\]")
                        rest_of_string_pattern = re.compile(r"\[([^\]]+)\](.*)", re.DOTALL)

                        # search for the method name
                        match = pattern.search(i)
                        name = match.group(1) if match else None

                        # extract the rest of the string
                        rest_match = rest_of_string_pattern.search(i)
                        rest_of_string = rest_match.group(2).strip() if rest_match else None

                        methods.append(
                            parse_method(llm_response=rest_of_string, method_name=name)
                        )
                            
                # extract respective Problem types from response
                objects = parse_objects(llm_response)
                initial = parse_initial(llm_response)
                goal = parse_goal(llm_response)

                if self.isHTN:
                    return types, type_hierarchy, tasks, methods, actions, predicates, objects, initial, goal, llm_response
                else:
                    return types, type_hierarchy, actions, predicates, objects, initial, goal, llm_response

            except Exception as e:
                print(
                    f"Error encountered: {e}. Retrying {attempt + 1}/{max_retries}..."
                )
                time.sleep(2)  # add a delay before retrying

        raise RuntimeError("Max retries exceeded. Failed to extract task.")

    def task_desc(self, task: Task) -> str:
        """Helper function to format task descriptions"""
        param_str = "\n".join(
            [f"{name} - {type}" for name, type in task["params"].items()]
        )  # name includes ?
        desc = f"(:task {task['name']}\n"
        desc += f"   :parameters (\n{indent(string=param_str, level=2)}\n   )\n"
        desc += ")"
            
    def tasks_descs(self, tasks) -> str:
        """Helper function to combine all task descriptions"""
        desc = ""
        for task in tasks:
            desc += "\n\n" + indent(self.method_desc(task), level=1)
        return desc
    
    def method_desc(self, method: Method) -> str:
        """Helper function to format method descriptions"""
        param_str = "\n".join(
            [f"{name} - {type}" for name, type in method["params"].items()]
        )  # name includes ?
        desc = f"(:method {method['name']}\n"
        desc += f"   :parameters (\n{indent(string=param_str, level=2)}\n   )\n"
        desc += f"   :task\n{indent(string=method['task'], level=2)}\n"
        desc += f"   :ordered-subtasks\n{indent(string=method['ordered_subtasks'], level=2)}\n"
        desc += ")"
        return desc
    
    def method_descs(self, methods) -> str:
        """Helper function to combine all methods descriptions"""
        desc = ""
        for method in methods:
            desc += "\n\n" + indent(self.method_desc(method), level=1)
        return desc
    
    def generate_hddl_domain(
        self,
        domain: str,
        types: str,
        predicates: str,
        tasks: list[Task],
        methods: list[Method],
        actions: list[Action],
        requirements: list[str],
    ) -> str:
        """
        Generates PDDL domain from given information

        Args:
            domain (str): domain name
            types (str): domain types
            predicates (str): domain predicates
            actions (list[Action]): domain actions
            requirements (list[str]): domain requirements

        Returns:
            desc (str): PDDL domain
        """
        desc = ""
        desc += f"(define (domain {domain})\n"
        desc += (
            indent(string=f"(:requirements\n   {' '.join(requirements)})", level=1)
            + "\n\n"
        )
        desc += f"   (:types \n{indent(string=types, level=2)}\n   )\n\n"
        desc += f"   (:predicates \n{indent(string=predicates, level=2)}\n   )"
        if self.isHTN:
            desc += self.tasks_descs(tasks)
            desc += self.method_descs(methods)
        desc += self.action_descs(actions)
        desc += "\n)"
        desc = desc.replace("AND", "and").replace("OR", "or")
        return desc
