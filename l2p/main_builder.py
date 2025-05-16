"""
This file contains collection of functions for HDDL domain and problem generation purposes
"""

import re, time
import traceback
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
    
    def __init__(self, domain_name, problem_name, isHTN: bool = False):
        """
        Initializes the MainBuilder class.

        Args:
            isHTN (bool): Flag to indicate if the model is HTN or not.
        """
        super().__init__()
        self.isHTN = isHTN
        self.domain_name = domain_name
        self.problem_name = problem_name
    
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

        prompt = prompt_template.replace("{task_desc}", task_desc)

        # iterate through attempts in case of extraction failure
        for attempt in range(max_retries):
            try:
                model.reset_tokens()

                llm_response = model.query(prompt)
                
                # print(llm_response)

                # extract respective types from response
                raw_types = extract_section_by_name(llm_response, "TYPES").split("\n## OUTPUT")[1]
                types = convert_to_dict(llm_response=raw_types)
                
                raw_types_hierarchy = extract_section_by_name(llm_response, "TYPES HIERARCHY").split("\n## OUTPUT")[1]
                type_hierarchy = convert_to_dict(llm_response=raw_types_hierarchy)
                
                # extract respective types predicates and tasks from response
                predicates = parse_new_predicates(llm_response)
  
                if self.isHTN:
                    tasks = parse_tasks(llm_response)
                        
                # extract respective actions from response
                raw_actions = extract_section_by_name(llm_response, "ACTIONS").split("\n## NEXT ACTION")
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
                    raw_methods = extract_section_by_name(llm_response, "METHODS").split("## NEXT METHOD")
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

                        method = parse_method(llm_response=rest_of_string, method_name=name)
                        methods.append(method)
                            
                # extract respective Problem types from response
                objects = parse_objects(llm_response, md_mode=True)
                initial = parse_initial(llm_response, md_mode=True)
                goal = parse_goal(llm_response, md_mode=True)
                
                self.types = types
                self.type_hierarchy = type_hierarchy
                self.predicates = predicates
                self.actions = actions
                self.objects = objects
                self.initial = initial
                self.goal = goal
                if self.isHTN:
                    self.tasks = tasks
                    self.methods = methods
                

                if self.isHTN:
                    return types, type_hierarchy, tasks, methods, actions, predicates, objects, initial, goal, llm_response
                else:
                    return types, type_hierarchy, actions, predicates, objects, initial, goal, llm_response

            except Exception as e:
                print(
                    f"Error encountered: {e}. Retrying {attempt + 1}/{max_retries}..."
                )
                print(traceback.format_exc())
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
        return desc
            
    def tasks_descs(self, tasks) -> str:
        """Helper function to combine all task descriptions"""
        desc = ""
        for task in tasks:
            desc += "\n\n" + indent(self.task_desc(task), level=1)
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
    
    def get_domain(self) -> str:
        """
        Generates PDDL domain from given information

        Args:
            domain (str): domain name
            self.types (str): domain types
            self.predicates (str): domain predicates
            self.actions (list[Action]): domain actions
            self.requirements (list[str]): domain requirements

        Returns:
            desc (str): PDDL domain
        """
        
        #Extract types string
        types = format_types(self.type_hierarchy)  # retrieve types
        pruned_types = {
            name: description
            for name, description in types.items()
        }
        types_str = "\n".join(pruned_types)
        
        #Extract predicates string
        predicate_str = "\n".join(
            [pred["clean"].replace(":", " ; ") for pred in self.predicates]
        ) 
        
        desc = ""
        desc += f"(define (domain {self.domain_name})\n"
        desc += (
            indent(string=f"(:requirements\n   {' '.join(self.requirements)})", level=1)
            + "\n\n"
        )
        desc += f"   (:types \n{indent(string=types_str, level=2)}\n   )\n\n"
        desc += f"   (:predicates \n{indent(string=predicate_str, level=2)}\n   )"
        if self.isHTN:
            desc += self.tasks_descs(self.tasks)
            desc += self.method_descs(self.methods)
        desc += self.action_descs(self.actions)
        desc += "\n)"
        desc = desc.replace("AND", "and").replace("OR", "or")
        return desc


    def get_problem(self) -> str:
        """
        Generates PDDL problem from given information
        Args:
            self.domain_name (str): domain name
            self.problem_name (str): problem name
            self.objects (str): domain objects
            self.initial (str): domain initial state
            self.goal (str): domain goal state
        Returns:
            desc (str): PDDL problem
        """
        # construct PDDL components into PDDL problem file
        object_str = self.format_objects(self.objects)
        initial_state_str = self.format_initial(self.initial)
        goal_state_str = self.format_goal(self.goal)

        # generate proper PDDL structure
        desc = self.generate_task(
            self.domain_name,
            self.problem_name,
            object_str,
            initial_state_str,
            goal_state_str,
        )
        return desc
