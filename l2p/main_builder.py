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
    
    def __init__(self, domain_name, problem_name, requirements: list[str] = [], isHTN: bool = False):
        """
        Initializes the MainBuilder class.

        Args:
            domain_name (str): Name of the domain.
            problem_name (str): Name of the problem.
            requirements (list[str]): List of requirements for the domain.
            isHTN (bool): Flag to indicate if the model is HTN or not.
        """
        super().__init__()
        self.isHTN = isHTN
        self.domain_name = domain_name
        self.problem_name = problem_name
        self.requirements = requirements
    
    @require_llm
    def extract_domain_and_problem(
        self,
        model: LLM,
        task_desc: str,
        prompt_template: str,
        max_retries: int = 0,
    ) -> tuple[dict[str, str], list[dict[str, str]], list[dict[str, str]], str]:
        """
        Extracts the domain and problem from a given task descrption, via One-shot LLM

        Args:
            model (LLM): LLM
            task_desc (str): problem description
            prompt_template (str): prompt template class
            max_retries (int): max # of retries if failure occurs

        Returns:
            type_hierarchy (dict[str,str]): dictionary of type hierarchy
            predicates (list[Predicate]): a list of new predicates
            tasks ([Task]): list of constructed tasks class with their methods (only if isHTN=True)
            actions (Action): list of constructed action class
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
                raw_types_hierarchy = extract_section_by_name(llm_response, "TYPES").split("\n## OUTPUT")[1]
                types_hierarchy = convert_to_dict(llm_response=raw_types_hierarchy)
                # extract respective types predicates and tasks from response
                raw_predicates = extract_section_by_name(llm_response, "PREDICATES").split("\n## OUTPUT")[1]
                predicates = parse_list_of_predicates(raw_predicates)
  
                if self.isHTN:
                    # extract respective tasks and methods from responseç
                    raw_tasks = extract_section_by_name(llm_response, "TASKS").split("\n## OUTPUT")[1]
                    tasks = parse_tasks(raw_tasks)
                    
                    for task in tasks.items():
                        task_name = task[0]
                        methods = list()
                        raw_task_info = extract_section_by_name(llm_response, task_name)
                        raw_methods_list = raw_task_info.split("\n## ")[1:]
                        for j in raw_methods_list:
                            method_name, rest_of_string = j.split("\n", 1)
                            method = parse_method(llm_response=rest_of_string, method_name=method_name)
                            methods.append(method)
                        tasks[task_name]["methods"] = methods
                                             
                # extract respective actions from response
                raw_actions = extract_section_by_name(llm_response, "ACTIONS").split("\n## ")[1:]
                actions = []
                for i in raw_actions:
                    action_name, rest_of_string = i.split("\n", 1)
                    actions.append(
                        parse_action(llm_response=rest_of_string, action_name=action_name)
                    )
                    
       
                # extract respective Problem types from response
                raw_objects = extract_section_by_name(llm_response, "OBJECTS").split("\n## OUTPUT")[1]
                objects = parse_objects(raw_objects, md_mode=True)
                raw_initial = extract_section_by_name(llm_response, "INITIAL").split("\n## OUTPUT")[1]
                initial = parse_initial(raw_initial, md_mode=True)
                raw_goal = extract_section_by_name(llm_response, "GOAL").split("\n## OUTPUT")[1]
                goal = parse_goal(raw_goal, md_mode=True)
                
                self.types_hierarchy = types_hierarchy
                self.predicates = predicates
                self.actions = actions
                self.objects = objects
                self.initial = initial
                self.goal = goal
                if self.isHTN:
                    self.tasks = tasks

                if self.isHTN:
                    return types_hierarchy, predicates, tasks, actions, objects, initial, goal, llm_response
                else:
                    return types_hierarchy, predicates, actions, objects, initial, goal, llm_response

            except Exception as e:
                print(
                    f"Error encountered: {e}. Retrying {attempt + 1}/{max_retries}..."
                )
                print(traceback.format_exc())
                time.sleep(2)  # add a delay before retrying

        raise RuntimeError("Max retries exceeded. Failed to extract task.")
    
    def method_desc(self, method: HPDLMethod) -> str:
        """Helper function to format method descriptions"""
        # param_str = "\n".join(
        #     [f"{name} - {type}" for name, type in method["params"].items()]
        # )  # name includes ?
        desc = f"(:method {method['name']}\n"
        # desc += f"   :parameters (\n{indent(string=param_str, level=2)}\n   )\n"
        # desc += f"   :task\n{indent(string=method['task'], level=2)}\n"
        desc += f"   :tasks\n{indent(string=method['tasks'], level=2)}\n"
        desc += ")"
        return desc
    
    def methods_desc(self, methods) -> str:
        """Helper function to combine all methods descriptions"""
        desc = ""
        for method in methods:
            desc += "\n\n" + indent(self.method_desc(method), level=1)
        return desc
    
    
    def task_desc(self, task: HPDLTask) -> str:
        """Helper function to format task descriptions"""
        param_str = "\n".join(
            [f"{name} - {type}" for name, type in task["params"].items()]
        )  # name includes ?
        desc = f"(:task {task['name']}\n"
        desc += f"   :parameters (\n{indent(string=param_str, level=2)}\n   )\n"
        desc += f"   {indent(string=self.methods_desc(task['methods']), level=0)}\n"
        desc += ")"
        return desc
            
    def tasks_descs(self, tasks) -> str:
        """Helper function to combine all task descriptions"""
        desc = ""
        for task in tasks.values():
            desc += "\n\n" + indent(self.task_desc(task), level=1)
        return desc
    
    def get_domain(self, language='PRED') -> str:
        """
        Generates PDDL/HPDL/HDDL domain from given information

        Args:
            domain (str): domain name
            self.types (str): domain types
            self.predicates (str): domain predicates
            self.actions (list[Action]): domain actions
            self.requirements (list[str]): domain requirements

        Returns:
            desc (str): PDDL/HPDL/HDDL domain
        """
        
        if language == 'PRED':
            if self.isHTN:
                language = 'HDDL'
            else:
                language = 'PDDL'
        
        #Extract types string
        types = format_types(self.types_hierarchy)  # retrieve types
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
            # desc += self.method_descs(self.methods)
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
