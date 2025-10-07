"""
This file contains collection of functions for HDDL domain and problem generation purposes
"""

import re, time
import traceback
from collections import OrderedDict
from .utils import *
from .llm import BaseLLM, require_llm
from .domain_builder import DomainBuilder
from .task_builder import TaskBuilder

class ModelBuilder(DomainBuilder, TaskBuilder):
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
    
    # @require_llm
    def extract_domain_and_problem(
        self,
        model: BaseLLM,
        task_desc: str,
        prompt_template: str,
        max_retries: int = 1,
    ) -> str:
        """
        Extracts the domain and problem from a given task descrption, via One-shot LLM, and stores the information in the MainBuilder instance atributes.
        
        This function uses a prompt template to format the task description and queries the LLM for the domain and problem information.
        
        It handles retries in case of extraction failure, allowing for a specified number of attempts to retrieve the information.
        
        This function is designed to be used with an LLM that can process the task description and return structured information about the domain and problem.

        Args:
            model (LLM): LLM
            task_desc (str): problem description
            prompt_template (str): prompt template class
            max_retries (int): max # of retries if failure occurs
        Returns:
            llm_response (str): the raw string LLM response
        """
        model.reset_tokens()

        prompt = prompt_template.replace("{task_desc}", task_desc)

        # iterate through attempts in case of extraction failure
        for attempt in range(max_retries):
            try:
                model.reset_tokens()
                self.llm_response = model.query(prompt)
                
                # extract respective types from response
                raw_types = extract_section_by_name(self.llm_response, "TYPES")
                raw_types = extract_section_by_name(raw_types, "OUTPUT", level=2)
                self.types = convert_to_dict(llm_response=raw_types)

                # extract respective types predicates and tasks from response
                raw_predicates = extract_section_by_name(self.llm_response, "PREDICATES")
                raw_predicates = extract_section_by_name(raw_predicates, "OUTPUT", level=2)
                self.predicates = parse_list_of_predicates(raw_predicates)
  
                if self.isHTN:
                    # extract respective tasks and methods from responseç
                    raw_tasks = extract_section_by_name(self.llm_response, "TASKS")
                    raw_tasks = extract_section_by_name(raw_tasks, "OUTPUT", level=2)
                    self.tasks = parse_tasks(raw_tasks)
                    for task in self.tasks.items():
                        task_name = task[0]
                        raw_task_info = extract_section_by_name(self.llm_response, task_name)
                        methods = parse_methods(raw_task_info)
                        self.tasks[task_name]["methods"] = methods
                                             
                # extract respective actions from response
                raw_actions = extract_section_by_name(self.llm_response, "ACTIONS")
                raw_actions_list = split_sections(raw_actions, level=2)
                self.actions = parse_actions_list(raw_actions_list)
       
       
                # --- Extract respective Problem types from response ---
                
                # raw_objects = extract_section_by_name(self.llm_response, "OBJECTS")
                # raw_objects = extract_section_by_name(raw_objects, "OUTPUT", level=2)
                self.objects = parse_objects_md(self.llm_response)
                
                # raw_initial = extract_section_by_name(self.llm_response, "INITIAL")
                # raw_initial = extract_section_by_name(raw_initial, "OUTPUT", level=2)
                self.initial = parse_initial_md(self.llm_response)
                
                # raw_goal = extract_section_by_name(self.llm_response, "GOAL")
                # raw_goal = extract_section_by_name(raw_goal, "OUTPUT", level=2)
                if self.isHTN:
                    self.goal = parse_goal_htn(self.llm_response)
                else:
                    self.goal = parse_goal(self.llm_response)
                
                return self.llm_response

            except Exception as e:
                print(
                    f"Error encountered: {e}. Retrying {attempt + 1}/{max_retries}..."
                )
                print(traceback.format_exc())
                time.sleep(2)  # add a delay before retrying

        raise RuntimeError("Max retries exceeded. Failed to extract task.")

    
    def HPDLmethod_desc(self, method: HPDLMethod) -> str:
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
    
    def HPDLmethods_desc(self, methods) -> str:
        """Helper function to combine all methods descriptions"""
        desc = ""
        for method in methods:
            desc += "\n\n" + indent(self.HPDLmethod_desc(method), level=1)
        return desc
    
    def HDDLmethods_desc(self, methods) -> str:
        """Helper function to combine all methods descriptions"""
        desc = ""
        for method in methods:
            desc += "\n\n" + indent(self.HDDLmethod_desc(method), level=1)
        return desc
    
    def HDDLmethod_desc(self, method) -> str:
        """Helper function to format method descriptions"""
        param_str = "\n".join(
            [f"{name} - {type}" for name, type in method["params"].items()]
        )  # name includes ?
        desc = f"(:method {method['name']}\n"
        desc += f"   :parameters (\n{indent(string=param_str, level=2)}\n   )\n"
        desc += f"   :task\n{indent(string=method['task'], level=2)}\n"
        desc += f"   :ordered-tasks\n{indent(string=method['ordered_subtasks'], level=2)}\n"
        desc += ")"
        return desc
    
    
    def HPDLtask_desc(self, task) -> str:
        """Helper function to format task descriptions"""
        param_str = "\n".join(
            [f"{name} - {type}" for name, type in task["params"].items()]
        )  # name includes ?
        desc = f"(:task {task['name']}\n"
        desc += f"   :parameters (\n{indent(string=param_str, level=2)}\n   )\n"
        desc += f"   {indent(string=self.HPDLmethods_desc(task['methods']), level=0)}\n"
        desc += ")"
        return desc
            
    def HPDLtasks_descs(self, tasks: HPDLTask) -> str:
        """Helper function to combine all task descriptions"""
        desc = ""
        for task in tasks.values():
            desc += "\n\n" + indent(self.HPDLtask_desc(task), level=1)
        return desc
    
    def HDDLtask_desc(self, task) -> str:
        """Helper function to format task descriptions"""
        param_str = "\n".join(
            [f"{name} - {type}" for name, type in task["params"].items()]
        )  # name includes ?
        desc = f"(:task {task['name']}\n"
        desc += f"   :parameters (\n{indent(string=param_str, level=2)}\n   )\n"
        desc += ")"
        return desc
    
    def HDDLtasks_descs(self, tasks: HDDLTask) -> str:
        """Helper function to combine all task descriptions"""
        desc = ""
        for task in tasks.values():
            desc += "\n\n" + indent(self.HDDLtask_desc(task), level=1)
        return desc
    
    
    def get_domain(self, language='PRED') -> str:
        """
        Generates PDDL/HPDL/HDDL domain from given information

        Args:
            language (str): language to use for the domain, can be 'PDDL', 'HPDL', 'HDDL', or 'PRED'

        Returns:
            desc (str): PDDL/HPDL/HDDL domain
        """
        
        if language == 'PRED':
            if self.isHTN:
                language = 'HDDL'
            else:
                language = 'PDDL'
        
       # generates requirements if not set
        if not self.requirements:
            requirements = self.generate_requirements(
                types=self.types, functions=self.functions, actions=self.actions
            )

        desc = ""
        desc += f"(define (domain {self.domain_name})\n"
        desc += indent(string=f"(:requirements\n   {' '.join(self.requirements)})", level=1)
        if self.types:
            types_str = format_types_to_string(self.types)
            desc += f"\n\n   (:types \n{indent(string=types_str, level=2)}\n   )"

        if self.constants:
            const_str = format_constants(self.constants)
            desc += f"\n\n   (:constants \n{indent(string=const_str, level=2)}\n   )"

        if not self.predicates:
            print(
                "[WARNING]: Domain has no predicates. This may cause planners to reject the domain or behave unexpectedly."
            )
        else:
            pred_str = format_expression(self.predicates)
            desc += f"\n\n   (:predicates \n{indent(string=pred_str, level=2)}\n   )"

        if self.functions:
            func_str = format_expression(self.functions)
            desc += f"\n\n   (:functions \n{indent(string=func_str, level=2)}\n   )"

        if language == 'HPDL':
            desc += self.HPDLtasks_descs(self.tasks)
        elif language == 'HDDL':
            desc += self.HDDLtasks_descs(self.tasks)
            for task in self.tasks.values():
                desc += self.HDDLmethods_desc(task["methods"])

        if not self.actions:
            print(
                "[WARNING]: Domain has no actions. The planner will not be able to generate any plan unless the goal is already satisfied."
            )
        else:
            desc += format_actions(self.actions)
        desc += "\n)"
        desc = desc.replace("AND", "and").replace("OR", "or")
        return desc


    def get_problem(self, language: str | None = None) -> str:
        """
        Generates PDDL problem from given information
        Args:
            language (str): language to use for the problem, can be 'PDDL', 'HPDL', or 'HDDL'
            self.domain_name (str): domain name
            self.problem_name (str): problem name
            self.objects (str): domain objects
            self.initial (str): domain initial state
            self.goal (str): domain goal state
        Returns:
            desc (str): PDDL problem
        """
        
        if language is None:
            if self.isHTN:
                language = 'HDDL'
            else:
                language = 'PDDL'
                
        desc = "(define\n"
        desc += f"   (problem {self.problem_name})\n"
        desc += f"   (:domain {self.domain_name})\n\n"
        desc += f"   (:objects \n{indent(format_objects(self.objects))}\n   )\n\n"
        
        if language == 'PDDL':
            desc += f"   (:init\n{indent(format_initial(self.initial))}\n   )\n\n"
            desc += f"   (:goal\n{indent(format_goal(self.goal))}\n   )\n\n"
        
        if language == 'HPDL':
            desc += f"   (:tasks-goal\n{indent(':tasks')}\n{indent(format_goal(self.goal), 3)}\n   )\n\n"
            desc += f"   (:init\n{indent(format_initial(self.initial))}\n   )\n\n"
        
        if language == 'HDDL':
            desc += f"  (:htn\n{indent(':parameters ()')}\n{indent('    :ordered-subtasks', 1)}\n{indent(format_goal(self.goal), 3)}\n   )\n\n"
            desc += f"   (:init\n{indent(format_initial(self.initial))}\n   )\n\n"
        
        desc += ")"
        desc = desc.replace("AND", "and").replace("OR", "or")
        return desc