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
    def formalize_domain_and_task(
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
    
    @require_llm
    def extract_nl_tasks(
        self,
        model: BaseLLM,
        domain_desc: str,
        prompt_template: str,
        types: dict[str, str] | list[dict[str, str]] | None = None,
        nl_tasks: dict[str, str] | None = None,
        max_retries: int = 3,
    ) -> tuple[dict[str, str], str]:
        """
        Extract tasks in natural language given domain description using BaseLLM.
        
        Args:
            model (BaseLLM): LLM to query
            domain_desc (str): general domain description
            prompt_template (str): structured prompt template for dictionary extraction
            types (dict[str,str] | list[dict[str,str]]): current types in specification, defaults to None
            nl_tasks (dict[str, str]): NL tasks currently in class object w/ {<name>: <description>} key-value pair
            max_retries (int): max # of retries if failure occurs

        Returns:
            nl_tasks (dict[str, str]): a dictionary of extracted NL tasks {<name>: <description>}
            llm_output (str): the raw string BaseLLM response
        """
        types_str = pretty_print_dict(types) if types else "No types provided."
        nl_task_str = (
            "\n".join(f" - {name}: {desc}" for name, desc in nl_tasks.items())
            if nl_tasks
            else "No tasks provided."
        )

        prompt = (
            prompt_template.replace("{domain_desc}", domain_desc)
            .replace("{types}", types_str)
            .replace("{nl_tasks}", nl_task_str)
        )

        # iterate through attempts in case of extraction failure
        for attempt in range(max_retries):
            try:
                model.reset_tokens()
                llm_output = model.query(prompt=prompt)

                # extract respective nl tasks from response
                nl_tasks = parse_types(llm_output=llm_output, heading="TASKS")

                if nl_tasks is not None:
                    return nl_tasks, llm_output

            except Exception as e:
                print(
                    f"Error encountered during attempt {attempt + 1}/{max_retries}: {e}. "
                    f"\nLLM Output: \n\n{llm_output if 'llm_output' in locals() else 'None'}\n\n Retrying..."
                )
                time.sleep(2)  # add a delay before retrying

        raise RuntimeError("Max retries exceeded. Failed to extract NL tasks.")
    
    @require_llm
    def formalize_HPDLTasks(
        self,
        model: BaseLLM,
        domain_desc: str,
        prompt_template: str,
        types: dict[str, str] | list[dict[str, str]] | None = None,
        tasks: list[HPDLTask] | None = None,
        max_retries: int = 3,
    ) -> tuple[dict[str, HPDLTask], str]:
        """
        Extract complete task definitions including parameters from domain description using BaseLLM.
        
        Args:
            model (BaseLLM): LLM to query
            domain_desc (str): general domain description
            prompt_template (str): structured prompt template for task extraction
            types (dict[str,str] | list[dict[str,str]]): current types in specification
            tasks (dict[str, dict]): current tasks in specification
            max_retries (int): max # of retries if failure occurs

        Returns:
            task_dict (dict[str, dict]): Dictionary containing task definitions with parameters
            llm_output (str): The raw string BaseLLM response
        """
        types_str = pretty_print_dict(types) if types else "No types provided."
        tasks_str = ""
        if tasks:
            for task in tasks:
                params_str = ", ".join(f"{param}: {type}" for param, type in task["params"].items())
                tasks_str += f" - {task['name']} ({params_str}): {task['desc']}\n"
        else:
            tasks_str = "No tasks provided."

        prompt = (
            prompt_template.replace("{domain_desc}", domain_desc)
            .replace("{types}", types_str)
            .replace("{tasks}", tasks_str)
        )

        for attempt in range(max_retries):
            try:
                model.reset_tokens()
                llm_output = model.query(prompt=prompt)

                # Extract task definitions from response
                tasks_section = extract_section_by_name(llm_output, "TASKS", level=2)
                if not tasks_section:
                    raise ValueError("No TASKS section found in LLM response")

                # Parse tasks and their parameters
                new_tasks = parse_tasks(tasks_section)

                return new_tasks, llm_output

            except Exception as e:
                print(
                    f"Error encountered during attempt {attempt + 1}/{max_retries}: {e}. "
                    f"\nLLM Output: \n\n{llm_output if 'llm_output' in locals() else 'None'}\n\n Retrying..."
                )
                time.sleep(2)
                
        # # Validate the extracted tasks
        # validation_info = self.validate_tasks(new_tasks, types)
        # if not validation_info[0]:
        #     raise ValueError(f"Task validation failed: {validation_info[1]}")

        raise RuntimeError("Max retries exceeded. Failed to extract tasks.")

    def validate_tasks(self, tasks: dict[str, HPDLTask], types: dict[str, str]) -> tuple[bool, str]:
        """
        Validate extracted tasks against type definitions and structural requirements.
        
        Args:
            tasks (dict[str, dict]): Dictionary of tasks to validate
            types (dict[str, str]): Type definitions to validate against

        Returns:
            tuple[bool, str]: (is_valid, error_message)
        """
        for task_name, task in tasks.items():
            # Validate task name
            if not task_name or not isinstance(task_name, str):
                return False, f"Invalid task name: {task_name}"

            for param_name, param_type in task["params"].items():
                if param_type not in types:
                    return False, f"Task {task_name} parameter {param_name} has undefined type {param_type}"

        return True, "Tasks validation successful"
    
    
    @require_llm
    def extract_methods_list(
        self,
        model: BaseLLM,
        domain_desc: str,
        types: dict[str, str] | list[dict[str, str]] | None = None,
        tasks: list[str] | None = None,
        task: HPDLTask | None = None,
        prompt_template: str = "",
        max_retries: int = 3,
    ) -> tuple[dict[str,str], str]:
        """
        Extract list of methods needed for a specific task using BaseLLM.
        
        Args:
            model (BaseLLM): LLM to query
            domain_desc (str): general domain description
            types (dict[str,str] | list[dict[str,str]]): current types in specification
            tasks (list[str]): list of all available tasks
            task (dict): the specific task for which to extract methods
            prompt_template (str): structured prompt template for method extraction
            max_retries (int): max # of retries if failure occurs
            
        Returns:
            tuple[list[str], str]: A tuple containing:
                - List of method names needed for the task
                - The raw string LLM response
        """
        types_str = pretty_print_dict(types) if types else "No types provided."
        tasks_str = "\n".join(tasks) if tasks else "No tasks provided."
        task_str = f"{task['name']} {task['params']}: {task['desc']}" if task else "No task provided."

        prompt = (
            prompt_template.replace("{domain_desc}", domain_desc)
            .replace("{types}", types_str)
            .replace("{tasks}", tasks_str)
            .replace("{task}", task_str)
        )

        for attempt in range(max_retries):
            try:
                model.reset_tokens()
                llm_output = model.query(prompt=prompt)
                
                # Parse methods list
                methods = parse_types(llm_output=llm_output, heading="METHODS")
                if not methods:
                    raise ValueError(f"No methods were extracted for task {task['name']}")

                return methods, llm_output

            except Exception as e:
                print(
                    f"Error encountered during attempt {attempt + 1}/{max_retries}: {e}. "
                    f"\nLLM Output: \n\n{llm_output if 'llm_output' in locals() else 'None'}\n\n Retrying..."
                )
                time.sleep(2)

        raise RuntimeError(f"Max retries exceeded. Failed to extract methods for task {task['name']}")
    
    @require_llm
    def formalize_method(
        self,
        model: BaseLLM,
        domain_desc: str,
        types: dict[str, str] | list[dict[str, str]] | None = None,
        method_name: str | None = None,
        method_desc: str | None = None,
        method_task: HPDLTask | None = None,
        tasks_list: list[str] | None = None,
        predicates_list: list[Predicate] | None = None,
        nl_actions: dict[str, str] | None = None,
        prompt_template: str = "",
        extract_new_preds: bool = True,
        extract_new_actions: bool = True,
        max_retries: int = 3,
    ) -> tuple[HPDLMethod, list[Predicate], dict[str, str], str]:
        """
        Formalize a single HTN method (build method structure suitable for HPDL/HDDL).

        Returns:
            method_formal (dict): Formalized method structure with keys:
                'name', 'params', 'task', 'ordered_subtasks', 'tasks' (where applicable)
            new_predicates (list[str]): Any newly discovered predicates
            new_nl_actions (dict[str,str]): Any new NL actions discovered while formalizing methods
            llm_output (str): Raw LLM output
        """
        types_str = pretty_print_dict(types) if types else "No types provided."
        tasks_str = "\n".join(tasks_list) if tasks_list else "No tasks provided."

        # Compose context representations
        task_str = "No task provided."
        if method_task:
            params_repr = " ".join(f"{p} - {t}" for p, t in method_task.get("params", {}).items())
            task_str = f"{method_task.get('name')} ({params_repr}): {method_task.get('desc','')}"
        elif method_name:
            task_str = f"{method_name}: {method_desc or ''}"
            
        preds_str = (
            "\n".join([f"{pred['raw']}" for pred in predicates_list])
            if predicates_list
            else "No predicates provided."
        )
        
        actions_str = (
            "\n".join([f"- {a}: {desc}" for a, desc in nl_actions.items()])
            if nl_actions
            else "No actions provided."
        )

        prompt = (
            prompt_template.replace("{domain_desc}", domain_desc)
            .replace("{types}", types_str)
            .replace("{tasks}", tasks_str)
            .replace("{task}", task_str)
            .replace("{predicates}", preds_str)
            .replace("{actions}", actions_str)
            .replace("{method_name}", method_name or "No method name provided.")
            .replace("{method_desc}", method_desc or "No method description provided.")
        )

        for attempt in range(max_retries):
            try:
                model.reset_tokens()
                llm_output = model.query(prompt=prompt)

                # Extract method details from response
                parameters, _ = parse_params(llm_output)
                preconditions = parse_preconditions(llm_output)
                subtasks = parse_ordered_subtasks(llm_output)

                # Construct formalized method
                method_formal = {
                    "name": method_name,
                    "params": parameters,
                    "preconditions": preconditions,
                    "task": f'{method_task["clean"]}',
                    "ordered_subtasks": subtasks,
                    "desc": method_desc,
                }

                 # --- Extract new predicates and actions if applicable ---
                new_predicates = []
                new_nl_actions = {}
                if extract_new_preds:
                    new_predicates = parse_new_predicates(llm_output=llm_output)
                if extract_new_actions:
                    new_nl_actions = parse_new_nl_actions(llm_output=llm_output)

                return method_formal, new_predicates, new_nl_actions, llm_output

            except Exception as e:
                print(
                    f"Error encountered during attempt {attempt + 1}/{max_retries}: {e}. "
                    f"\nLLM Output: \n\n{llm_output if 'llm_output' in locals() else 'None'}\n\n Retrying..."
                )
                time.sleep(2)

        # # Validate the extracted tasks
        # validation_info = self.validate_method(method_formal, types)
        # if not validation_info[0]:
        #     raise ValueError(f"Task validation failed: {validation_info[1]}")

        raise RuntimeError(f"Max retries exceeded. Failed to formalize method {method_name}")
    
    def validate_method(self, method: dict, types: dict[str, str]) -> tuple[bool, str]:
        """
        Validate extracted method against type definitions and structural requirements.
        
        Args:
            method (dict): Method to validate
            types (dict[str, str]): Type definitions to validate against

        Returns:
            tuple[bool, str]: (is_valid, error_message)
        """
        # Validate method name
        if not method.get("name") or not isinstance(method["name"], str):
            return False, f"Invalid method name: {method.get('name')}"

        for param_name, param_type in method.get("params", {}).items():
            if param_type not in types:
                return False, f"Method {method['name']} parameter {param_name} has undefined type {param_type}"

        return True, "Method validation successful"