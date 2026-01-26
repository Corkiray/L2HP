import traceback
from typing import Tuple, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator
from google import genai
from google.api_core import retry
import instructor
from typing import Annotated, Literal
import outlines

class PDDL_Type(BaseModel):
    """A PDDL type representation.
    Clasically, a type have a parent type, represented as 'type - parent'.
    If the type has no parent, the parent is 'entity'.
    Note that the type 'entity' is the root of all types, so it should not be declared."""
    desc: str = Field(description="A brief description of the type")
    type: str
    parent: Optional[str] = Field(default="entity", description="The parent type, default is 'entity'")
    
    def to_pddl(self) -> str:
        """Convert to PDDL type syntax: 'type - parent'"""
        if self.parent:
            return f"{self.type} - {self.parent}"
        return self.type

    def to_dict(self) -> dict:
        """Return type as dictionary"""
        return {
            "type": self.type,
            "parent": self.parent,
            "description": self.desc
        }
        
class List_of_Types(BaseModel):
    """A representation of the list of types in a PDDL Domain.
    Note that the type 'entity' is the root of all types, so it should not be declared.
    Also, ensure that each type has a unique name, including the 'entity' type and the parent types.
    """
    # thinking: str = Field(description="A brief space to think about the types")
    types: List[PDDL_Type] = Field(description="A list of types in the domain")
    
    def to_pddl(self) -> str:
        """Convert to PDDL types syntax"""
        types_str = "\n    ".join(t.to_pddl() for t in self.types)
        return f"(:types\n    {types_str}\n  )"

    def to_dict(self) -> dict:
        """Return types list as dictionary"""
        return {
            "types": [t.to_dict() for t in self.types],
            # "thinking": self.thinking
        }        
        
        
PDDLAvalibleTypes = ['entity']
    
def set_avalible_types(list_of_types: List_of_Types) -> list[str]:
    """Crear un Literal con todos los tipos"""

    global PDDLAvalibleTypes
    
    type_names = [t.type for t in list_of_types.types]
    
    PDDLAvalibleTypes = type_names + ['entity']
            
    return PDDLAvalibleTypes


class PDDL_Parameter(BaseModel):
    """A PDDL parameter representation (?name - type)."""
    name: str = Field(description="The name of the parameter. All the parameters always have to start with ?, e.g., ?x")
    type: str = Field(description="The type of the parameter")
    
    def to_pddl(self) -> str:
        """Convert to PDDL parameter syntax: '?name - type'"""
        return f"{self.name} - {self.type}"

    def to_dict(self) -> dict:
        """Return parameter as dictionary"""
        return {
            "name": self.name,
            "type": self.type
        }
        
    @field_validator('type')
    @classmethod
    def validate_type(cls, v):
        if PDDLAvalibleTypes and v not in PDDLAvalibleTypes:
            raise ValueError(f"Type '{v}' not in available types: {PDDLAvalibleTypes}")
        return v
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, name):
        if not name.startswith('?'):
            raise ValueError(f"Parameter name '{name}' must start with '?'")
        return name


class PDDL_Object(BaseModel):
    """A PDDL object representation"""
    name: str = Field(description="The name of the object, e.g., A")
    type: str = Field(description="The type of the object")
    
    def to_pddl(self) -> str:
        """Convert to PDDL object syntax: 'name - type'"""
        return f"{self.name} - {self.type}"

    def to_dict(self) -> dict:
        """Return object as dictionary"""
        return {
            "name": self.name,
            "type": self.type
        }
        
    @field_validator('type')
    @classmethod
    def validate_type(cls, v):
        if PDDLAvalibleTypes and v not in PDDLAvalibleTypes:
            raise ValueError(f"Type '{v}' not in available types: {PDDLAvalibleTypes}")
        return v
    
 
class List_of_Objects(BaseModel):
    """A representation of the list of objects in a PDDL Problem.
    Note that objects and types share names, so each name should be unique across both."""
    # thinking: str = Field(description="A brief space to think about the objects")
    objects: List[PDDL_Object] = Field(description="A list of objects in the problem")
    
    def to_pddl(self) -> str:
        """Convert to PDDL objects syntax"""
        objects_str = "\n    ".join(obj.to_pddl() for obj in self.objects)
        return f"(:objects\n    {objects_str}\n  )"
    
    def to_dict(self) -> dict:
        """Return objects list as dictionary"""
        return {
            "objects": [obj.to_dict() for obj in self.objects],
            # "thinking": self.thinking
        }
        
        
class PDDL_Predicate(BaseModel):
    """A PDDL predicate representation."""
    name: str = Field(description="The name of the predicate, e.g., on")
    parameters: List[PDDL_Parameter] = Field(description="A list of arguments for the predicate")
    
    def to_pddl(self) -> str:
        args_str = " ".join(param.to_pddl() for param in self.parameters)
        if args_str:
            return f"({self.name} {args_str})"
        return f"({self.name})"
    
    def to_dict(self) -> dict:
        """Return predicate as dictionary"""
        return {
            "name": self.name,
            "parameters": [arg.to_dict() for arg in self.parameters],
        }
        
        
class List_of_Predicates(BaseModel):
    """A representation of the list of predicates in a PDDL Domain."""
    # thinking: str = Field(description="A brief space to think about the predicates that are going to be needed within the domain")
    predicates: List[PDDL_Predicate] = Field(description="A list of predicates in the domain")
    
    def to_pddl(self) -> str:
        """Convert to PDDL predicates syntax"""
        predicates_str = "\n    ".join(pred.to_pddl() for pred in self.predicates)
        return f"(:predicates\n    {predicates_str}\n  )"
    
    def to_dict(self) -> dict:
        """Return predicates list as dictionary"""
        return {
            "predicates": [pred.to_dict() for pred in self.predicates],
            # "thinking": self.thinking
        }
        
        
PDDLAvaliblePredicates = list[PDDL_Predicate]()
    
def set_avalible_predicates(list_of_predicates: List_of_Predicates) -> list[PDDL_Predicate]:
    """Crear un Literal con todos los predicados"""
    global PDDLAvaliblePredicates   
    PDDLAvaliblePredicates = list_of_predicates.predicates
    return PDDLAvaliblePredicates


class PDDL_Fact(BaseModel):
    """A PDDL fact representation, i.e. a predicate or a negated one.
    Note that all the of them has to be instances of the provided predicates defined in the domain.
    """
    desc: str = Field(description="A brief description of this fact, e.g., 'Block A is not on Block B'")
    is_negative: Optional[bool] = Field(description="When a fact is part of a precondition or effect, it could be a negated predicate, i.e., (not (on ?a ?b))", default=False)
    name: str = Field(description="The name of the predicate, e.g., on.")
    args: List[str] = Field(description="A list of arguments for the fact, which defines an instance of the predicate. These arguments must match the predicate's parameters types.")

    def to_pddl(self) -> str:
        """Convert to PDDL literal syntax: '(name ?arg1 ?arg2)' or '(not (name ?arg1 ?arg2))'"""
        params_str = " ".join(self.args)
        if params_str:
            literal_str = f"({self.name} {params_str})"
        else:
            literal_str = f"({self.name})"
        
        if self.is_negative:
            return f"(not {literal_str})"
        return literal_str
    
    def to_dict(self) -> dict:
        """Return literal as dictionary"""
        return {
            "is_negative": self.is_negative,
            "name": self.name,
            "args": self.args
        }
        
    @field_validator('name')
    @classmethod
    def validate_name(cls, name):
        if PDDLAvaliblePredicates and name not in [p.name for p in PDDLAvaliblePredicates]:
            raise ValueError(f"Literal name '{name}' not in available predicates: {[p.name for p in PDDLAvaliblePredicates]}")
        return name

    @model_validator(mode='after')
    def validate_parameters(self):
        name = self.name
        args = self.args
        if PDDLAvaliblePredicates:
            predicate = next((p for p in PDDLAvaliblePredicates if p.name == name), None)
            if predicate and len(args) != len(predicate.parameters):
                raise ValueError(f"Literal '{name}' expects {len(predicate.parameters)} arguments, got {len(args)}")
            # if predicate:
            #     for i, param in enumerate(args):
            #         expected_type = predicate.parameters[i].type
            #         if param.type != expected_type:
            #             raise ValueError(f"Parameter {i+1} of literal '{name}' expects type '{expected_type}', got '{param.type}'")
        return self
    
    
class PDDL_Action(BaseModel):
    """A PDDL action representation."""
    # thinking: str = Field(description="A space to think about the action")
    name: str = Field(description="The name of the action")
    parameters: List[PDDL_Parameter] = Field(description="A list of parameters for the action. Each parameter must be an instance of a PDDLParameter, which includes its name and type.")
    preconditions: List[PDDL_Fact] = Field(description="A list of preconditions for the action. Each precondition must be an instance of a PDDLLiteral, which includes its name, parameters, and whether it is negated.")
    effects: List[PDDL_Fact] = Field(description="A list of effects for the action. Each effect must be an instance of a PDDLLiteral, which includes its name, parameters, and whether it is negated.")
    
    def to_pddl(self) -> str:
        """Convert to PDDL action syntax"""
        params_str = " ".join(param.to_pddl() for param in self.parameters)
        
        preconditions_str = "\n    ".join(pred.to_pddl() for pred in self.preconditions)
        effects_str = "\n    ".join(pred.to_pddl() for pred in self.effects)
        
        pddl = f"(:action {self.name}\n"
        if self.parameters:
            pddl += f"  :parameters ({params_str})\n"
        pddl += f"  :precondition (and\n    {preconditions_str}\n  )\n"
        pddl += f"  :effect (and\n    {effects_str}\n  )\n )\n"
        
        return pddl

    def to_dict(self) -> dict:
        """Return action as dictionary"""
        return {
            "name": self.name,
            "parameters": [param.to_dict() for param in self.parameters],
            "preconditions": [pred.to_dict() for pred in self.preconditions],
            "effects": [pred.to_dict() for pred in self.effects],
            # "thinking": self.thinking
        }
        
                
class List_of_Actions(BaseModel):
    """A preview list of actions that include this PDDL Domain.
    Including the action name and its parameters, e.g. 'stack ?x - block ?y - block'"""
    # thinking: str = Field(description="A brief space to think about the actions in the domain")
    actions: List[str] = Field(description="A list of actions in the domain")

    def to_dict(self) -> dict:
        """Return actions list as dictionary"""
        return {
            "actions": self.actions,
            # "thinking": self.thinking
        }
        
        
PDDLAvalibleActions = list[PDDL_Action]()
def set_avalible_actions(list_of_actions: List[PDDL_Action]) -> list[PDDL_Action]:
    """Crear un Literal con todos las acciones"""
    global PDDLAvalibleActions   
    PDDLAvalibleActions = list_of_actions
    return PDDLAvalibleActions


class HTN_Task(BaseModel):
    """A HTN task representation."""
    name: str = Field(description="The name of the task")
    parameters: List[PDDL_Parameter] = Field(description="A list of parameters for the task. Each parameter must be an instance of a PDDLParameter, which includes its name and type.")
    
    def to_pddl(self) -> str:
        """Convert to HDDL task syntax"""
        params_str = " ".join(param.to_pddl() for param in self.parameters)
        return f"(:task {self.name} :parameters ({params_str}))"
    
    def to_dict(self) -> dict:
        """Return task as dictionary"""
        return {
            "name": self.name,
            "parameters": [arg.to_dict() for arg in self.parameters],
        }

class List_of_Tasks(BaseModel):
    """A representation of the list of compound tasks in a HTN Domain."""
    # thinking: str = Field(description="A brief space to think about the compound tasks that are going to be needed within the domain")
    tasks: List[HTN_Task] = Field(description="A list of compound tasks in the domain")
    
    def to_dict(self) -> dict:
        """Return tasks list as dictionary"""
        return {
            "tasks": [task.to_dict() for task in self.tasks],
            # "thinking": self.thinking
        }


HTNAvalibleTasks = list[HTN_Task]()
def set_avalible_tasks(list_of_tasks: List[HTN_Task]) -> list[HTN_Task]:
    """Crear un Literal con todos las tareas"""
    global HTNAvalibleTasks   
    HTNAvalibleTasks = list_of_tasks
    return HTNAvalibleTasks


class HTN_Subtask(BaseModel):
    """A HTN subtask representation, which is an instance of a task or action."""
    name: str = Field(description="The name of the task or action")
    args: List[str] = Field(description="A list of arguments for the tasks. Each argument must be the name of a parameter, e.g., '?x'")

    def to_dict(self) -> dict:
        """Return subtask as dictionary"""
        return {
            "name": self.name,
            "arguments": self.args,
        }
    
    def to_pddl(self) -> str:
        """Convert to PDDL subtask syntax: '(name ?arg1 ?arg2)'"""
        params_str = " ".join(self.args)
        if params_str:
            return f"({self.name} {params_str})"
        return f"({self.name})" 

    @field_validator('name')
    @classmethod
    def validate_name(cls, name):
        original_task = next((t for t in HTNAvalibleTasks + PDDLAvalibleActions if t.name == name), None)
        if not original_task:
            raise ValueError(f"Subtask name '{name}' not found in available actions or tasks:\n - Actions: {[a.name for a in PDDLAvalibleActions]}\n - Tasks: {[t.name for t in HTNAvalibleTasks]}")
        return name
    

class List_of_Methods(BaseModel):
    """A preview list of methods that include this PDDL Domain.
    Including the method name and its parameters, e.g. 'assemble ?x - widget ?y - gadget'"""
    # thinking: str = Field(description="A brief space to think about the methods in the domain")
    methods: List[str] = Field(description="A list of methods in the domain")

    def to_dict(self) -> dict:
        """Return methods list as dictionary"""
        return {
            "methods": self.methods,
            # "thinking": self.thinking
        }


class HTN_Method(BaseModel):
    """A HTN method representation."""
    # description: str = Field(description="A brief description of the method")
    name: str = Field(description="The name of the method")
    parameters: List[PDDL_Parameter] = Field(description="A list of parameters for the method, including all the parameters that are going to be used in the subtasks.")
    task: HTN_Subtask = Field(description="The task that this method decomposes")
    preconditions: List[PDDL_Fact] = Field(description="A list of preconditions for the method.")
    ordered_subtasks: List[HTN_Subtask] = Field(description="A list of subtasks for the method.")
    
    def to_pddl(self) -> str:
        """Convert to PDDL method syntax"""
        params_str = " ".join(param.to_pddl() for param in self.parameters)
        
        preconditions_str = "\n    ".join(pred.to_pddl() for pred in self.preconditions)
        
        subtasks_str = "\n    ".join(subtask.to_pddl() for subtask in self.ordered_subtasks)
        
        pddl = f"(:method {self.name}\n"
        if self.parameters:
            pddl += f"  :parameters ({params_str})\n"
        pddl += f"  :task {self.task.to_pddl()}\n"
        pddl += f"  :precondition (and\n    {preconditions_str}\n  )\n"
        pddl += f"  :ordered-subtasks (and\n    {subtasks_str}\n  )\n )\n"
        
        return pddl
    
    def to_dict(self) -> dict:
        """Return method as dictionary"""
        return {
            # "description": self.description,
            "name": self.name,
            "parameters": [param.to_dict() for param in self.parameters],
            "task": self.task.to_dict(),
            "preconditions": [pred.to_dict() for pred in self.preconditions],
            "ordered_subtasks": [subtask.to_dict() for subtask in self.ordered_subtasks],
        }
    
    @model_validator(mode='after')
    def validate_subtasks(self):
        for subtask in self.ordered_subtasks + [self.task]:
            original_task = next((t for t in HTNAvalibleTasks + PDDLAvalibleActions if t.name == subtask.name), None)
            if not original_task:
                raise ValueError("Unexpected error: Subtask name is invalid")
            
            if original_task and len(subtask.args) != len(original_task.parameters):
                raise ValueError(f"Method task '{subtask.name}' expects {len(original_task.parameters)} arguments, got {len(subtask.args)}")

            for i, arg in enumerate(subtask.args):
                expected_type = original_task.parameters[i].type
                param_name = arg
                param = next((p for p in self.parameters if p.name == param_name), None)
                if not param:
                    raise ValueError(f"Argument '{param_name}' of method's task '{subtask.name}' not found in method parameters: {[p.name for p in self.parameters]}")
                if param.type != expected_type:
                    raise ValueError(f"Argument '{param_name}' of method's task '{subtask.name}' expects type '{expected_type}', got '{param.type}'")
                
        return self
        

class Initial_State(BaseModel):
    """A representation of the initial state of this Problem."""
    # thinking: str = Field(description="A brief space to think about the initial state")
    literals: List[PDDL_Fact] = Field(description="A list of the literals representing the initial state")
    
    def to_pddl(self) -> str:
        """Convert to PDDL initial state syntax"""
        literals_str = "\n    ".join(literal.to_pddl() for literal in self.literals)
        return f"(:init\n    {literals_str}\n  )"

    def to_dict(self) -> dict:
        """Return initial state as dictionary"""
        return {
            "literals": [literal.to_dict() for literal in self.literals],
            # "thinking": self.thinking
        }

class Goal(BaseModel):
    """A representation of the goal state of this Problem."""
    # thinking: str = Field(description="A brief space to think about the goal state")
    literals: List[PDDL_Fact] = Field(description="A list of the literals representing the goal state")
    
    def to_pddl(self) -> str:
        """Convert to PDDL goal syntax"""
        literals_str = "\n    ".join(pred.to_pddl() for pred in self.literals)
        return f"(:goal (and\n    {literals_str}\n    )\n  )"

    def to_dict(self) -> dict:
        """Return goal as dictionary"""
        return {
            "literals": [literal.to_dict() for literal in self.literals],
            # "thinking": self.thinking
        }
        
class HTN_Goal(BaseModel):
    """A representation of the HTN goal of this Problem."""
    # thinking: str = Field(description="A brief space to think about the HTN goal")
    literals: List[HTN_Subtask] = Field(description="A list of the tasks representing the HTN goal")
    
    def to_dict(self) -> dict:
        """Return HTN goal as dictionary"""
        return {
            "literals": [literal.to_dict() for literal in self.literals],
            # "thinking": self.thinking
        }
        
    def to_pddl(self) -> str:
        """Convert to PDDL HTN goal syntax"""
        literals_str = "\n      ".join(pred.to_pddl() for pred in self.literals)
        return f"""(:htn
    :ordered-subtasks (and\n      {literals_str}\n    )\n  )"""

        
class UltimatePDDLModeler:
    """Main class for generating PDDL domains and problems using Gemini API."""
    
    pddl_types: List_of_Types
    predicates: List_of_Predicates
    actions_preview: List_of_Actions
    actions: List[PDDL_Action]
    objects: List_of_Objects
    initial_state: Initial_State
    goal: Goal
    tasks: List_of_Tasks
    methods: List[HTN_Method]
    
    def __init__(self, client, requirements: List[str] = None) -> None:
        self.client = client
        self.requirements = requirements if requirements is not None else []
        if ':hierarchy' in self.requirements:
            self.isHTN = True
        else:
            self.isHTN = False
        
        # Automated retry for API rate limiting
        # This setup ensures requests are retried when per-minute quota is reached.
        # is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})

        # if not hasattr(genai.models.Models.generate_content, '__wrapped__'):
        #     genai.models.Models.generate_content = 
        # .Retry(
        #     predicate=is_retriable)(genai.models.Models.generate_content)
            
    def set_requirements(self, requirements: List[str]) -> None:
        """Set the PDDL requirements for the domain generation."""
        self.requirements = requirements
        if ':hierarchy' in self.requirements:
            self.isHTN = True
        else:
            self.isHTN = False
            
    def get_types(self, task_desc: str) -> List_of_Types:
        """Implementation to extract types from the task description"""
        return self.client.create(messages=task_desc, response_model=List_of_Types)
    
    def get_predicates(self, task_desc: str, pddltypes: List_of_Types) -> List_of_Predicates:
        """Implementation to extract predicates from the task description and types"""
        prompt = f"""{task_desc}
        You have the following available PDDL types: {pddltypes.types}
        """
        return self.client.create(messages=prompt, response_model=List_of_Predicates)
    
    def get_actions(self, task_desc: str, pddltypes: List_of_Types) -> List_of_Actions:
        """Implementation to extract actions from the task description and types"""
        prompt = f"""{task_desc}
        You have the following PDDL types: {pddltypes.types}"""
        return self.client.create(messages=prompt, response_model=List_of_Actions)
    
    def build_action(self, task_desc, action_str, pddltypes: List_of_Types, predicates) -> PDDL_Action:
        """Implementation to build a PDDL action from its string representation, types and predicates"""
        prompt = f"""{task_desc}
        You have the following available PDDL types: {pddltypes.types}
        You have the following PDDL predicates: {predicates}
        You can create new predicates if needed.
        Build the PDDL action for the following action string: {action_str}"""
        return self.client.create(messages=prompt, response_model=PDDL_Action)
    
    def get_tasks(self, task_desc: str, pddltypes: List_of_Types, actions: List_of_Actions) -> List_of_Tasks:
        """Implementation to extract the neccesary compound tasks from the task description, types and actions"""
        prompt = f"""{task_desc}
        You have the following available PDDL types: {pddltypes.types}
        You have the following PDDL actions: {actions}
        """
        return self.client.create(messages=prompt, response_model=List_of_Tasks)
    
    def get_methods_preview(self, task_desc: str, method_task: HTN_Task, pddltypes: List_of_Types, tasks: List_of_Tasks, actions: List_of_Actions) -> List_of_Methods:
        """Implementation to extract methods of an specific HTN Task, given the problem description, types, tasks and actions"""
        prompt = f"""{task_desc}
        You have the following available PDDL types: {pddltypes.types}
        You have the following available HTN tasks: {tasks.tasks}
        You have the following available PDDL actions: {actions}
        Generate a list of methods for the following HTN task: {method_task.name}"""
        return self.client.create(messages=prompt, response_model=List_of_Methods)
    
    def build_method(self, task_desc, method_name: str, method_task: HTN_Task, pddltypes: List_of_Types, pddlpredicates: List_of_Predicates, tasks: List_of_Tasks, actions: List[PDDL_Action]) -> HTN_Method:
        """Implementation to build a HTN method from its string representation, types, tasks and actions"""
        prompt = f"""{task_desc}
        You have the following available PDDL types: {pddltypes.types}
        You have the following available PDDL predicates: {pddlpredicates.predicates}
        You have the following available HTN compound tasks: {tasks.tasks}
        You have the following available PDDL actions: {actions}
        Build ONLY ONE HTN method with the name '{method_name}' that decomposes the following HTN task: '{method_task.name}'. Return only this single method definition, nothing else."""
        return self.client.create(messages=prompt, response_model=HTN_Method)

    def get_objects(self, task_desc: str, pddltypes: List_of_Types) -> List_of_Objects:
        """Implementation to extract objects from the task description and types"""
        prompt = f"""{task_desc}
        You have the following available PDDL types: {pddltypes.types}
        """
        return self.client.create(messages=prompt, response_model=List_of_Objects)
    
    def get_initial_state(self, task_desc: str, objects: List_of_Objects, predicates) -> Initial_State:
        """Implementation to extract the initial state from the task description, objects and predicates"""
        prompt = f"""{task_desc}
        You have the following available PDDL objects: {objects.objects}
        You have the following PDDL predicates: {predicates}
        Extract the initial state."""
        return self.client.create(messages=prompt, response_model=Initial_State)
    
    def get_goal(self, task_desc: str, objects: List_of_Objects, initial: Initial_State, predicates: List_of_Predicates, tasks: List_of_Tasks = None) -> Goal | HTN_Goal:
        """Implementation to extract the goal from the task description, objects and predicates"""
        prompt = f"""{task_desc}
        You have the following initial state: {initial.literals}
        You have the following available PDDL objects: {objects.objects}"""
        if self.isHTN:
            prompt += f"You have the following available tasks: {tasks.tasks}"
            prompt += " In HTN, the goal is to accomplish a list of ordered tasks, to be accomplished in sequence. Use the available tasks to define the goal, using the available objects to define their parameters."
            return self.client.create(messages=prompt, response_model=HTN_Goal)
        if not self.isHTN:
            prompt += f"You have the following available PDDL predicates: {predicates.predicates}"
            prompt += " The goal is to reach a specific state defined by a set of predicates. Extract that goal using the available objects to define their parameters."
            return self.client.create(messages=prompt, response_model=Goal)

    def generate_model(self, task_desc: str) -> Tuple[str, int]:
        """
        Runs the agent to extract the domain and problem from task description.
        
        :param task_desc: The task description to run the agent on.
        :return: Tuple of (status_string, exit_code)
        """
        try:
            # Extract types
            self.pddl_types = self.get_types(task_desc)
            print("✓ Types extracted")
            
            set_avalible_types(self.pddl_types)
            
            # Extract predicates
            self.predicates = self.get_predicates(task_desc, self.pddl_types)
            print("✓ Predicates extracted")
            
            set_avalible_predicates(self.predicates)
            
            # Extract actions
            self.actions_preview = self.get_actions(task_desc, self.pddl_types)
            print("✓ Actions preview extracted")
                       
            # Build actions
            self.actions = []
            for action_str in self.actions_preview.actions:
                action = self.build_action(task_desc, action_str, self.pddl_types, self.predicates.predicates)
                self.actions.append(action)
            print("✓ Actions built")
            
            set_avalible_actions(self.actions)
            
            # Extract tasks and methods if HTN
            if self.isHTN:
                self.tasks = self.get_tasks(task_desc, self.pddl_types, self.actions)
                print("✓ Compound tasks extracted")
                
                set_avalible_tasks(self.tasks.tasks)
                
                self.methods_preview = []
                for task in self.tasks.tasks:
                    methods_for_task = self.get_methods_preview(task_desc, task, self.pddl_types, self.tasks, self.actions)
                    self.methods_preview.append((task.name, methods_for_task))
                print("✓ Methods preview extracted")
            
                self.methods = []
                for task_name, methods_for_task in self.methods_preview:
                    for method_str in methods_for_task.methods:
                        print(f"Building method '{method_str}' for task '{task_name}'")
                        method_built = self.build_method(task_desc, method_str, next(t for t in self.tasks.tasks if t.name == task_name), self.pddl_types, self.predicates, self.tasks, self.actions)
                        self.methods.append(method_built)
                print("✓ Methods built")
                        
            # Extract objects
            self.objects = self.get_objects(task_desc, self.pddl_types)
            print("✓ Objects extracted")
            
            # Extract initial state
            self.initial_state = self.get_initial_state(task_desc, self.objects, self.predicates)
            print("✓ Initial state extracted")
            
            # Extract goal
            self.goal = self.get_goal(task_desc, self.objects, self.initial_state, self.predicates, self.tasks)
            print("✓ Goal extracted")
                        
            return "Success", 0
        
        except Exception as e:
            return f"Error extracting domain and problem: {e}\n" + traceback.format_exc(), 1
        
    def generate_domain(self, domain_name: str) -> str:
        """
        Generate a complete PDDL domain as a string.
        :param domain_name: Name of the domain
        :return: Complete PDDL domain as string
        """
        
        if self.requirements:
            requirements = self.requirements
        else:
            requirements = [":typing"]          
        
        requirements_str = " ".join(requirements)
        
        predicates_str = "\n    ".join(predicate.to_pddl() for predicate in self.predicates.predicates)
        
        if self.isHTN:
            tasks_str = "\n  ".join(f"{task.to_pddl()}" for task in self.tasks.tasks)
            methods_str = "\n  ".join(method.to_pddl() for method in self.methods)
            htn_section = f"""{tasks_str}\n
  {methods_str}\n"""
        else:
            htn_section = ""
        
        actions_str = "\n  ".join(action.to_pddl() for action in self.actions)
        
        domain = f"""(define (domain {domain_name})
  (:requirements {requirements_str})
  {self.pddl_types.to_pddl()}
  (:predicates
    {predicates_str}\n  )\n
  {htn_section}
  {actions_str}
)"""
        return domain
    
    def generate_problem(self,
        problem_name: str,
        domain_name: str
    ) -> str:
        """
        Generate a complete PDDL problem as a string.
        
        :param problem_name: Name of the problem
        :param domain_name: Name of the domain
        :param objects: List of objects
        :param initial_state: Initial state
        :param goal: Goal state
        :return: Complete PDDL problem as string
        """
        problem = f"""(define (problem {problem_name})
  (:domain {domain_name})"""
        if self.isHTN:
            problem += f"""
  {self.objects.to_pddl()}
  {self.goal.to_pddl()}
  {self.initial_state.to_pddl()}
)"""
        else:
            problem += f"""
  {self.objects.to_pddl()}
  {self.initial_state.to_pddl()}
  {self.goal.to_pddl()}
)"""
        return problem
    
    def to_dict(self) -> dict:
        """Return all modeler data as a dictionary"""
        dict_representation = {
            "types": self.pddl_types.to_dict() if hasattr(self, 'pddl_types') else None,
            "predicates": self.predicates.to_dict() if hasattr(self, 'predicates') else None,
            "actions_preview": self.actions_preview.to_dict() if hasattr(self, 'actions_preview') else None,
            "actions": [action.to_dict() for action in self.actions] if hasattr(self, 'actions') else None,
        }
        if self.isHTN:
            dict_representation.update({
                "tasks": self.tasks.to_dict() if hasattr(self, 'tasks') else None,
                "methods": [method.to_dict() for method in self.methods] if hasattr(self, 'methods') else None,
            })
        dict_representation.update({
            "objects": self.objects.to_dict() if hasattr(self, 'objects') else None,
            "initial_state": self.initial_state.to_dict() if hasattr(self, 'initial_state') else None,
            "goal": self.goal.to_dict() if hasattr(self, 'goal') else None,
        })
        
    def read_from_dict(self, data: dict) -> None:
        """Load all modeler data from a dictionary"""
        self.pddl_types = List_of_Types.model_validate(data['types'])
        self.predicates = List_of_Predicates.model_validate(data['predicates'])
        self.actions_preview = List_of_Actions.model_validate(data['actions_preview'])
        self.actions = [PDDL_Action.model_validate(action) for action in data['actions']]
        if self.isHTN:
            self.tasks = List_of_Tasks.model_validate(data['tasks'])
            self.methods = [HTN_Method.model_validate(method) for method in data['methods']]
        self.objects = List_of_Objects.model_validate(data['objects'])
        self.initial_state = Initial_State.model_validate(data['initial_state'])
        if self.isHTN:
            self.goal = HTN_Goal.model_validate(data['goal'])
        else:
            self.goal = Goal.model_validate(data['goal'])
            
# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    """
    Example usage of the PDDLModeler with Ollama.
    
    Requirements:
        pip install instructor ollama unified-planning
        ollama pull llama3.1  # or another model
        ollama serve  # Start Ollama server in another terminal
    """
    import sys
    
    # Example task description
    task_description = """
    Stack planning: We have 3 blocks (A, B, C) on a table.
    We need to stack them such that A is on B, and B is on C.
    We can only move one block at a time, and we can only pick up 
    a clear block (nothing on top of it).
    """
    
    print("=" * 70)
    print("PDDL Modeler Example - Block Stacking")
    print("=" * 70)
    
    try:
        # Create Ollama client
        print("\n1. Connecting to Ollama...")
        client = create_ollama_client(
            model="llama3.1",
            host="http://localhost:11434",
            max_retries=2
        )
        print("   ✓ Connected to Ollama")
        
        # Initialize modeler (STRIPS, not HTN)
        print("\n2. Initializing PDDL Modeler...")
        modeler = PDDLModeler(client, is_htn=False, max_retries=2, verbose=True)
        print("   ✓ Modeler initialized")
        
        # Generate full model
        print("\n3. Generating PDDL model from task description...")
        print(f"   Task: {task_description.strip()}\n")
        status, exit_code = modeler.generate_full_model(task_description)
        
        if exit_code == 0:
            # Export to PDDL
            print("\n4. Exporting to PDDL...")
            domain, problem = modeler.get_pddl("blocksworld", "stack_example")
            
            print("\n" + "=" * 70)
            print("GENERATED DOMAIN (first 500 chars)")
            print("=" * 70)
            print(domain[:500] + "...\n")
            
            print("=" * 70)
            print("GENERATED PROBLEM (first 500 chars)")
            print("=" * 70)
            print(problem[:500] + "...\n")
            
            # Save to files
            with open("blocksworld_domain.pddl", "w") as f:
                f.write(domain)
            with open("blocksworld_problem.pddl", "w") as f:
                f.write(problem)
            
            print("✓ Saved to blocksworld_domain.pddl and blocksworld_problem.pddl")
            print(f"\nSuccess! Generated valid PDDL model with:")
            print(f"  - {len(modeler.builder.type_names)} types")
            print(f"  - {len(modeler.builder.predicate_names)} predicates")
            print(f"  - {len(modeler.builder.action_names)} actions")
            print(f"  - {len(modeler.builder.object_names)} objects")
        else:
            print(f"\n✗ Generation failed: {status}")
            sys.exit(1)
            
    except ImportError as e:
        print(f"✗ Missing dependencies: {e}")
        print("\nInstall with:")
        print("  pip install instructor ollama unified-planning")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)