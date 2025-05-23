"""
This file contains classes of PDDL types
"""

from typing import TypedDict, NewType, List, Optional
from collections import OrderedDict
from dataclasses import dataclass

ParameterList = NewType(
    "ParameterList", OrderedDict[str, str]
)  # {param_name: param_type}
ObjectList = NewType("ObjectList", dict[str, str])  # {obj_name: obj_type}


class Predicate(TypedDict):
    name: str
    desc: Optional[str]
    raw: str
    params: ParameterList
    clean: str


class Action(TypedDict):
    name: str
    raw: str
    params: ParameterList
    preconditions: str
    effects: str
    

# Domain details data class including predicates and actions
@dataclass
class DomainDetails:
    name: str
    domain_desc: str
    domain_pddl: str
    types: str
    requirements: list[str]
    predicates: List[Predicate]  # List of Predicate objects
    actions: List[Action]  # List of Action objects


# Problem details data class
@dataclass
class ProblemDetails:
    name: str
    problem_desc: str
    problem_pddl: str
    objects: tuple[dict[str, str], str]
    initial: tuple[dict[str, str], str]
    goal: tuple[dict[str, str], str]


# Plan details data class
@dataclass
class PlanDetails:
    plan_pddl: str
    plan_nl: str


# HTN Types data classes
class HDDLTask(TypedDict):
    name: str
    desc: Optional[str]
    raw: str
    params: ParameterList
    clean: str

class HDDLMethod(TypedDict):
    name: str
    params: ParameterList
    desc: Optional[str]
    raw: str
    task: HDDLTask
    ordered_subtasks: str
        
class HPDLMethod(TypedDict):
    name: str
    params: ParameterList
    desc: Optional[str]
    raw: str
    ordered_subtasks: str
    
class HPDLTask(TypedDict):
    name: str
    desc: Optional[str]
    raw: str
    params: ParameterList
    clean: str
    methods: list[HPDLMethod]
