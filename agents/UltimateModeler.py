"""
Ultimate PDDL/HTN Modeler Framework

A modular framework for LLM-based PDDL/HDDL generation with Unified Planning validation.

═══════════════════════════════════════════════════════════════════════════════
FRAMEWORK OVERVIEW
═══════════════════════════════════════════════════════════════════════════════

This module provides a flexible API for generating planning domains and problems
from natural language descriptions. It's designed as a **framework** where users
can either use the full pipeline or compose their own workflows from individual
methods.

Key Components:
    - PDDLModeler: Main class with modular API for custom generation workflows
    - PDDLBuilder: Unified Planning wrapper for validation (used internally)
    - Pydantic Models: Structured output schemas for LLM responses
    - StructuredOutputClient: Backend-agnostic LLM interface

═══════════════════════════════════════════════════════════════════════════════
USAGE PATTERNS
═══════════════════════════════════════════════════════════════════════════════

1. SIMPLE - Full automatic generation:

    from UltimateModeler import create_structured_client, PDDLModeler
    
    client = create_structured_client(model="llama3.1", mode="instructor")
    modeler = PDDLModeler(client, is_htn=False)
    modeler.generate_full_model("Stack blocks A on B on C")
    domain, problem = modeler.get_pddl()

2. CUSTOM WORKFLOW - Step-by-step control:

    modeler = PDDLModeler(client, is_htn=True)
    
    # Generate domain structure
    modeler.generate_types(task_desc)
    modeler.generate_predicates(task_desc)
    modeler.generate_objects(task_desc)
    
    # Actions with filtering
    action_names = modeler.generate_action_names(task_desc)
    action_names = [n for n in action_names if 'debug' not in n]
    for name in action_names:
        modeler.generate_action(task_desc, name)
    
    # HTN: Tasks and methods
    modeler.generate_tasks(task_desc)
    for task in modeler.tasks:
        method_names = modeler.generate_method_names(task_desc, task)
        for method in method_names:
            modeler.generate_method(task_desc, task, method)
    
    # Problem instance
    modeler.generate_initial_state(task_desc)
    modeler.generate_goal(task_desc)

3. INSPECTION - Check intermediate results:

    # Access generated content at any point
    print(f"Types: {modeler.types}")
    print(f"Actions: {modeler.actions}")
    
    # Get raw LLM outputs
    for action in modeler.raw_actions.actions:
        print(f"{action.name}: {len(action.preconditions)} preconditions")

═══════════════════════════════════════════════════════════════════════════════
STRUCTURED OUTPUT BACKENDS
═══════════════════════════════════════════════════════════════════════════════

    - INSTRUCTOR: Pydantic validation with retry (recommended for complex schemas)
    - OUTLINES: Grammar-constrained generation (guarantees valid JSON)
    - RAW_OLLAMA: Native Ollama JSON mode (simplest, minimal dependencies)

═══════════════════════════════════════════════════════════════════════════════

See PDDLModeler class docstring for complete API reference.
"""

from __future__ import annotations
import sys
import os
import time
import traceback
import logging
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Literal, Optional, Dict, Tuple, Union, Any, Callable, Type, TypeVar, Set
from numpy import trace
from pydantic import BaseModel, Field, field_validator, model_validator

# Unified Planning imports
import unified_planning as up
from unified_planning.model import Problem as UPProblem
from unified_planning.model.htn import HierarchicalProblem, Method
from unified_planning.io import PDDLWriter
from unified_planning.shortcuts import OneshotPlanner
from unified_planning.shortcuts import Not

# Ollama
from ollama import Client as OllamaClient

# Configure logging
logger = logging.getLogger(__name__)

# Type variable for Pydantic models
T = TypeVar('T', bound=BaseModel)


# =============================================================================
# CONFIGURATION CONSTANTS - Centralized defaults for all framework parameters
# =============================================================================
# All framework defaults are centralized here for easy customization.
# To change default behavior, modify these constants before creating PDDLModeler instances.
#
# CONSTANT CATEGORIES:
#   - Ollama connection: Server and model names
#   - LLM behavior: Retry policy and temperatures for different phases
#   - Model generation: HTN mode, analysis enable
#   - Logging/debugging: Log levels, metrics, interactive mode, log file
#   - PDDL naming: Domain and problem names
#   - PDDL semantics: Default fact values
#   - Method parameters: Parameter defaults used across API methods
#   - Logging format: Display limits for traces and warnings
#
# Usage example:
#   DEFAULT_MAX_RETRIES = 5  # Change retry attempts globally
#   modeler = PDDLModeler(client)  # Uses custom constant
# =============================================================================

# Ollama connection defaults
DEFAULT_OLLAMA_MODEL = "gemma3"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"

# LLM behavior parameters
DEFAULT_MAX_RETRIES = 3
DEFAULT_STRUCTURED_OUTPUT_MODE = "raw_ollama"  # "INSTRUCTOR=0", "OUTLINES=1", or "RAW_OLLAMA=2"

# Model generation defaults
DEFAULT_IS_HTN = False
DEFAULT_USE_ANALYSIS = False  # Enable initial problem analysis for better context
DEFAULT_ALLOW_DUPLICATE_NAMES = True  # Disallow duplicate names by default
DEFAULT_THINK_MODE = False  # Enable two-step reasoning: think first, then generate JSON

# Logging and debugging defaults
DEFAULT_LOG_LEVEL = 'info'  # QUIET=0, INFO=1, DEBUG=2, TRACE=3
DEFAULT_TRACK_METRICS = True  # Enable metrics tracking by default
DEFAULT_INTERACTIVE_MODE = False  # Enable interactive mode by default (pauses at each step)
DEFAULT_LOG_FILE = None  # No log file by default

# PDDL naming defaults
DEFAULT_DOMAIN_NAME = "PlaceHolder"
DEFAULT_PROBLEM_NAME = "PlaceHolderP"

# PDDL semantic defaults
DEFAULT_INITIAL_FACT_VALUE = True
DEFAULT_GOAL_FACT_NEGATED = False

# Method/Function parameter defaults
DEFAULT_SHOW_METRICS = True  # Show metrics summary in generate_full_model()
DEFAULT_ALLOW_NONE_ANALYSIS = False  # Allow None values in analysis results
DEFAULT_AS_PARAM = False  # Default for parameter normalization
DEFAULT_DYN_CONTEXT_LIMIT = 2000  # Character limit for dynamic context inclusion

# Logging format defaults
DEFAULT_PROMPT_LOG_LIMIT = 2000  # Max characters to display in TRACE logs for prompts
DEFAULT_RESPONSE_LOG_LIMIT = 3000  # Max characters to display in TRACE logs for responses
DEFAULT_PROMPT_DISPLAY_SUFFIX = "..."  # Suffix when truncating logs
DEFAULT_MODEL_STATE_PREVIEW_ITEMS = -1  # Max items to show per component in state display. If -1, show all items.
DEFAULT_WARNING_SAMPLE_SIZE = 5  # Max items to show in warning messages about missing items

# General defaults
DEFAULT_LOG_MESSAGE_END = "\n"  # Line ending for log messages
DEFAULT_JSON_INDENT = 2  # JSON indentation spaces
DEFAULT_METRICS_TOP_N = 3  # Number of top items to show in metrics
DEFAULT_TASK_INPUT_OPTION = "3"  # Default task input option (1=input, 2=example, 3=debug, 4=file, 5=quit)
DEFAULT_EXECUTE_PLANNER = True  # Default option to execute planner after generation

# Temperature configurations
TEMPERATURE_OVERRIDE = 0  # Set this to override ALL temperatures at once (None = use TEMPERATURE_CONFIG).
TEMPERATURE_CONFIG = {    # Temperature configuration for generation phases
    "analysis": 0.7,        # Creative: needs to infer entities and relationships
    "think": 0.5,           # Balanced: reason about structure before generation
    "types": 0.5,           # Balanced: some creativity for type hierarchy
    "predicates": 0.6,      # Balanced: needs to invent good relations
    "actions": 0.5,         # Lower: preconditions/effects need precision
    "tasks": 0.6,           # Balanced: abstract goal definition
    "methods": 0.4,         # Lower: decomposition needs precision
    "objects": 0.3,         # Low: should match problem description exactly
    "initial_state": 0.2,   # Very low: factual, must match objects/predicates
    "goal": 0.3,            # Low: must be achievable and precise
    "default": 0.5,         # Fallback
}

def get_temperature(step_key: str) -> float:
    """
    Get the temperature for a step.
    
    If TEMPERATURE_OVERRIDE is set, uses that for all steps.
    Otherwise uses TEMPERATURE_CONFIG per-step.
    
    Args:
        step_key: The step identifier (e.g., "actions", "analysis")
    
    Returns:
        The temperature value to use
    """
    if TEMPERATURE_OVERRIDE is not None:
        return TEMPERATURE_OVERRIDE
    return TEMPERATURE_CONFIG.get(step_key, TEMPERATURE_CONFIG["default"])

# =============================================================================
# LOGGING SYSTEM - Unified logging with levels and optional interactivity
# =============================================================================

class LogLevel(Enum):
    """
    Log verbosity levels for PDDLModeler.
    
    QUIET: Only errors and final result
    INFO: Phase progress (default)
    DEBUG: Detailed step info, attempts, errors
    TRACE: Everything including prompts and responses
    """
    QUIET = 0   # Minimal output
    INFO = 1    # Phase progress (default)
    DEBUG = 2   # Detailed execution info
    TRACE = 3   # Full prompts and responses


# Dynamic context compression - fields relevant to each step
# Determines what context to include in prompts to avoid exceeding token limits
CONTEXT_RELEVANCE = {
    "analysis": {"analysis": [], "builder": []},  # Analysis step doesn't need prior context
    "types": {"analysis": ["types", "entities"], "builder": []},
    "predicates": {"analysis": ["initial", "goals", "capabilities"], "builder": ["types"]},
    "objects": {"analysis": ["entities"], "builder": ["types", "predicates"]},
    "actions": {"analysis": ["capabilities"], "builder": ["types", "predicates", "objects"]},
    "tasks": {"analysis": ["goals", "capabilities"], "builder": ["types", "actions"]},
    "methods": {"analysis": ["capabilities"], "builder": ["types", "predicates", "actions", "tasks"]},
    "initial_state": {"analysis": ["initial"], "builder": ["predicates", "objects"]},
    "goal": {"analysis": ["goals"], "builder": ["predicates", "objects", "tasks"]},
}

# =============================================================================
# CUSTOM EXCEPTIONS
# =============================================================================

class PDDLGenerationError(Exception):
    """Raised when PDDL generation fails after all retry attempts."""
    
    def __init__(self, step_name: str, attempts: int, error_messages: List[str], traceback_str: str = ""):
        import traceback as tb_module
        
        self.step_name = step_name
        self.attempts = attempts
        self.error_messages = error_messages
        self.traceback_str = traceback_str or tb_module.format_exc()
        
        message = f"Failed to generate valid {step_name.lower()} after {attempts} attempts:\n"
        message +=f"".join(f"\n - Attempt {i+1}:  {e}" for i, e in enumerate(error_messages))
        # Add traceback if available
        if self.traceback_str and self.traceback_str.strip():
            message += f"\n\nTraceback:\n{self.traceback_str}"
        
        super().__init__(message)


# =============================================================================
# GENERATION METRICS - Track statistics during model generation
# =============================================================================

@dataclass
class StepMetrics:
    """
    Metrics for a single generation step.
    
    Tracks timing, attempts, and success/failure information for one
    component generation (e.g., types, predicates, actions).
    """
    step_name: str
    start_time: float = 0.0
    end_time: float = 0.0
    attempts: int = 0
    success: bool = False
    first_attempt_success: bool = False
    error_messages: List[str] = field(default_factory=list)
    
    @property
    def duration_seconds(self) -> float:
        """Total time spent on this step in seconds."""
        return self.end_time - self.start_time if self.end_time else 0.0
    
    @property
    def duration_formatted(self) -> str:
        """Human-readable duration string."""
        secs = self.duration_seconds
        if secs < 60:
            return f"{secs:.2f}s"
        return f"{int(secs // 60)}m {secs % 60:.1f}s"


@dataclass
class GenerationMetrics:
    """
    Complete metrics for a full model generation run.
    
    Aggregates all step metrics and provides summary statistics for
    analyzing generation performance, identifying bottlenecks, and
    tuning retry parameters.
    """
    steps: Dict[str, StepMetrics] = field(default_factory=dict)
    total_start_time: float = 0.0
    total_end_time: float = 0.0
    
    def start_step(self, step_name: str) -> StepMetrics:
        """Begin tracking a new generation step."""
        step = StepMetrics(step_name=step_name, start_time=time.time())
        self.steps[step_name] = step
        return step
    
    def end_step(self, step_name: str, success: bool, attempts: int, 
                 errors: Optional[List[str]] = None) -> None:
        """Finalize metrics for a generation step."""
        if step_name in self.steps:
            step = self.steps[step_name]
            step.end_time = time.time()
            step.success = success
            step.attempts = attempts
            step.first_attempt_success = success and attempts == 1
            step.error_messages = errors or []
    
    def start_generation(self) -> None:
        """Mark the start of a full generation run."""
        self.total_start_time = time.time()
        self.steps.clear()
    
    def end_generation(self) -> None:
        """Mark the end of a full generation run."""
        self.total_end_time = time.time()
    
    @property
    def total_duration_seconds(self) -> float:
        """Total generation time in seconds."""
        return self.total_end_time - self.total_start_time if self.total_end_time else 0.0
    
    @property
    def total_attempts(self) -> int:
        """Sum of attempts across all steps."""
        return sum(s.attempts for s in self.steps.values())
    
    @property
    def successful_steps(self) -> int:
        """Number of steps that succeeded."""
        return sum(1 for s in self.steps.values() if s.success)
    
    @property
    def first_attempt_success_rate(self) -> float:
        """Fraction of steps that succeeded on first attempt (0.0 to 1.0)."""
        if not self.steps:
            return 0.0
        return sum(1 for s in self.steps.values() if s.first_attempt_success) / len(self.steps)
    
    @property
    def overall_success_rate(self) -> float:
        """Fraction of steps that eventually succeeded (0.0 to 1.0)."""
        if not self.steps:
            return 0.0
        return self.successful_steps / len(self.steps)
    
    def get_slowest_steps(self, n: int = DEFAULT_METRICS_TOP_N) -> List[Tuple[str, float]]:
        """Get the n slowest steps by duration."""
        sorted_steps = sorted(
            [(s.step_name, s.duration_seconds) for s in self.steps.values()],
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_steps[:n]
    
    def get_most_retried_steps(self, n: int = DEFAULT_METRICS_TOP_N) -> List[Tuple[str, int]]:
        """Get the n steps with most retry attempts."""
        sorted_steps = sorted(
            [(s.step_name, s.attempts) for s in self.steps.values()],
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_steps[:n]
    
    def summary(self) -> str:
        """Generate a human-readable summary of all metrics."""
        lines = [
            "═" * 60,
            "GENERATION METRICS SUMMARY",
            "═" * 60,
            f"Total Duration: {self.total_duration_seconds:.2f}s",
            f"Total Steps: {len(self.steps)}",
            f"Successful Steps: {self.successful_steps}/{len(self.steps)}",
            f"Total Attempts: {self.total_attempts}",
            f"First-Attempt Success Rate: {self.first_attempt_success_rate:.1%}",
            f"Overall Success Rate: {self.overall_success_rate:.1%}",
            "",
            "Step Details:",
            "-" * 60,
        ]
        
        for step in self.steps.values():
            status = "✓" if step.success else "✗"
            first = "(1st)" if step.first_attempt_success else f"({step.attempts})"
            lines.append(
                f"  {status} {step.step_name:25} {step.duration_formatted:>10} {first:>6}"
            )
            if step.error_messages:
                for err in step.error_messages[-1:]:
                    err_summary = err[:10].replace('\n', '\t').replace('\t', ' ') + " (...) " + err[-110:].replace('\n', '\t').replace('\t', ' ') if len(err) > 120 else err.replace('\n', '\t').replace('\t', ' ')
                    lines.append(f"      └─ Last error: {err_summary}")
        
        lines.append("-" * 60)
        
        # Slowest steps
        slowest = self.get_slowest_steps(3)
        if slowest:
            lines.append("Slowest Steps:")
            for name, duration in slowest:
                lines.append(f"  • {name}: {duration:.2f}s")
        
        # Most retried
        most_retried = [(n, a) for n, a in self.get_most_retried_steps(3) if a > 1]
        if most_retried:
            lines.append("Most Retried Steps:")
            for name, attempts in most_retried:
                lines.append(f"  • {name}: {attempts} attempts")
        
        lines.append("═" * 60)
        return "\n".join(lines)
    
    def to_dict(self) -> dict:
        """Export metrics as a dictionary for JSON serialization."""
        return {
            "total_duration_seconds": self.total_duration_seconds,
            "total_steps": len(self.steps),
            "successful_steps": self.successful_steps,
            "total_attempts": self.total_attempts,
            "first_attempt_success_rate": self.first_attempt_success_rate,
            "overall_success_rate": self.overall_success_rate,
            "steps": {
                name: {
                    "duration_seconds": step.duration_seconds,
                    "attempts": step.attempts,
                    "success": step.success,
                    "first_attempt_success": step.first_attempt_success,
                    "errors": step.error_messages,
                }
                for name, step in self.steps.items()
            }
        }


# =============================================================================
# SHARED VALIDATION HELPERS
# =============================================================================

def _normalize_name(value: str) -> str:
    """Core normalization: lowercase, strip, replace spaces/hyphens with underscores."""
    return value.lower().strip().replace(" ", "_").replace("-", "_")


def _normalize(value: Any, field: str = "Value", *, allow_none: bool = False, as_param: bool = False) -> Optional[str]:
    """
    Universal normalizer for PDDL identifiers.
    
    Args:
        value: Value to normalize
        field: Field name for error messages
        allow_none: If True, None values are allowed
        as_param: If True, adds ? prefix (for PDDL parameters)
    """
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{field} cannot be None")
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string, got {type(value).__name__}")
    
    clean = _normalize_name(value.lstrip('?') if as_param else value)
    if not clean:
        raise ValueError(f"{field} cannot be empty (received: '{value}')")
    if not clean[0].isalpha():
        raise ValueError(
            f"{field} '{clean}' must start with a letter. "
            f"PDDL identifiers cannot start with numbers or special characters."
        )
    if not all(c.isalnum() or c == '_' for c in clean):
        invalid_chars = [c for c in clean if not (c.isalnum() or c == '_')]
        raise ValueError(
            f"{field} '{clean}' contains invalid characters: {invalid_chars}. "
            f"Use only letters, numbers, and underscores."
        )
    
    return f"?{clean}" if as_param else clean


def _validate_list(v: Any, name: str, required: bool = False) -> list:
    """Validate list, optionally non-empty."""
    if not isinstance(v, list):
        raise ValueError(f"{name} must be a list, got {type(v).__name__}")
    if required and not v:
        raise ValueError(f"At least one {name.lower()} is required")
    return v


def _normalize_args(value: Any, *, as_params: bool = True) -> List[str]:
    """Normalize a list of arguments, optionally adding ? prefix."""
    if not isinstance(value, list):
        raise ValueError("Arguments must be a list")
    return [_normalize(arg, "Argument", as_param=as_params) for arg in value]


def _check_duplicate_names(items: List, name_getter, item_type: str) -> None:
    """
    Check for duplicate names in a list of items.
    
    Args:
        items: List of items to check
        name_getter: Function to extract name from item (can be str for dicts or callable)
        item_type: Type name for error messages (e.g., "type", "predicate")
    
    Raises:
        ValueError: If duplicates are found, with details about which names are duplicated
    """
    seen = {}
    duplicates = []
    
    for i, item in enumerate(items):
        if callable(name_getter):
            name = name_getter(item)
        elif isinstance(item, dict):
            name = item.get(name_getter, f"item_{i}")
        else:
            name = getattr(item, name_getter, f"item_{i}")
        
        normalized = _normalize_name(str(name))
        if normalized in seen:
            duplicates.append((normalized, seen[normalized], i))
        else:
            seen[normalized] = i
    
    if duplicates:
        dup_details = [f"'{name}' appears at positions {first+1} and {second+1}" 
                      for name, first, second in duplicates]
        raise ValueError(
            f"Duplicate {item_type} names found:\n  " + "\n  ".join(dup_details) + "\n"
            f"Each {item_type} must have a unique name."
        )


def _check_duplicate_parameters(parameters: List, context: str) -> None:
    """
    Check for duplicate parameter names in a parameter list.
    
    Args:
        parameters: List of ParameterDef or dicts with 'name' key
        context: Context for error message (e.g., "action 'pick_up'")
    """
    seen = set()
    duplicates = []
    
    for param in parameters:
        name = param.get('name') if isinstance(param, dict) else getattr(param, 'name', None)
        if name:
            normalized = _normalize_name(name.lstrip('?'))
            if normalized in seen:
                duplicates.append(normalized)
            seen.add(normalized)
    
    if duplicates:
        raise ValueError(
            f"Duplicate parameter names in {context}: {duplicates}. "
            f"Each parameter must have a unique name."
        )


def _validate_fact_args_are_parameters(fact: dict, param_names: set, context: str) -> None:
    """
    Validate that all arguments in a fact are defined parameters.
    
    Args:
        fact: Fact dictionary with 'predicate' and 'args'
        param_names: Set of valid parameter names (normalized, without ?)
        context: Context for error message
    """
    predicate = fact.get('predicate', 'unknown')
    args = fact.get('args', [])
    
    invalid_args = []
    for arg in args:
        if isinstance(arg, str) and arg.startswith('?'):
            normalized = _normalize_name(arg.lstrip('?'))
            if normalized not in param_names:
                invalid_args.append(arg)
    
    if invalid_args:
        raise ValueError(
            f"In {context}, predicate '{predicate}' uses undefined parameters: {invalid_args}.\n"
            f"Available parameters: {['?' + p for p in sorted(param_names)]}.\n"
            f"All arguments must be parameters defined in this action/method."
        )


# =============================================================================
# STRUCTURED OUTPUT MODE - Configurable backend selection
# =============================================================================

class StructuredOutputMode(Enum):
    """
    Available modes for structured output generation.
    
    INSTRUCTOR: Uses instructor library for Pydantic validation with automatic retry.
                - Adds schema to prompt automatically
                - Validates response with Pydantic
                - Retries with error feedback on validation failure
                Best for: Complex schemas, when you need error correction
                Requires: pip install instructor ollama
    
    OUTLINES: Uses outlines library for grammar-constrained token generation.
              - Constrains token sampling to valid JSON at generation time
              - Guarantees syntactically valid output matching schema
              - Uses outlines.from_ollama() for direct integration
              Best for: Guaranteed structure, faster inference, simpler schemas
              Requires: pip install outlines ollama
    
    RAW_OLLAMA: Uses Ollama's native format parameter with JSON schema.
                - Passes JSON schema directly to Ollama API
                - Ollama constrains output to match schema
                - Simple Pydantic validation after generation
                Best for: Minimal dependencies, testing, simple use cases
                Requires: pip install ollama
    """
    INSTRUCTOR = 1
    OUTLINES = 2
    RAW_OLLAMA = 3


# =============================================================================
# STRUCTURED OUTPUT CLIENT INTERFACE
# =============================================================================

class StructuredOutputClient(ABC):
    """
    Abstract interface for structured output generation clients.
    
    All backend implementations must provide a `create` method that:
    1. Takes messages (chat format) and a Pydantic response_model
    2. Returns a validated instance of the response_model
    3. Raises exceptions on validation failure
    """
    _ollama_client: OllamaClient
    _model: str
    
    def chat(self, messages: List[Dict[str, str]] | str, temperature: Optional[float]) -> str:
        """Generic chat method for raw text responses (not structured)."""
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        
        if temperature != 0.0:
            print(f"[WARNING] Model not deterministic: Temperature={float(temperature)}")
        
        response = self._ollama_client.chat(
            model=self._model,
            messages=messages,
            options={"temperature": temperature}
        )
        return response['message']['content']
    
    @abstractmethod
    def create(self, messages: List[Dict[str, str]], response_model: Type[T], temperature: Optional[float] = None) -> T:
        """
        Generate structured output from the LLM.
        
        Args:
            messages: Chat messages in [{"role": "user", "content": "..."}] format
            response_model: Pydantic model class defining the expected output structure
            temperature: Optional temperature override (0.0-1.0). If None, uses default.
            
        Returns:
            Validated instance of response_model
            
        Raises:
            Exception: If generation or validation fails
        """
    
    @property
    @abstractmethod
    def mode(self) -> StructuredOutputMode:
        """Return the structured output mode this client uses."""
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the name of the LLM model being used."""


# =============================================================================
# INSTRUCTOR CLIENT - Pydantic validation with automatic retry
# =============================================================================

class InstructorClient(StructuredOutputClient):
    """
    Structured output client using the Instructor library.
    
    Instructor intercepts LLM calls and:
    1. Adds JSON schema to the prompt automatically
    2. Validates response with Pydantic
    3. Retries with error feedback if validation fails
    
    This is the most robust option for complex schemas where validation
    errors are likely and automatic correction is desired.
    """
    
    def __init__(
        self,
        model: str = DEFAULT_OLLAMA_MODEL,
        host: str = DEFAULT_OLLAMA_HOST,
        max_retries: int = DEFAULT_MAX_RETRIES,
        **kwargs
    ):
        try:
            import instructor
        except ImportError as e:
            raise ImportError(
                "Instructor mode requires: pip install instructor ollama\n"
                f"Original error: {e}"
            ) from e
        
        self._model = model
        self._host = host
        self._max_retries = max_retries
        
        # Create Ollama client and patch with instructor
        self._ollama_client = OllamaClient(host=host)
        self._client = instructor.from_provider(
            'ollama/' + model,
            **kwargs
        )
    
    def create(
        self,
        messages: List[Dict[str, str]],
        response_model: Type[T],
        temperature: Optional[float] = None
    ) -> T:
        """Generate structured output using instructor validation with retry."""
        # Instructor doesn't directly support temperature, but we can pass options
        # Note: This depends on instructor version and backend support
        
        if temperature != 0.0:
            print(f"[WARNING] Model not deterministic: Temperature={float(temperature)}")

        return self._client.create(
            messages=messages,
            response_model=response_model,
            temperature=temperature
        )
    
    @property
    def mode(self) -> StructuredOutputMode:
        return StructuredOutputMode.INSTRUCTOR
    
    @property
    def model_name(self) -> str:
        return self._model


# =============================================================================
# OUTLINES CLIENT - Grammar-constrained token generation
# =============================================================================

class OutlinesClient(StructuredOutputClient):
    """
    Structured output client using the Outlines library.
    
    Outlines constrains token generation at sampling time using a finite
    state machine derived from the JSON schema. This guarantees that the
    output is always valid JSON matching the schema structure.
    
    Best for: Guaranteed structure, faster inference, simpler schemas.
    """
    
    def __init__(
        self,
        model: str = DEFAULT_OLLAMA_MODEL,
        host: str = DEFAULT_OLLAMA_HOST
    ):
        try:
            import outlines
        except ImportError as e:
            raise ImportError(
                "Outlines mode requires: pip install outlines ollama\n"
                f"Original error: {e}"
            ) from e
        
        self._model = model
        self._host = host
        
        # Create Ollama client and wrap with outlines
        self._ollama_client = OllamaClient(host=host)
        self._client = outlines.from_ollama(self._ollama_client, model)
    
    def create(self, messages: List[Dict[str, str]], response_model: Type[T],
               temperature: Optional[float] = None) -> T:
        """Generate with grammar constraints."""       
        # Convert messages to prompt string
        prompt = self._messages_to_prompt(messages)
        
        # Use provided temperature or default
        temp = temperature if temperature is not None else get_temperature("default")
        if temp != 0.0:
            print(f"[WARNING] Model not deterministic: Temperature={float(temp)}")
        
        # Generate with grammar constraints - returns validated Pydantic object
        content = self._client(prompt, response_model, temp)
        
        try:
            data = json.loads(content)
            return response_model.model_validate(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON: {e}") from e
        except Exception as e:
            raise ValueError(f"Pydantic validation failed: {e}") from e
            
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert chat messages to a single prompt string."""
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"System: {content}\n")
            elif role == "user":
                parts.append(content)
            elif role == "assistant":
                parts.append(f"Assistant: {content}\n")
        return "\n".join(parts)
    
    @property
    def mode(self) -> StructuredOutputMode:
        return StructuredOutputMode.OUTLINES
    
    @property
    def model_name(self) -> str:
        return self._model


# =============================================================================
# RAW OLLAMA CLIENT - Native JSON schema support
# =============================================================================

class RawOllamaClient(StructuredOutputClient):
    """
    Structured output client using Ollama's native format parameter.
    
    Ollama's API supports passing a JSON schema directly via the `format`
    parameter. The model will constrain its output to match the schema.
    
    This is the simplest option with minimal dependencies.
    """
    
    def __init__(
        self,
        model: str = DEFAULT_OLLAMA_MODEL,
        host: str = DEFAULT_OLLAMA_HOST
    ):       
        self._model = model
        self._host = host
        self._ollama_client = OllamaClient(host=host)
        self._client = self._ollama_client
    
    def create(self, messages: List[Dict[str, str]], response_model: Type[T],
               temperature: Optional[float] = None) -> T:
        """Generate using Ollama's JSON schema format."""
        # Get JSON schema from Pydantic model
        json_schema = response_model.model_json_schema()
        
        # Use provided temperature or default
        temp = temperature if temperature is not None else get_temperature("default")
        if temp != 0.0:
            print(f"[WARNING] Model not deterministic: Temperature={float(temp)}")
            
        # Call Ollama with JSON schema as format
        response = self._client.chat(
            model=self._model,
            messages=messages,
            format=json_schema,  # Ollama natively supports JSON schema
            options={"temperature": temp}
        )
        
        # Parse and validate with Pydantic
        content = response['message']['content']
        
        try:
            data = json.loads(content)
            return response_model.model_validate(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON: {e}") from e
        except Exception as e:
            raise ValueError(f"Pydantic validation failed: {e}") from e
    
    @property
    def mode(self) -> StructuredOutputMode:
        return StructuredOutputMode.RAW_OLLAMA
    
    @property
    def model_name(self) -> str:
        return self._model


# =============================================================================
# CLIENT FACTORY
# =============================================================================

def create_structured_client(
    model: str = DEFAULT_OLLAMA_MODEL,
    host: str = DEFAULT_OLLAMA_HOST,
    mode: str = DEFAULT_STRUCTURED_OUTPUT_MODE,
    max_retries: int = DEFAULT_MAX_RETRIES,
    **kwargs
) -> StructuredOutputClient:
    """
    Factory function to create a structured output client.
    
    Args:
        model: Ollama model name (e.g., "llama3.1", "qwen2.5", "mistral")
        host: Ollama server URL
        mode: Structured output mode as string: "instructor", "outlines", or "raw_ollama"
        max_retries: Maximum retry attempts (only used by INSTRUCTOR mode)
        **kwargs: Additional arguments passed to the client
    
    Returns:
        StructuredOutputClient instance configured for the selected mode
    """
    # Convert string mode to enum
    mode_enum = StructuredOutputMode[mode.upper()] if isinstance(mode, str) else mode
    
    if mode_enum == StructuredOutputMode.INSTRUCTOR:
        return InstructorClient(model=model, host=host, max_retries=max_retries, **kwargs)
    elif mode_enum == StructuredOutputMode.OUTLINES:
        return OutlinesClient(model=model, host=host)
    elif mode_enum == StructuredOutputMode.RAW_OLLAMA:
        return RawOllamaClient(model=model, host=host)
    else:
        raise ValueError(f"Unknown structured output mode: {mode}")


# =============================================================================
# PDDL BUILDER - Core validation engine using Unified Planning
# =============================================================================

class PDDLBuilder:
    """
    Builds and validates PDDL/HTN models using Unified Planning.
    
    This is the single source of truth for all validation. Every type, predicate,
    action, task, method, and object is validated through the Unified Planning
    library when added to the model.
    
    The builder maintains internal dictionaries for quick lookup and provides
    helper methods that follow the "Tell, Don't Ask" pattern - they either
    succeed or raise descriptive errors.
    """
    
    def __init__(self, is_htn: bool = DEFAULT_IS_HTN, allow_duplicate_names: bool = DEFAULT_ALLOW_DUPLICATE_NAMES):
        self.is_htn = is_htn
        self.env = up.environment.get_environment()
        self.env.error_used_name = not allow_duplicate_names  # Allow same names for different elements
        self.problem: Union[UPProblem, HierarchicalProblem] = (
            HierarchicalProblem() if is_htn else UPProblem()
        )
        self._types: Dict[str, up.model.Type] = {}
        self._fluents: Dict[str, up.model.Fluent] = {}
        self._actions: Dict[str, up.model.InstantaneousAction] = {}
        self._tasks: Dict[str, Any] = {}
        self._methods: List[Any] = []
        self._method_names: Set[str] = set()
        self._objects: Dict[str, up.model.Object] = {}
    
    # Internal Helpers
    
    def _require_type(self, type_name: str, ctx: str = "") -> up.model.Type:
        """Get a type by name, raising a descriptive error if not found."""
        name = _normalize_name(type_name)
        if name not in self._types:
            available_types = list(self._types.keys())
            raise ValueError(
                f"Type '{name}' does not exist{ctx}.\n"
                f"  Available types: {available_types}\n"
                f"  Suggestion: Check spelling or define this type first. "
                f"Types must be defined before they can be used."
            )
        return self._types[name]
    
    def _require_fluent(self, pred_name: str, ctx: str = "") -> up.model.Fluent:
        """Get a predicate (fluent) by name, raising a descriptive error if not found."""
        name = _normalize_name(pred_name)
        if name not in self._fluents:
            available_predicates = list(self._fluents.keys())
            raise ValueError(
                f"Predicate '{name}' does not exist{ctx}.\n"
                f"  Available predicates: {available_predicates}\n"
                f"  Suggestion: Check spelling carefully. Predicate names must match exactly "
                f"(lowercase, underscores for spaces)."
            )
        return self._fluents[name]
    
    def _build_params(self, params: List["ParameterDef"], ctx: str = "") -> Dict[str, up.model.Type]:
        return {
            _normalize_name(p.name.lstrip('?')): self._require_type(p.type, f" for {ctx} param '{p.name}'")
            for p in params
        }
    
    def _resolve_args(self, args: List[str], param_vars: Dict, ctx: str = "") -> List:
        """Resolve argument names to parameter variables, with descriptive error messages."""
        result = []
        for arg in args:
            key = _normalize_name(arg.lstrip('?'))
            if key not in param_vars:
                available_params = list(param_vars.keys())
                raise ValueError(
                    f"Parameter '{arg}' is not defined{ctx}.\n"
                    f"  Available parameters: {available_params}\n"
                    f"  Suggestion: All arguments must be parameters of the current action/method. "
                    # f"Use parameter names with '?' prefix (e.g., '?block', '?from')."
                )
            result.append(param_vars[key])
        return result
    
    def _build_condition(self, fact: "FactDef", param_vars: Dict, ctx: str = ""):
        fluent = self._require_fluent(fact.predicate, ctx)
        args = self._resolve_args(fact.args, param_vars, ctx)
        cond = fluent(*args)
        return Not(cond) if fact.negated else cond
    
    def _htn(self) -> HierarchicalProblem:
        """Get problem as HTN, or raise."""
        if not self.is_htn:
            raise ValueError("HTN operation not allowed on non-HTN problem")
        return self.problem  # type: ignore
    
    # Type Management
    
    def add_type(self, name: str, parent: Optional[str] = None) -> up.model.Type:
        """Add a type to the model. Validates that parent type exists if specified."""
        name = _normalize_name(name)
        
        if name in self._types:
            return self._types[name]
        
        if parent and _normalize_name(parent) != 'object':
            parent = _normalize_name(parent)
            if parent not in self._types:
                available_types = list(self._types.keys())
                raise ValueError(
                    f"Cannot create type '{name}' with parent '{parent}': parent type does not exist.\n"
                    f"  Available types: {available_types}\n"
                    f"  Suggestion: Define parent types BEFORE child types. "
                    f"Order your types so that '{parent}' appears before '{name}'."
                )
            up_type = self.env.type_manager.UserType(name, self._types[parent])
        else:
            up_type = self.env.type_manager.UserType(name)
        
        self._types[name] = up_type
        return up_type
    
    def get_type(self, name: str) -> up.model.Type:
        """Get a type by name."""
        return self._require_type(name)
    
    @property
    def type_names(self) -> List[str]:
        return list(self._types.keys())
    
    # Predicate Management
    
    def add_predicate(self, name: str, parameters: List["ParameterDef"]) -> up.model.Fluent:
        """Add a predicate to the model."""
        name = _normalize_name(name)
        if name in self._fluents:
            return self._fluents[name]
        
        signature = self._build_params(parameters, f"predicate '{name}'")
        fluent = up.model.Fluent(name, up.shortcuts.BoolType(), **signature)
        self.problem.add_fluent(fluent, default_initial_value=False)
        self._fluents[name] = fluent
        return fluent
    
    def get_predicate(self, name: str) -> up.model.Fluent:
        """Get a predicate by name."""
        return self._require_fluent(name)
    
    @property
    def predicate_names(self) -> List[str]:
        return list(self._fluents.keys())
    
    def get_predicate_signature(self, name: str) -> List[Tuple[str, str]]:
        fluent = self.get_predicate(name)
        return [(str(p.name), str(p.type.name)) for p in fluent.signature]
    
    # Action Management
    
    def add_action(
        self,
        name: str,
        parameters: List["ParameterDef"],
        preconditions: List["FactDef"],
        effects: List["FactDef"]
    ) -> up.model.InstantaneousAction:
        """Add an action to the model with full validation."""
        name = _normalize_name(name)
        ctx = f" in action '{name}'"
        
        action = up.model.InstantaneousAction(name, **self._build_params(parameters, f"action '{name}'"))
        param_vars = {_normalize_name(p.name.lstrip('?')): p for p in action.parameters}
        
        for fact in preconditions:
            up_expression = self._build_condition(fact, param_vars, ctx + " preconditions")
            action.add_precondition(up_expression)
        
        for fact in effects:
            fluent = self._require_fluent(fact.predicate, ctx + " effects")
            args = self._resolve_args(fact.args, param_vars, ctx)
            action.add_effect(fluent(*args), not fact.negated)
        
        self.problem.add_action(action)
        self._actions[name] = action
        return action
    
    def get_action(self, name: str) -> up.model.InstantaneousAction:
        """Get an action by name, raising a descriptive error if not found."""
        name = _normalize_name(name)
        if name not in self._actions:
            available_actions = list(self._actions.keys())
            raise ValueError(
                f"Action '{name}' does not exist.\n"
                f"  Available actions: {available_actions}\n"
                f"  Suggestion: Check spelling carefully. Action names must match exactly."
            )
        return self._actions[name]
    
    @property
    def action_names(self) -> List[str]:
        return list(self._actions.keys())
    
    def get_action_signature(self, name: str) -> List[Tuple[str, str]]:
        """Get the parameter signature of an action."""
        action = self.get_action(name)
        return [(str(p.name), str(p.type.name)) for p in action.parameters]
    
    # HTN Task Management
    
    def add_task(self, name: str, parameters: List["ParameterDef"]) -> Any:
        """Add an HTN task."""
        problem = self._htn()
        name = _normalize_name(name)
        
        if name in self._tasks:
            return self._tasks[name]
        
        task = problem.add_task(name, **self._build_params(parameters, f"task '{name}'"))
        self._tasks[name] = task
        return task
    
    def get_task(self, name: str):
        """Get an HTN task by name, raising a descriptive error if not found."""
        name = _normalize_name(name)
        if name not in self._tasks:
            available_tasks = list(self._tasks.keys())
            raise ValueError(
                f"HTN Task '{name}' does not exist.\n"
                f"  Available tasks: {available_tasks}\n"
                f"  Suggestion: Check spelling. Tasks are abstract goals, not primitive actions. "
                f"Make sure you're referring to a task, not an action."
            )
        return self._tasks[name]
    
    @property
    def task_names(self) -> List[str]:
        return list(self._tasks.keys())
    
    @property
    def method_names(self) -> Set[str]:
        return self._method_names
    
    def get_task_signature(self, name: str) -> List[Tuple[str, str]]:
        """Get the parameter signature of a task."""
        return [(str(p.name), str(p.type.name)) for p in self.get_task(name).parameters]
    
    # HTN Method Management
    
    def _subtask_ref(self, name: str):
        """Get a task or action by name for use as a subtask in a method."""
        n = _normalize_name(name)
        if n in self._tasks: 
            return self._tasks[n]
        if n in self._actions: 
            return self._actions[n]
        
        available_tasks = list(self._tasks.keys())
        available_actions = list(self._actions.keys())
        raise ValueError(
            f"Subtask '{name}' is neither a defined task nor an action.\n"
            f"  Available tasks: {available_tasks}\n"
            f"  Available actions: {available_actions}\n"
            f"  Suggestion: Subtasks in a method can be either:\n"
            f"    - Other HTN tasks (for hierarchical decomposition)\n"
            f"    - Primitive actions (for direct execution)\n"
            f"  Check spelling and ensure the task/action was defined earlier."
        )
    
    def add_method(
        self,
        name: str,
        task_name: str,
        task_args: List[str],
        parameters: List["ParameterDef"],
        preconditions: List["FactDef"],
        subtasks: List["SubtaskDef"]
    ):
        """Add an HTN method with full validation."""
        problem = self._htn()
        name = _normalize_name(name)
        task_name = _normalize_name(task_name)
        ctx = f" in method '{name}'"
        
        if task_name not in self._tasks:
            available_tasks = list(self._tasks.keys())
            raise ValueError(
                f"Method '{name}' references task '{task_name}' which does not exist.\n"
                f"  Available tasks: {available_tasks}\n"
                f"  Suggestion: The 'task' field of a method must be an existing HTN task. "
                f"Check spelling and ensure the task was defined in generate_tasks()."
            )
        
        # Crear el objeto Method según la API de Unified Planning
        method = Method(name, **self._build_params(parameters, f"method '{name}'"))
        param_vars = {_normalize_name(p.name.lstrip('?')): p for p in method.parameters}
        
        # Set target task
        method.set_task(self._tasks[task_name], *self._resolve_args(task_args, param_vars, ctx))
        
        # Add preconditions
        for fact in preconditions:
            up_expression = self._build_condition(fact, param_vars, ctx)
            method.add_precondition(up_expression)
        
        # Add subtasks
        for subtask in subtasks:
            ref = self._subtask_ref(subtask.name)
            method.add_subtask(ref, *self._resolve_args(subtask.args, param_vars, ctx))
        
        # Añadir el método completo al problema
        problem.add_method(method)
        
        self._methods.append(method)
        self._method_names.add(name)
        return method
    
    # Object Management
    
    def add_object(self, name: str, type_name: str) -> up.model.Object:
        name = _normalize_name(name)
        if name in self._objects:
            # Object already exists, return it. But show a warning:
            print(f"Warning: Object '{name}' already exists. Skipping re-definition.")
            return self._objects[name]
        obj = up.model.Object(name, self._require_type(type_name, f" for object '{name}'"))
        self.problem.add_object(obj)
        self._objects[name] = obj
        return obj
    
    def get_object(self, name: str) -> up.model.Object:
        """Get an object by name, raising a descriptive error if not found."""
        name = _normalize_name(name)
        if name not in self._objects:
            available_objects = list(self._objects.keys())
            raise ValueError(
                f"Object '{name}' does not exist in the problem.\n"
                f"  Available objects: {available_objects}\n"
                f"  Suggestion: Check spelling. Objects must be defined before they can be "
                f"used in the initial state, goal, or HTN goal tasks."
            )
        return self._objects[name]
    
    @property
    def object_names(self) -> List[str]:
        return list(self._objects.keys())
    
    def get_objects_by_type(self) -> Dict[str, List[str]]:
        """Get all objects grouped by their type name."""
        result: Dict[str, List[str]] = {}
        for obj_name, obj in self._objects.items():
            type_name = str(obj.type.name)
            if type_name not in result:
                result[type_name] = []
            result[type_name].append(obj_name)
        return result
    
    # Initial State & Goal
    
    def _resolve_objects(self, args: List[str], sig, ctx: str = "") -> List:
        """Resolve argument names to objects, validating count and types."""
        if len(args) != len(sig):
            expected_params = [(str(p.name), str(p.type.name)) for p in sig]
            raise ValueError(
                f"Wrong number of arguments{ctx}.\n"
                f"  Expected {len(sig)} arguments: {expected_params}\n"
                f"  Got {len(args)} arguments: {args}\n"
                f"  Suggestion: Check the predicate/task signature and provide exactly "
                f"the right number of arguments with matching types."
            )
        
        result = []
        for i, arg in enumerate(args):
            name = _normalize_name(arg.lstrip('?'))
            if name not in self._objects:
                available_objects = list(self._objects.keys())
                raise ValueError(
                    f"Object '{arg}' does not exist{ctx}.\n"
                    f"  Available objects: {available_objects}\n"
                    f"  Suggestion: Use only objects that were defined earlier. "
                    f"Check spelling (names are case-insensitive but must match)."
                )
            obj = self._objects[name]
            expected_type = sig[i].type
            if not obj.type.is_compatible(expected_type):
                raise ValueError(
                    f"Type mismatch for argument '{arg}'{ctx}.\n"
                    f"  Object '{arg}' has type '{obj.type.name}'\n"
                    f"  But position {i+1} expects type '{expected_type.name}'\n"
                    f"  Suggestion: Use an object of the correct type, or check if "
                    f"you have the arguments in the wrong order."
                )
            result.append(obj)
        return result
    
    def add_initial_fact(self, pred: str, args: List[str], value: bool = DEFAULT_INITIAL_FACT_VALUE):
        fluent = self._require_fluent(pred)
        self.problem.set_initial_value(fluent(*self._resolve_objects(args, fluent.signature, f" for '{pred}'")), value)
    
    def add_goal_fact(self, pred: str, args: List[str], negated: bool = DEFAULT_GOAL_FACT_NEGATED):
        fluent = self._require_fluent(pred)
        obj_args = self._resolve_objects(args, fluent.signature, f" for '{pred}'")
        cond = fluent(*obj_args)
        self.problem.add_goal(Not(cond) if negated else cond)
    
    def set_htn_goal(self, tasks: List[Tuple[str, List[str]]]):
        problem = self._htn()
        for task_name, args in tasks:
            task_name = _normalize_name(task_name)
            task = self.get_task(task_name)
            obj_args = self._resolve_objects(args, task.parameters, f" for task '{task_name}'")
            problem.task_network.add_subtask(task, *obj_args)
    
    # PDDL Export
    
    def to_pddl(self, domain_name: str = DEFAULT_DOMAIN_NAME, problem_name: str = DEFAULT_PROBLEM_NAME) -> Tuple[str, str]:
        """Export to PDDL (domain, problem) strings."""
        _ = domain_name  # Reserved for future use; PDDLWriter derives domain name
        self.problem.name = problem_name
        writer = PDDLWriter(self.problem)
        return writer.get_domain(), writer.get_problem()
    
    def validate(self) -> Tuple[bool, str]:
        """Validate the current model by attempting PDDL export."""
        try:
            self.to_pddl()
            return True, "Validation successful"
        except Exception as e:
            return False, str(e)


# =============================================================================
# PYDANTIC MODELS - Simple data structures for LLM output
# =============================================================================

class ProblemAnalysis(BaseModel):
    """
    Initial analysis of the task description to guide PDDL generation.
    
    This analysis extracts key information from natural language that will be used
    to generate the formal PDDL model. It identifies all entities, their types,
    the initial situation, what needs to be achieved, and what actions are needed.
    """
    description: str = Field(
        default="",
        description="IMPORTANT: Provide a clear analysis of the planning problem. This helps ensure the generated domain actually solves the right problem. Summarize: What is the goal? What constraints exist? What makes this problem interesting? Example: 'Package delivery: transport packages from sources to destinations, respecting vehicle capacity limits and driver availability windows.'",
        json_schema_extra={"importance": "high", "helps_validation": True}
    )
    entities: List[str] = Field(
        description="All concrete objects/entities mentioned or implied in the problem. "
                    "Use lowercase names with underscores. Examples: 'block_a', 'truck_1', "
                    "'city_paris', 'package_heavy', 'robot_arm'"
    )
    entity_types: Dict[str, str] = Field(
        description="Mapping of each entity name to its type category. Types should be "
                    "general categories that group similar entities. "
                    "Example: {'block_a': 'block', 'block_b': 'block', 'truck_1': 'vehicle', 'paris': 'city'}"
    )
    initial_conditions: List[str] = Field(
        description="Natural language descriptions of facts that are true at the start. "
                    "Examples: ['block_a is on block_b', 'truck_1 is at warehouse', 'robot hand is empty']"
    )
    goal_conditions: List[str] = Field(
        description="Natural language descriptions of what must be true at the end. "
                    "Examples: ['block_a is on the table', 'package delivered to destination', 'all blocks stacked']"
    )
    required_capabilities: List[str] = Field(
        description="Actions or capabilities needed to solve the problem. "
                    "Examples: ['pick up objects', 'put down objects', 'move between locations', 'stack blocks']"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [{
                "entities": ["block_a", "block_b", "block_c", "table"],
                "entity_types": {"block_a": "block", "block_b": "block", "block_c": "block", "table": "surface"},
                "initial_conditions": ["block_a is on block_b", "block_b is on the table", "block_c is on the table", "block_a is clear", "block_c is clear"],
                "goal_conditions": ["block_a is on block_b", "block_b is on block_c", "block_c is on the table"],
                "required_capabilities": ["pick up a block", "put down a block", "stack a block on another", "unstack a block"]
            }]
        }
    }
    
    @field_validator('entities', mode='before')
    @classmethod
    def validate_entities(cls, v):
        return [_normalize_name(e) for e in _validate_list(v, "Entity", True)]
    
    @field_validator('entity_types', mode='before')
    @classmethod
    def validate_entity_types(cls, v):
        if not isinstance(v, dict):
            raise ValueError("Entity types must be a dictionary")
        return {_normalize_name(k): _normalize_name(t) for k, t in v.items()}


class TypeDef(BaseModel):
    """
    Definition of a PDDL type.
    
    Types categorize objects in the domain. They can form a hierarchy where
    child types inherit from parent types (e.g., 'truck' could be a child of 'vehicle').
    """
    description: str = Field(
        description="Brief explanation of what this type represents in the domain. "
                    "Example: 'A movable block that can be stacked'"
    )
    name: str = Field(
        description="Type name in lowercase with underscores. "
                    "Examples: 'block', 'location', 'vehicle', 'heavy_package'"
    )
    parent: Optional[str] = Field(
        default=None, 
        description="Parent type name for hierarchical types, or null/None for root types. "
                    "Example: 'truck' might have parent 'vehicle'. Only set if there's a clear is-a relationship."
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {"name": "block", "parent": None, "description": "A stackable block that can be picked up and placed"},
                {"name": "location", "parent": None, "description": "A place where objects can be located"},
                {"name": "city", "parent": "location", "description": "A city, which is a type of location"}
            ]
        }
    }
    
    @field_validator('name', mode='before')
    @classmethod
    def validate_name(cls, v):
        return _normalize(v, "Name")
    
    @field_validator('parent', mode='before')
    @classmethod
    def validate_parent(cls, v):
        return _normalize(v, "Parent", allow_none=True)


class TypeList(BaseModel):
    """
    Collection of all types needed for the PDDL domain.
    
    Types should be ordered so that parent types appear before their children.
    """
    types: List[TypeDef] = Field(
        description="All types needed for the domain. Order matters: define parent types before children."
    )
    
    @field_validator('types', mode='before')
    @classmethod
    def validate_types(cls, v):
        v = _validate_list(v, "Type", True)
        _check_duplicate_names(v, 'name', 'type')
        return v

class ParameterDef(BaseModel):
    """
    Definition of a typed parameter for predicates, actions, tasks, or methods.
    
    Parameters are variables that get bound to specific objects when the
    predicate/action is used.
    """
    description: str = Field(
        default="",
        description="Optional description of what this parameter represents."
    )
    name: str = Field(
        description="Parameter name starting with '?'. Use descriptive names. "
                    "Examples: '?block', '?from_location', '?package', '?robot'"
    )
    type: str = Field(
        description="The type of this parameter. Must be a type defined in the domain. "
                    "Examples: 'block', 'location', 'vehicle'"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {"name": "?block", "type": "block"},
                {"name": "?from", "type": "location"},
                {"name": "?to", "type": "location"}
            ]
        }
    }
    
    @field_validator('name', mode='before')
    @classmethod
    def validate_name(cls, v):
        return _normalize(v, "Parameter name", as_param=True)
    
    @field_validator('type', mode='before')
    @classmethod
    def validate_type(cls, v):
        return _normalize(v, "Type")

class PredicateDef(BaseModel):
    """
    Definition of a PDDL predicate (relation or property).
    
    Predicates describe relationships between objects or properties of objects.
    They are used in preconditions and effects of actions.
    """
    description: str = Field(
        description="Clear explanation of what this predicate means when true. "
                    "Example: 'Block ?x is directly on top of block ?y'"
    )
    name: str = Field(
        description="Predicate name in lowercase with underscores. "
                    "Examples: 'on', 'at', 'holding', 'clear', 'connected_to'"
    )
    parameters: List[ParameterDef] = Field(
        description="List of typed parameters for this predicate. "
                    "Example: for 'on(?x, ?y)', parameters would be [(?x, block), (?y, block)]"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "name": "on",
                    "parameters": [{"name": "?upper", "type": "block"}, {"name": "?lower", "type": "block"}],
                    "description": "Block ?upper is directly on top of ?lower"
                },
                {
                    "name": "clear",
                    "parameters": [{"name": "?block", "type": "block"}],
                    "description": "Nothing is on top of ?block, so it can be picked up or have something placed on it"
                },
                {
                    "name": "holding",
                    "parameters": [{"name": "?obj", "type": "object"}],
                    "description": "The robot/agent is currently holding ?obj"
                }
            ]
        }
    }
    
    @field_validator('name', mode='before')
    @classmethod
    def validate_name(cls, v):
        return _normalize(v, "Predicate name")
    
    @field_validator('parameters', mode='before')
    @classmethod
    def validate_parameters(cls, v):
        v = _validate_list(v, "Parameters")
        if v:
            _check_duplicate_parameters(v, "predicate")
        return v


class PredicateList(BaseModel):
    """
    Collection of all predicates needed for the PDDL domain.
    
    Include predicates for:
    - Relationships between objects (e.g., 'on', 'at', 'connected')
    - Properties of objects (e.g., 'clear', 'empty', 'locked')
    - Agent states (e.g., 'holding', 'arm_empty')
    """
    predicates: List[PredicateDef] = Field(
        description="All predicates needed to describe states in this domain"
    )
    
    @field_validator('predicates', mode='before')
    @classmethod
    def validate_predicates(cls, v):
        v = _validate_list(v, "Predicate", True)
        _check_duplicate_names(v, 'name', 'predicate')
        return v

class FactDef(BaseModel):
    """
    A fact (predicate instance) used in preconditions, effects, initial state, or goals.
    
    Facts instantiate predicates with specific arguments. Arguments can be:
    - Parameters (like ?x) when used in action preconditions/effects
    - Objects (like block_a) when used in initial state or goals
    """
    description: str = Field(
        default="",
        description="Optional description of what this fact represents."
    )
    negated: bool = Field(
        default=False,
        description="True if this fact should be negated (NOT). "
                    "In preconditions: requires the fact to be FALSE. "
                    "In effects: makes the fact FALSE (delete effect). "
                    "Example: negated=True for 'clear(?block)' means 'block is NOT clear'"
    )
    predicate: str = Field(
        description="Name of the predicate being instantiated. Must match a defined predicate exactly. "
                    "Examples: 'on', 'clear', 'holding', 'at'"
    )
    args: List[str] = Field(
        description="Arguments for the predicate. Use ?param_name for parameters (in actions/methods) "
                    "or object names (in initial state/goals). "
                    "Examples: ['?block', '?surface'] or ['block_a', 'table']"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {"predicate": "on", "args": ["?block", "?surface"], "negated": False},
                {"predicate": "clear", "args": ["?block"], "negated": True},
                {"predicate": "holding", "args": ["block_a"], "negated": False}
            ]
        }
    }
    
    @field_validator('predicate', mode='before')
    @classmethod
    def validate_predicate(cls, v):
        return _normalize(v, "Predicate name")
    
    @field_validator('args', mode='before')
    @classmethod
    def validate_args(cls, v):
        """Normalize args, preserving ? prefix for parameters vs objects."""
        if not isinstance(v, list):
            raise ValueError("Arguments must be a list")
        result = []
        for arg in v:
            is_param = isinstance(arg, str) and arg.startswith('?')
            normalized = _normalize(arg, "Argument", as_param=is_param)
            result.append(normalized)
        return result

class ActionDef(BaseModel):
    """
    Definition of a PDDL action (operator).
    
    Actions are the primitive operations that change the world state.
    Each action has parameters, preconditions (what must be true to execute),
    and effects (what changes when executed).
    """
    description: str = Field(
        default="",
        description="Brief description of what this action does."
    )
    name: str = Field(
        description="Action name in lowercase with underscores. Use verb phrases. "
                    "Examples: 'pick_up', 'put_down', 'move', 'stack', 'drive_to'"
    )
    parameters: List[ParameterDef] = Field(
        description="Typed parameters for this action. These are the variables used in preconditions and effects. "
                    "Example: [(?block, block), (?from, location), (?to, location)]"
    )
    preconditions: List[FactDef] = Field(
        description="Facts that must be true BEFORE the action can execute. "
                    "All arguments must be action parameters. "
                    "Example: to pick up a block, it must be clear and the hand must be empty"
    )
    effects: List[FactDef] = Field(
        description="Facts that change AFTER the action executes. "
                    "negated=False means the fact becomes TRUE (add effect). "
                    "negated=True means the fact becomes FALSE (delete effect). "
                    "Example: after picking up, holding becomes true, clear becomes false"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [{
                "name": "pick_up",
                "parameters": [{"name": "?b", "type": "block"}],
                "preconditions": [
                    {"predicate": "clear", "args": ["?b"], "negated": False},
                    {"predicate": "on_table", "args": ["?b"], "negated": False},
                    {"predicate": "arm_empty", "args": [], "negated": False}
                ],
                "effects": [
                    {"predicate": "holding", "args": ["?b"], "negated": False},
                    {"predicate": "clear", "args": ["?b"], "negated": True},
                    {"predicate": "on_table", "args": ["?b"], "negated": True},
                    {"predicate": "arm_empty", "args": [], "negated": True}
                ]
            }]
        }
    }
    
    @field_validator('name', mode='before')
    @classmethod
    def validate_name(cls, v):
        return _normalize(v, "Action name")
    
    @field_validator('preconditions', 'effects', mode='before')
    @classmethod
    def validate_conditions(cls, v):
        return _validate_list(v, "Conditions")
    
    @model_validator(mode='after')
    def validate_action_consistency(self):
        """Validate that all fact arguments reference defined parameters."""
        # Get parameter names (normalized, without ?)
        param_names = {_normalize_name(p.name.lstrip('?')) for p in self.parameters}
        
        # Check preconditions
        for fact in self.preconditions:
            _validate_fact_args_are_parameters(
                {'predicate': fact.predicate, 'args': fact.args},
                param_names,
                f"action '{self.name}' preconditions"
            )
        
        # Check effects
        for fact in self.effects:
            _validate_fact_args_are_parameters(
                {'predicate': fact.predicate, 'args': fact.args},
                param_names,
                f"action '{self.name}' effects"
            )
        
        # Check for duplicate parameters
        _check_duplicate_parameters(
            [{'name': p.name} for p in self.parameters], 
            f"action '{self.name}'"
        )
        
        return self


class ActionList(BaseModel):
    """
    Collection of all actions (operators) for the PDDL domain.
    
    Actions should cover all the capabilities needed to transform
    the initial state into the goal state.
    """
    actions: List[ActionDef] = Field(
        description="All primitive actions that agents can perform in this domain"
    )
    
    @field_validator('actions', mode='before')
    @classmethod
    def validate_actions(cls, v):
        v = _validate_list(v, "Action", True)
        _check_duplicate_names(v, 'name', 'action')
        return v

class TaskDef(BaseModel):
    """
    Definition of an HTN compound task.
    
    Tasks are abstract goals that need to be decomposed into subtasks or primitive
    actions via methods. Unlike actions, tasks don't directly change the world -
    they represent WHAT needs to be done, not HOW to do it.
    """
    description: str = Field(
        default="",
        description="Optional description of what this task accomplishes."
    )
    name: str = Field(
        description="Task name in lowercase with underscores. Use goal-oriented names. "
                    "Examples: 'deliver_package', 'build_tower', 'transport_person', 'achieve_goal'"
    )
    parameters: List[ParameterDef] = Field(
        description="Typed parameters for this task. These define what the task operates on. "
                    "Example: deliver_package(?pkg, ?dest) has parameters for package and destination"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "name": "deliver_package",
                    "parameters": [{"name": "?pkg", "type": "package"}, {"name": "?dest", "type": "location"}]
                },
                {
                    "name": "build_tower",
                    "parameters": [{"name": "?base", "type": "block"}, {"name": "?top", "type": "block"}]
                }
            ]
        }
    }
    
    _name = field_validator('name', mode='before')(lambda cls, v: _normalize(v, "Task name"))
    
    @field_validator('parameters', mode='before')
    @classmethod
    def validate_params(cls, v):
        v = _validate_list(v, "Parameters")
        if v:
            _check_duplicate_parameters(v, "task")
        return v


class TaskList(BaseModel):
    """
    Collection of all HTN compound tasks for the domain.
    
    Tasks should represent high-level goals that can be achieved through
    different methods (decomposition strategies).
    """
    tasks: List[TaskDef] = Field(
        description="All compound tasks that can be decomposed by methods. "
                    "Do NOT include primitive actions here - those are separate."
    )
    
    @field_validator('tasks', mode='before')
    @classmethod
    def validate_tasks(cls, v):
        v = _validate_list(v, "Task", True)
        _check_duplicate_names(v, 'name', 'task')
        return v

class SubtaskDef(BaseModel):
    """
    Reference to a subtask within a method's decomposition.
    
    Subtasks can be either primitive actions or other compound tasks,
    enabling hierarchical decomposition.
    """
    description: str = Field(
        default="",
        description="Optional description of this subtask reference."
    )
    name: str = Field(
        description="Name of the task or action to execute. Must be a defined task or action. "
                    "Examples: 'pick_up', 'deliver_package', 'move_to'"
    )
    args: List[str] = Field(
        description="Arguments to pass to the subtask. Must be parameters of the parent method. "
                    "Use ?param_name format. Examples: ['?block', '?destination']"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {"name": "pick_up", "args": ["?package"]},
                {"name": "drive_to", "args": ["?truck", "?destination"]},
                {"name": "deliver_package", "args": ["?pkg", "?loc"]}
            ]
        }
    }
    
    _name = field_validator('name', mode='before')(lambda cls, v: _normalize(v, "Subtask name"))
    _args = field_validator('args', mode='before')(lambda cls, v: _normalize_args(v, as_params=True))

class MethodDef(BaseModel):
    """
    Definition of an HTN method for decomposing a task.
    
    Methods define HOW to accomplish a task by specifying:
    - Which task they decompose
    - Under what conditions (preconditions)
    - What sequence of subtasks to execute
    
    Multiple methods can decompose the same task, providing alternative strategies.
    """
    description: str = Field(
        default="",
        description="Optional description of this decomposition method."
    )
    name: str = Field(
        description="Unique method name in lowercase with underscores. "
                    "Examples: 'deliver_by_truck', 'deliver_by_air', 'build_tower_recursive'"
    )
    task: str = Field(
        description="Name of the task this method decomposes. Must be a defined task. "
                    "Example: 'deliver_package'"
    )
    task_args: List[str] = Field(
        description="Arguments to match with the task's parameters. Must use method parameters. "
                    "The count and types must match the task's signature. "
                    "Example: if task is deliver_package(?pkg, ?dest), task_args could be ['?package', '?destination']"
    )
    parameters: List[ParameterDef] = Field(
        description="Method's parameters. Must include all parameters needed for task_args, "
                    "plus any additional parameters used in preconditions or subtasks. "
                    "Example: [(?package, package), (?destination, location), (?truck, vehicle)]"
    )
    preconditions: List[FactDef] = Field(
        description="Conditions that must be true for this method to apply. "
                    "Use to distinguish between different methods for the same task. "
                    "Example: 'truck_available(?truck)' for deliver_by_truck method"
    )
    subtasks: List[SubtaskDef] = Field(
        description="Ordered sequence of subtasks to execute. Can be actions or other tasks. "
                    "Order matters - they execute in sequence. "
                    "Example: [load(?pkg, ?truck), drive(?truck, ?dest), unload(?pkg, ?truck)]"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [{
                "name": "deliver_by_truck",
                "task": "deliver_package",
                "task_args": ["?pkg", "?dest"],
                "parameters": [
                    {"name": "?pkg", "type": "package"},
                    {"name": "?dest", "type": "location"},
                    {"name": "?truck", "type": "truck"},
                    {"name": "?src", "type": "location"}
                ],
                "preconditions": [
                    {"predicate": "at", "args": ["?pkg", "?src"], "negated": False},
                    {"predicate": "truck_available", "args": ["?truck"], "negated": False}
                ],
                "subtasks": [
                    {"name": "drive_to", "args": ["?truck", "?src"]},
                    {"name": "load", "args": ["?pkg", "?truck"]},
                    {"name": "drive_to", "args": ["?truck", "?dest"]},
                    {"name": "unload", "args": ["?pkg", "?truck"]}
                ]
            }]
        }
    }
    
    _names = field_validator('name', 'task', mode='before')(lambda cls, v: _normalize(v, "Method/task name"))
    _task_args = field_validator('task_args', mode='before')(lambda cls, v: _normalize_args(v, as_params=True))
    _subtasks = field_validator('subtasks', mode='before')(lambda cls, v: _validate_list(v, "Subtask", True))
    
    @model_validator(mode='after')
    def validate_method_consistency(self):
        """Validate that all arguments reference defined parameters."""
        # Get parameter names (normalized, without ?)
        param_names = {_normalize_name(p.name.lstrip('?')) for p in self.parameters}
        
        # Check task_args are parameters
        for arg in self.task_args:
            normalized = _normalize_name(arg.lstrip('?'))
            if normalized not in param_names:
                raise ValueError(
                    f"In method '{self.name}', task_arg '{arg}' is not a method parameter.\n"
                    f"Available parameters: {['?' + p for p in sorted(param_names)]}"
                )
        
        # Check preconditions
        for fact in self.preconditions:
            _validate_fact_args_are_parameters(
                {'predicate': fact.predicate, 'args': fact.args},
                param_names,
                f"method '{self.name}' preconditions"
            )
        
        # Check subtask arguments
        for subtask in self.subtasks:
            for arg in subtask.args:
                normalized = _normalize_name(arg.lstrip('?'))
                if normalized not in param_names:
                    raise ValueError(
                        f"In method '{self.name}', subtask '{subtask.name}' uses undefined parameter '{arg}'.\n"
                        f"Available parameters: {['?' + p for p in sorted(param_names)]}"
                    )
        
        # Check for duplicate parameters
        _check_duplicate_parameters(
            [{'name': p.name} for p in self.parameters], 
            f"method '{self.name}'"
        )
        
        return self


class MethodList(BaseModel):
    """
    Collection of methods for decomposing a specific task.
    
    Include multiple methods if there are different ways to accomplish the task
    under different circumstances.
    """
    methods: List[MethodDef] = Field(
        description="Methods that decompose the target task. Each method provides a different "
                    "strategy for accomplishing the task."
    )
    
    @field_validator('methods', mode='before')
    @classmethod
    def validate_methods(cls, v):
        v = _validate_list(v, "Method", True)
        _check_duplicate_names(v, 'name', 'method')
        return v

class ObjectDef(BaseModel):
    """
    Definition of a problem object (instance of a type).
    
    Objects are the concrete entities that exist in the specific problem instance.
    Each object has a name and belongs to a type defined in the domain.
    """
    description: str = Field(
        default="",
        description="Optional description of this problem object."
    )
    name: str = Field(
        description="Object name in lowercase, optionally with numbers. "
                    "Examples: 'block_a', 'truck_1', 'paris', 'package_heavy'"
    )
    type: str = Field(
        description="Type of this object. Must be a type defined in the domain. "
                    "Examples: 'block', 'vehicle', 'city', 'package'"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {"name": "block_a", "type": "block"},
                {"name": "truck_1", "type": "vehicle"},
                {"name": "warehouse", "type": "location"}
            ]
        }
    }
    
    _name = field_validator('name', mode='before')(lambda cls, v: _normalize(v, "Object name"))
    _type = field_validator('type', mode='before')(lambda cls, v: _normalize(v, "Object type"))


class ObjectList(BaseModel):
    """
    Collection of all objects in the problem instance.
    
    Include every entity mentioned or implied in the problem description.
    """
    objects: List[ObjectDef] = Field(
        description="All concrete objects that exist in this problem. "
                    "Make sure to include every entity from the problem description."
    )
    
    @field_validator('objects', mode='before')
    @classmethod
    def validate_objects(cls, v):
        v = _validate_list(v, "Object", True)
        _check_duplicate_names(v, 'name', 'object')
        return v

class InitialState(BaseModel):
    """
    The initial state of the problem (facts true at the start).
    
    Uses closed-world assumption: only list facts that are TRUE.
    Any fact not listed is assumed to be FALSE.
    Do NOT include negated facts.
    """
    facts: List[FactDef] = Field(
        description="All facts that are TRUE in the initial state. "
                    "Arguments must be object names (not parameters). "
                    "Do NOT include negated facts - only positive facts. "
                    "Example: (on block_a block_b), (clear block_a), (arm_empty)"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [{
                "facts": [
                    {"predicate": "on", "args": ["block_a", "block_b"], "negated": False},
                    {"predicate": "on_table", "args": ["block_b"], "negated": False},
                    {"predicate": "clear", "args": ["block_a"], "negated": False},
                    {"predicate": "arm_empty", "args": [], "negated": False}
                ]
            }]
        }
    }
    
    _facts = field_validator('facts', mode='before')(lambda cls, v: _validate_list(v, "Initial fact", True))


class GoalState(BaseModel):
    """
    The goal state of the problem (what must be true at the end).
    
    Unlike initial state, goals CAN include negated facts if you need
    to specify that something must NOT be true.
    """
    facts: List[FactDef] = Field(
        description="Facts that must be true (or false if negated) in the goal state. "
                    "Arguments must be object names. "
                    "Use negated=True if a fact must be FALSE in the goal. "
                    "Example: (on block_a block_b), (NOT (on block_c table))"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [{
                "facts": [
                    {"predicate": "on", "args": ["block_a", "block_b"], "negated": False},
                    {"predicate": "on", "args": ["block_b", "block_c"], "negated": False},
                    {"predicate": "on_table", "args": ["block_c"], "negated": False}
                ]
            }]
        }
    }
    
    _facts = field_validator('facts', mode='before')(lambda cls, v: _validate_list(v, "Goal fact", True))


class HTNGoal(BaseModel):
    """
    HTN goal specification (tasks to accomplish).
    
    Unlike classical PDDL goals (which specify a desired state), HTN goals
    specify which high-level tasks need to be accomplished.
    """
    tasks: List[SubtaskDef] = Field(
        description="High-level tasks that must be accomplished to solve the problem. "
                    "Arguments must be object names (not parameters). "
                    "Example: deliver_package(package_1, warehouse_b)"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [{
                "tasks": [
                    {"name": "deliver_package", "args": ["package_1", "city_b"]},
                    {"name": "deliver_package", "args": ["package_2", "city_c"]}
                ]
            }]
        }
    }
    
    _tasks = field_validator('tasks', mode='before')(lambda cls, v: _validate_list(v, "Task", True))


# =============================================================================
# NAME LISTS - Simple lists of names for sequential generation
# =============================================================================

class ActionNameList(BaseModel):
    """List of action names to generate (step 1 of sequential generation)."""
    names: List[str] = Field(
        description="List of action names needed for this domain. "
                    "Use lowercase with underscores. "
                    "Examples: ['pick_up', 'put_down', 'stack', 'unstack']"
    )
    
    _names = field_validator('names', mode='before')(lambda cls, v: _validate_list(v, "Action name", True))


class MethodNameList(BaseModel):
    """List of method names to generate for a specific task (step 1 of sequential generation)."""
    names: List[str] = Field(
        description="List of method names that decompose this task. "
                    "Use lowercase with underscores. "
                    "Examples: ['deliver_by_truck', 'deliver_by_drone']"
    )
    
    _names = field_validator('names', mode='before')(lambda cls, v: _validate_list(v, "Method name", True))



# =============================================================================
# PDDL MODELER - Orchestrates LLM and validation
# =============================================================================

class PDDLModeler:
    """
    Framework for LLM-based PDDL/HTN model generation with validation.
    
    This class provides a modular API where each generation step is exposed
    publicly, allowing users to build custom workflows. The class handles:
    
    1. LLM interaction for generating model components
    2. Validation through Unified Planning library
    3. Error handling with retry logic
    4. Metrics tracking and logging
    
    ═══════════════════════════════════════════════════════════════════════════
    FRAMEWORK DESIGN - Two Ways to Use This Class
    ═══════════════════════════════════════════════════════════════════════════
    
    1. SIMPLE USE - Call generate_full_model() for automatic generation:
    
        modeler = PDDLModeler(client, is_htn=False)
        modeler.generate_full_model("Stack blocks A on B on C")
        domain, problem = modeler.get_pddl()
    
    2. CUSTOM WORKFLOW - Use individual methods for fine-grained control:
    
        modeler = PDDLModeler(client, is_htn=True)
        
        # Phase 0: Analysis (optional)
        modeler.analyze_problem(task_desc)
        
        # Phase 1: Domain structure
        modeler.generate_types(task_desc)
        modeler.generate_predicates(task_desc)
        modeler.generate_objects(task_desc)
        
        # Phase 2: Actions - with full control
        action_names = modeler.generate_action_names(task_desc)
        action_names = [n for n in action_names if 'unwanted' not in n]  # Filter
        for name in action_names:
            action = modeler.generate_action(task_desc, name)
            if not self._is_good_action(action):  # Custom validation
                action = modeler.generate_action(task_desc, name)  # Retry
        
        # Phase 3: HTN (if is_htn=True)
        modeler.generate_tasks(task_desc)
        for task_name in modeler.builder.task_names:
            method_names = modeler.generate_method_names(task_desc, task_name)
            for method_name in method_names:
                modeler.generate_method(task_desc, task_name, method_name)
        
        # Phase 4: Problem instance
        modeler.generate_initial_state(task_desc)
        modeler.generate_goal(task_desc)
    
    ═══════════════════════════════════════════════════════════════════════════
    PUBLIC API METHODS
    ═══════════════════════════════════════════════════════════════════════════
    
    Analysis:
        - analyze_problem(task_desc) → ProblemAnalysis
    
    Domain Structure:
        - generate_types(task_desc) → TypeList
        - generate_predicates(task_desc) → PredicateList
        - generate_objects(task_desc) → ObjectList
    
    Actions (two-step API for control):
        - generate_action_names(task_desc) → List[str]  # Step 1: Get names
        - generate_action(task_desc, name) → ActionDef  # Step 2: Generate one
        - generate_actions(task_desc, names=None) → ActionList  # Convenience
    
    HTN Tasks:
        - generate_tasks(task_desc) → TaskList
    
    Methods (two-step API for control):
        - generate_method_names(task_desc, task_name) → List[str]  # Step 1
        - generate_method(task_desc, task_name, name) → MethodDef  # Step 2
        - generate_methods(task_desc, task_name, names=None) → MethodList  # Convenience
    
    Problem Instance:
        - generate_initial_state(task_desc) → InitialState
        - generate_goal(task_desc) → GoalState | HTNGoal
    
    Output:
        - get_pddl(domain_name, problem_name) → Tuple[str, str]
        - to_dict() → dict
        - from_dict(data) → PDDLModeler  # class method
    
    Inspection:
        - builder: PDDLBuilder instance with validated model
        - raw_*: Raw LLM outputs (raw_types, raw_actions, raw_methods, etc.)
        - metrics: GenerationMetrics for performance tracking
    
    ═══════════════════════════════════════════════════════════════════════════
    
    See generate_full_model() for a complete example workflow.
    """
    
    def __init__(
        self, 
        client: StructuredOutputClient, 
        is_htn: bool = DEFAULT_IS_HTN, 
        max_retries: int = DEFAULT_MAX_RETRIES,
        use_analysis: bool = DEFAULT_USE_ANALYSIS,
        log_level: Optional[LogLevel] = LogLevel[DEFAULT_LOG_LEVEL.upper()],
        interactive: bool = DEFAULT_INTERACTIVE_MODE,
        log_file: Optional[str] = DEFAULT_LOG_FILE,
        track_metrics: bool = DEFAULT_TRACK_METRICS,
        think_mode: bool = DEFAULT_THINK_MODE
    ):
        """
        Initialize the PDDLModeler.
        
        Args:
            client: Structured output client for LLM interaction
            is_htn: Whether to generate HTN (hierarchical) models
            max_retries: Maximum LLM call attempts per component
            use_analysis: Whether to analyze problem before generation
            log_level: Verbosity level (0/QUIET, 1/INFO, 2/DEBUG, 3/TRACE)
            interactive: If True, pause at each step for inspection
            log_file: Optional path to save complete execution log
            track_metrics: If True, track generation metrics (default True)
            think_mode: If True, use two-step reasoning (think → JSON) instead of direct generation
        """
        self.client = client
        self.is_htn = is_htn
        self.max_retries = max_retries
        self.use_analysis = use_analysis
        self.think_mode = think_mode
        self.builder = PDDLBuilder(is_htn=is_htn)
        
        # Logging configuration - convert to LogLevel if needed (accepts int, str, or LogLevel)
        if isinstance(log_level, str):
            log_level = LogLevel[log_level.upper()]
        elif isinstance(log_level, int):
            log_level = LogLevel(log_level)
        self.log_level = log_level
        self.interactive = interactive
        self.log_file = log_file
        self.track_metrics = track_metrics
        self._log_buffer: List[str] = []  # Buffer for log file
        
        # Configure logging based on level
        if self.log_level.value >= LogLevel.DEBUG.value:
            logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')
            logger.setLevel(logging.DEBUG)
        
        # Raw LLM outputs
        self.raw_analysis: Optional[ProblemAnalysis] = None
        self.raw_types: Optional[TypeList] = None
        self.raw_predicates: Optional[PredicateList] = None
        self.raw_actions: Optional[ActionList] = None
        self.raw_tasks: Optional[TaskList] = None
        self.raw_methods: List[MethodDef] = []
        self.raw_objects: Optional[ObjectList] = None
        self.raw_initial: Optional[InitialState] = None
        self.raw_goal: Optional[Union[GoalState, HTNGoal]] = None
        
        # Generation metrics tracking
        self.metrics = GenerationMetrics()
        
        # Configuration summary
        self._log(LogLevel.INFO,     "╔══════════════════════════════════════════════════════════════╗")
        self._log(LogLevel.INFO,     "║                MODELING FRAMEWORK INITIALIZED:               ║")
        self._log(LogLevel.INFO,     "║                                                              ║")
        self._log(LogLevel.INFO,    f"║  - MODEL TYPE: {'Hierarchical (HTN)' if self.is_htn else 'Classical (PDDL)'}                            ║")
        self._log(LogLevel.INFO,     "║                                                              ║")
        self._log(LogLevel.INFO,    f"║  - BACKEND LLM: {type(self.client).__name__}                 ║")
        self._log(LogLevel.INFO,     "║                                                              ║")
        self._log(LogLevel.INFO,    f"║  - LOG LEVEL: {self.log_level.name}                          ║")
        if self.track_metrics:
            self._log(LogLevel.INFO, "║                                                              ║")
            self._log(LogLevel.INFO, "║  - METRICS TRACKING ENABLED                                  ║")
            self._log(LogLevel.INFO, "║  Performance data will be collected and reported at the end  ║")

        if self.interactive:
            self._log(LogLevel.INFO, "║                                                              ║")
            self._log(LogLevel.INFO, "║  - INTERACTIVE MODE ENABLED                                  ║")
            self._log(LogLevel.INFO, "║  At each step you can:                                       ║")
            self._log(LogLevel.INFO, "║  [Enter] Continue to next step                               ║")
            self._log(LogLevel.INFO, "║  [s]     Show current model state                            ║")
            self._log(LogLevel.INFO, "║  [p]     Show last prompt sent to LLM                        ║")
            self._log(LogLevel.INFO, "║  [r]     Show last LLM response                              ║")
            self._log(LogLevel.INFO, "║  [q]     Quit/abort execution                                ║")
        
        self._log(LogLevel.INFO,     "╚══════════════════════════════════════════════════════════════╝")
        

            
    
    # -------------------------------------------------------------------------
    # Logging System
    # -------------------------------------------------------------------------
    
    def _log(self, level: LogLevel, message: str, end: str = DEFAULT_LOG_MESSAGE_END) -> None:
        """
        Log a message if the current log level allows it.
        
        Args:
            level: Minimum level required to show this message
            message: The message to log
            end: Line ending (default newline)
        """
        if self.log_level.value >= level.value:
            print(message, end=end, flush=True)
        
        # Always save to buffer for log file
        self._log_buffer.append(message)
    
    def _log_section(self, title: str) -> None:
        """Log a section header."""
        separator = "═" * 70
        self._log(LogLevel.DEBUG, f"\n{separator}")
        self._log(LogLevel.DEBUG, f"  {title}")
        self._log(LogLevel.DEBUG, separator)
    
    def _log_prompt(self, prompt: str, step_name: str) -> None:
        """Log a prompt at TRACE level."""
        if self.log_level.value >= LogLevel.TRACE.value:
            self._log(LogLevel.TRACE, f"\n{'─'*40} PROMPT: '{step_name}' {'─'*40}")
            truncated = prompt[:DEFAULT_PROMPT_LOG_LIMIT] + DEFAULT_PROMPT_DISPLAY_SUFFIX if len(prompt) > DEFAULT_PROMPT_LOG_LIMIT else prompt
            self._log(LogLevel.TRACE, truncated)
            self._log(LogLevel.TRACE, "─" * 90)
    
    def _log_response(self, response: Any, step_name: str) -> None:
        """Log a response at TRACE level."""
        if self.log_level.value >= LogLevel.TRACE.value:
            response_str = response.model_dump_json(indent=2) if hasattr(response, 'model_dump_json') else str(response)
            self._log(LogLevel.TRACE, f"\n{'─'*40} RESPONSE: '{step_name}' {'─'*40}")
            truncated = response_str[:DEFAULT_RESPONSE_LOG_LIMIT] + DEFAULT_PROMPT_DISPLAY_SUFFIX if len(response_str) > DEFAULT_RESPONSE_LOG_LIMIT else response_str
            self._log(LogLevel.TRACE, truncated)
            self._log(LogLevel.TRACE, "─" * 90)
    
    def _save_log_file(self) -> None:
        """Save accumulated log to file if log_file is set."""
        if self.log_file and self._log_buffer:
            try:
                with open(self.log_file, 'w', encoding='utf-8') as f:
                    f.write("\n".join(self._log_buffer))
                print(f"\n📄 Log saved to: {self.log_file}")
            except Exception as e:
                print(f"\n⚠ Failed to save log file: {e}")
    
    def _get_model_state(self) -> str:
        """Get a formatted string of the current model state."""
        lines = [
            "┌─────────────────────────────────────────────────────────────────┐",
            "│  CURRENT MODEL STATE                                            │",
            "├─────────────────────────────────────────────────────────────────┤",
        ]
        
        def fmt_set(s: set, max_items: int = DEFAULT_MODEL_STATE_PREVIEW_ITEMS) -> str:
            # If max_items <= 0, return all items
            if max_items < 0:
                return str(list(s))
            items = list(s)[:max_items]
            suffix = '...' if len(s) > max_items else ''
            return f"{items}{suffix}"
        
        lines.append(f"│  Types ({len(self.builder.type_names)}): {fmt_set(self.builder.type_names)}")
        lines.append(f"│  Predicates ({len(self.builder.predicate_names)}): {fmt_set(self.builder.predicate_names)}")
        lines.append(f"│  Actions ({len(self.builder.action_names)}): {fmt_set(self.builder.action_names)}")
        lines.append(f"│  Objects ({len(self.builder.object_names)}): {fmt_set(self.builder.object_names)}")
        
        if self.is_htn:
            lines.append(f"│  Tasks ({len(self.builder.task_names)}): {fmt_set(self.builder.task_names)}")
            lines.append(f"│  Methods ({len(self.builder.method_names)}): {fmt_set(self.builder.method_names)}")
        
        lines.append("└─────────────────────────────────────────────────────────────────┘")
        return "\n".join(lines)
    
    # -------------------------------------------------------------------------
    # Public Properties - Easy access to model state for framework users
    # -------------------------------------------------------------------------
    
    @property
    def types(self) -> Set[str]:
        """Get set of all type names in the model."""
        return self.builder.type_names
    
    @property
    def predicates(self) -> Set[str]:
        """Get set of all predicate names in the model."""
        return self.builder.predicate_names
    
    @property
    def actions(self) -> Set[str]:
        """Get set of all action names in the model."""
        return self.builder.action_names
    
    @property
    def objects(self) -> Set[str]:
        """Get set of all object names in the model."""
        return self.builder.object_names
    
    @property
    def tasks(self) -> Set[str]:
        """Get set of all task names (HTN only)."""
        return self.builder.task_names
    
    @property
    def methods(self) -> Set[str]:
        """Get set of all method names (HTN only)."""
        return self.builder.method_names
    
    def get_action_def(self, name: str) -> Optional[ActionDef]:
        """
        Get the raw ActionDef for a specific action by name.
        
        Args:
            name: Name of the action
            
        Returns:
            ActionDef if found, None otherwise
        """
        if self.raw_actions:
            for action in self.raw_actions.actions:
                if _normalize_name(action.name) == _normalize_name(name):
                    return action
        return None
    
    def get_method_def(self, name: str) -> Optional[MethodDef]:
        """
        Get the raw MethodDef for a specific method by name.
        
        Args:
            name: Name of the method
            
        Returns:
            MethodDef if found, None otherwise
        """
        for method in self.raw_methods:
            if _normalize_name(method.name) == _normalize_name(name):
                return method
        return None
    
    def reset(self) -> None:
        """
        Reset the modeler to initial state, clearing all generated content.
        
        Useful for starting a new generation without creating a new instance.
        """
        self.builder = PDDLBuilder(is_htn=self.is_htn)
        self.raw_analysis = None
        self.raw_types = None
        self.raw_predicates = None
        self.raw_actions = None
        self.raw_tasks = None
        self.raw_methods = []
        self.raw_objects = None
        self.raw_initial = None
        self.raw_goal = None
        self.metrics = GenerationMetrics()
        self._log_buffer = []
    
    # -------------------------------------------------------------------------
    # Interactive Mode
    # -------------------------------------------------------------------------
    
    def _interactive_inspection(
        self, 
        step_name: str, 
        error_history: List[str] = [],
        error_messages: List[str] = [],
        last_prompt: Optional[str] = None, 
        last_response: Optional[Any] = None,
        thinking_prompt: Optional[str] = None,
        thinking_text: Optional[str] = None,
        traceback_str: Optional[str] = None,
        is_fatal: bool = False
    ) -> None:
        """
        Interactive inspection for both successful steps and errors.
        
        Args:
            step_name: Name of the step (completed or failed)
            error_history: List of error entries with full context (if any retries occurred)
            error_messages: List of error message summaries (if any retries occurred)
            last_prompt: The final JSON prompt sent to LLM
            last_response: The last LLM response (if available)
            thinking_prompt: The thinking prompt used (if think_mode enabled)
            thinking_text: The thinking text generated (if think_mode enabled)
            traceback_str: Full Python traceback (only for fatal errors)
            is_fatal: If True, force inspection regardless of interactive mode (only when aborting)
        """
        # If not fatal and not in interactive mode, don't pause
        if not self.interactive: # TO DO: Añadir not is_fatal. Quiero 3 modos de ejecución: normal, interactivo, y sólo en errores fatales
            return
        
        if is_fatal:
            self._log(LogLevel.INFO, f"\n❌ Step '{step_name}' FAILED after {len(error_history)} attempts (ABORTING).")
        else:
            self._log(LogLevel.INFO, f"\n✅ Step '{step_name}' completed successfully with {len(error_history) + 1} attempt(s).")
        
        help_text = "[Enter=continue, e=errors, p=prompt, r=response, u=thinking_prompt, t=thinking_text, s=state, b=traceback]: "
        
        while True:
            try:
                user_input = input(help_text).strip().lower()
            except EOFError:
                # Non-interactive environment
                return
            
            if user_input == "" or user_input == "c":
                return  # Continue
            elif user_input == "s":
                print(self._get_model_state())
            elif user_input == "p":
                if last_prompt:
                    print(f"\n{'─'*40} LAST JSON PROMPT {'─'*40}")
                    print(last_prompt)
                    print("─" * 90)
                else:
                    print("  No JSON prompt available for this step")
            elif user_input == "r":
                if last_response:
                    print(f"\n{'─'*40} LAST RESPONSE {'─'*40}")
                    if hasattr(last_response, 'model_dump_json'):
                        print(last_response.model_dump_json(indent=2))
                    else:
                        print(str(last_response))
                    print("─" * 90)
                else:
                    print("  No response available for this step")
            elif user_input == "u":
                if thinking_prompt:
                    print(f"\n{'─'*40} THINKING PROMPT {'─'*40}")
                    print(thinking_prompt)
                    print("─" * 90)
                else:
                    print("  No thinking prompt available (thinking mode disabled?)")
            elif user_input == "t":
                if thinking_text:
                    print(f"\n{'─'*40} THINKING TEXT {'─'*40}")
                    print(thinking_text)
                    print("─" * 90)
                else:
                    print("  No thinking text available (thinking mode disabled?)")
            elif user_input == "e":
                if len(error_history) == 0:
                    print("  NO ERROR HISTORY AVAILABLE (completed on first attempt?)")
                else:
                    print(f"{'─'*40} DETAILED ERROR HISTORY {'─'*40}")                
                    for i, error in enumerate(error_history):
                        print(f"\n[Attempt {i}]:")
                        print(error)
                    print(f"\n{'─'*40} ERROR SUMMARY ({len(error_history)} attempts) {'─'*40}")
                    if error_messages:
                        for i, msg in enumerate(error_messages, 1):
                            print(f"{i}. {msg}")
                    print("─" * 90)
                
            elif user_input == "b":
                # Traceback (only for fatal errors)
                if traceback_str:
                    print(f"\n{'─'*40} PYTHON TRACEBACK {'─'*40}")
                    print(traceback_str)
                    print("─" * 90)
                else:
                    print("  No traceback available")
            elif user_input == "q":
                self._save_log_file()
                raise KeyboardInterrupt("User requested abort in interactive mode")
            else:
                print(f"  Unknown command: '{user_input}'")
    
    # -------------------------------------------------------------------------
    # Context Builders - Build context strings for LLM prompts
    # -------------------------------------------------------------------------
    
    def _analysis_context(self, *fields: str) -> str:
        """
        Build context string from the problem analysis for specified fields.
        
        Args:
            fields: Which analysis fields to include ("types", "entities", "initial", "goals", "capabilities")
            
        Returns:
            Formatted string with the requested analysis information, or empty string if no analysis.
        """
        if not self.raw_analysis:
            return ""
        analysis = self.raw_analysis
        field_data = {
            "types": f"Identified types: {sorted(set(analysis.entity_types.values()))}",
            "entities": f"Entities: {analysis.entities}",
            "initial": f"Initial conditions: {analysis.initial_conditions}",
            "goals": f"Goal conditions: {analysis.goal_conditions}",
            "capabilities": f"Required capabilities: {analysis.required_capabilities}",
        }
        parts = [field_data[f] for f in fields if f in field_data]
        return "\nFrom problem analysis:\n" + "\n".join(f"- {p}" for p in parts) if parts else ""
    
    def _dynamic_context(self, step_key: str) -> str:
        """
        Build optimized context based on what's relevant for the current step.
        
        Uses CONTEXT_RELEVANCE configuration to include only the context that
        matters for each generation step, reducing prompt size.
        
        Args:
            step_key: The step identifier (e.g., "actions", "initial_state")
            
        Returns:
            Compressed context string with only relevant information in clear PDDL-like format
        """
        relevance = CONTEXT_RELEVANCE.get(step_key, {"analysis": [], "builder": ["types"]})
        parts = []
        
        # Add relevant analysis fields
        if self.raw_analysis and relevance.get("analysis"):
            analysis_parts = []
            field_map = {
                "types": f"Types identified: {sorted(set(self.raw_analysis.entity_types.values()))}",
                "entities": f"Entities: {self.raw_analysis.entities[:20]}{'...' if len(self.raw_analysis.entities) > 20 else ''}",
                "initial": f"Initial: {self.raw_analysis.initial_conditions[:3]}{'...' if len(self.raw_analysis.initial_conditions) > 3 else ''}",
                "goals": f"Goals: {self.raw_analysis.goal_conditions[:3]}{'...' if len(self.raw_analysis.goal_conditions) > 3 else ''}",
                "capabilities": f"Capabilities: {self.raw_analysis.required_capabilities[:5]}{'...' if len(self.raw_analysis.required_capabilities) > 5 else ''}",
            }
            for field in relevance["analysis"]:
                if field in field_map:
                    analysis_parts.append(field_map[field])
            if analysis_parts:
                analysis_prompt = f"""ANALYSIS:
A detailed analysis of the whole problem has been conducted to identify key elements.
The relevant findings are as follows:
"""
                analysis_prompt += "\n".join(f"  • {p}" for p in analysis_parts)
                parts.append(analysis_prompt)
                # parts.append("FROM ANALYSIS:\n" + "\n".join(f"  • {p}" for p in analysis_parts))
        
        # Add relevant builder state with clear PDDL-like formatting
        builder_fields = relevance.get("builder", [])
        builder_sections = []
        
        if "types" in builder_fields and self.builder.type_names:
            builder_sections.append("TYPES:\n" + self._format_types())
            
        if "predicates" in builder_fields and self.builder.predicate_names:
            builder_sections.append("PREDICATES:\n" + self._format_predicates())
            
        if "actions" in builder_fields and self.builder.action_names:
            builder_sections.append("ACTIONS:\n" + self._format_actions())
            
        if "objects" in builder_fields and self.builder.object_names:
            builder_sections.append("OBJECTS:\n" + self._format_objects())
            
        if "tasks" in builder_fields and self.is_htn and self.builder.task_names:
            builder_sections.append("TASKS:\n" + self._format_tasks())
        
        if builder_sections:
            builder_prompt = f"""\nCURRENT MODEL STATE:
The current state of the model includes the following components (response must use these exact names):
"""
            builder_prompt += "\n".join(f"• {p}" for p in builder_sections)
            parts.append(builder_prompt)
            # parts.append("CURRENT MODEL STATE:\n" + "\n\n".join(builder_sections))
                    
        context = "\n".join(parts)
        return f"{context}" if context else ""
    
    def _builder_context(self) -> str:
        """
        Build context string showing the current state of the model in clear PDDL-like format.
        
        Returns:
            Formatted string listing all components defined so far with their signatures.
        """
        sections = []
        
        if self.builder.type_names:
            sections.append("TYPES:\n" + self._format_types())
            
        if self.builder.predicate_names:
            sections.append("PREDICATES:\n" + self._format_predicates())
            
        if self.builder.action_names:
            sections.append("ACTIONS:\n" + self._format_actions())
            
        if self.builder.object_names:
            sections.append("OBJECTS:\n" + self._format_objects())
            
        if self.is_htn and self.builder.task_names:
            sections.append("TASKS:\n" + self._format_tasks())
        
        return "\n\n".join(sections) if sections else "No components defined yet"
    
    # -------------------------------------------------------------------------
    # Formatting Helpers - PDDL-like representation for clear prompts
    # -------------------------------------------------------------------------
    
    def _format_type(self, type_name: str) -> str:
        """Format a single type with its parent if any."""
        up_type = self.builder.get_type(type_name)
        if up_type.father and up_type.father.name != 'object':
            return f"{type_name} - {up_type.father.name}"
        return type_name
    
    def _format_types(self) -> str:
        """Format all types in PDDL-like syntax showing hierarchy."""
        if not self.builder.type_names:
            return "None defined yet"
        
        lines = []
        for type_name in self.builder.type_names:
            lines.append(f"  {self._format_type(type_name)}")
        return "\n".join(lines)
    
    def _format_predicate(self, pred_name: str) -> str:
        """Format predicate signature in PDDL syntax: (predicate_name ?param1 - type1 ?param2 - type2)"""
        sig = self.builder.get_predicate_signature(pred_name)
        if not sig:
            return f"({pred_name})"
        params = " ".join(f"?{name} - {typ}" for name, typ in sig)
        return f"({pred_name} {params})"
    
    def _format_predicates(self) -> str:
        """Format all predicates in PDDL syntax."""
        if not self.builder.predicate_names:
            return "None defined yet"
        
        lines = []
        for pred_name in self.builder.predicate_names:
            lines.append(f"  {self._format_predicate(pred_name)}")
        return "\n".join(lines)
    
    def _format_action(self, action_name: str) -> str:
        """Format action signature in PDDL syntax: (action_name ?param1 - type1 ?param2 - type2)"""
        sig = self.builder.get_action_signature(action_name)
        if not sig:
            return f"({action_name})"
        params = " ".join(f"?{name} - {typ}" for name, typ in sig)
        return f"({action_name} {params})"
    
    def _format_actions(self) -> str:
        """Format all actions in PDDL syntax."""
        if not self.builder.action_names:
            return "None defined yet"
        
        lines = []
        for action_name in self.builder.action_names:
            lines.append(f"  {self._format_action(action_name)}")
        return "\n".join(lines)
    
    def _format_task(self, task_name: str) -> str:
        """Format task signature in PDDL syntax: (task_name ?param1 - type1 ?param2 - type2)"""
        sig = self.builder.get_task_signature(task_name)
        if not sig:
            return f"({task_name})"
        params = " ".join(f"?{name} - {typ}" for name, typ in sig)
        return f"({task_name} {params})"
    
    def _format_tasks(self) -> str:
        """Format all tasks in PDDL syntax."""
        if not self.builder.task_names:
            return "None defined yet"
        
        lines = []
        for task_name in self.builder.task_names:
            lines.append(f"  {self._format_task(task_name)}")
        return "\n".join(lines)
    
    def _format_objects(self) -> str:
        """Format objects grouped by type in PDDL syntax."""
        if not self.builder.object_names:
            return "None defined yet"
        
        objects_by_type = self.builder.get_objects_by_type()
        lines = []
        for type_name, obj_names in sorted(objects_by_type.items()):
            objs_str = " ".join(obj_names)
            lines.append(f"  {objs_str} - {type_name}")
        return "\n".join(lines)
    
    def _format_signatures(self, names: List[str], signature_getter: Callable) -> str:
        """
        Format signatures of predicates, actions, or tasks for display in prompts.
        
        Args:
            names: List of predicate/action/task names
            signature_getter: Function that returns the signature for a given name
            
        Returns:
            Formatted string with each name and its parameter signature.
        """
        return "\n".join(f"  {name}: {signature_getter(name)}" for name in names)
    
    def _merge_actions(self, newly_added_actions: List[ActionDef]) -> None:
        """
        Merge newly added actions into raw_actions, avoiding duplicates.
        
        Args:
            newly_added_actions: List of ActionDef that were just validated and added to builder.
        """
        if newly_added_actions:
            existing_actions = list(self.raw_actions.actions) if self.raw_actions else []
            self.raw_actions = ActionList(actions=existing_actions + newly_added_actions)
    
    def _build_retry_prompt(
        self, 
        base_prompt: str, 
        error_history: List[str], 
        context_info: str,
        step_name: str,
        error_messages: List[str],
        last_result: Optional[BaseModel] = None
    ) -> str:
        """
        Build an enhanced retry prompt with detailed error feedback to help the LLM correct its mistakes.
        
        This method formats error information in a way that maximizes the LLM's ability to 
        understand and fix the issues. It includes:
        - Full error messages with their context
        - Current model state for reference
        - Specific guidance on common error patterns
        - Clear instructions on what needs to be fixed
        - Strategy variation for persistent errors
        
        Args:
            base_prompt: The original prompt
            error_history: List of error messages from previous attempts
            context_info: Current state of the model being built
            step_name: Name of the generation step (for context)
            last_result: The last generated result (to check for empty reasoning/description)
            
        Returns:
            Enhanced prompt with error feedback
        """
        attempt_num = len(error_history) + 1
        
        # Format errors with clear structure
        error_details = []
        for i, error in enumerate(error_history, 1):
            error_details.append(f"[ATTEMPT {i}]\n{error}")
        error_section = "\n\n".join(error_details)
        
        # Strategy variation based on retry count
        strategy_hint = ""
        if attempt_num >= 2:
            strategy_hint = f"""\n
{'=' * 80}
🔄 RETRY STRATEGY FOR ATTEMPT {attempt_num}
{'=' * 80}

Since previous attempts failed, try a DIFFERENT APPROACH:
- SIMPLIFY: Use fewer, simpler constructs
- BE CONSERVATIVE: Only use names you see in "Available" lists - don't invent new ones
- CHECK CAREFULLY: Copy-paste predicate/type names from the lists if needed
- THINK STEP BY STEP: Before writing each item, verify all its parts are valid

"""
        
        return f"""{base_prompt}

{'=' * 80}
⚠️  [IMPORTANT] GENERATION ERROR - This is your attempt #{attempt_num} to generate a correct response.
Your previous response(s) for '{step_name}' contained errors that must be fixed.
Here are the details of what went wrong on the previous attempt(s):
{'=' * 80}

{error_section}{strategy_hint}

{'=' * 80}
HOW TO FIX COMMON ERRORS:
{'=' * 80}

1. "Type 'X' does not exist"
   → The type name is misspelled OR it wasn't defined yet
   → Check the "Available types" list above and use exact spelling
   → Remember: types must be defined before they can be referenced

2. "Predicate 'X' does not exist"  
   → The predicate name is misspelled
   → Check the predicate signatures above for correct names
   → Use lowercase with underscores (e.g., 'on_table', not 'onTable')

3. "Parameter 'X' is not defined"
   → You're using a variable that isn't a parameter of the current action/method
   → All arguments in preconditions/effects must be from the action's parameters
   → Use ?param_name format and ensure it's listed in the parameters

4. "Wrong number of arguments"
   → Check the predicate/action signature for the correct parameter count
   → Example: if predicate is 'on(?x, ?y)' you need exactly 2 arguments

5. "Type mismatch"
   → The argument type doesn't match what the predicate/action expects
   → Check the signature: e.g., 'on(?block: block, ?surface: surface)'
   → Make sure your argument has the correct type

6. "Object 'X' does not exist"
   → The object name is misspelled or wasn't defined
   → Check "Available objects" above for exact names

7. "Task 'X' does not exist" (HTN)
   → The task name is misspelled
   → Remember: tasks are different from actions
   → Check "Available tasks" above

{'=' * 80}
CURRENT MODEL STATE - The following sections have been already defined and can not be changed.
Use ONLY the names and signatures from these lists in your response, and ensure your new content is consistent with them
{'=' * 80}

{context_info}

{'=' * 80}
SUMMARY OF ERRORS FROM PREVIOUS ATTEMPT(S) - Review these carefully to understand what needs to be fixed:
{'=' * 80}
""" + "\n".join(f"{i}. {msg}" for i, msg in enumerate(error_messages, 1)) + f"""

{'=' * 80}
YOUR RESPONSIBILITY:
- Carefully read the error messages above - they tell you exactly what's wrong
- Fix ALL issues mentioned - don't ignore any errors
- Use ONLY names from the "Available" lists in the current model state
- Ensure your output is valid JSON matching the required schema
- If your previous reasoning/description was empty or very short, provide more detailed explanations this time
- Note that, if you want to declare a negated predicate (like a 'not' precondition or goal), you must use the "negated": true field in the JSON definitiion. Do NOT use a 'not' operator in the predicate name or elsewhere.

Generate the corrected JSON output for: '{step_name}'
"""

    def _build_thinking_prompt(self, step_name: str, base_prompt: str, context_info: str) -> str:
        """
        Build a focused Chain-of-Thought prompt for the thinking step.
        
        The prompt is SPECIFIC to the current generation step - it doesn't ask the LLM
        to reason about the entire domain, just what's needed for this particular element.
        
        This includes the base_prompt so the LLM knows:
        - The exact TASK it's solving
        - Examples of correct output
        - All requirements and constraints
        
        Then adds step-specific thinking instructions to guide reasoning.
        
        Args:
            step_name: Name of the current generation step
            base_prompt: The original generation prompt (contains TASK, EXAMPLE, REQUIREMENTS)
            context_info: Current model state context (already included in base_prompt)
            
        Returns:
            Focused thinking prompt for this specific step
        """
        step_key = self._normalize_step_key(step_name)
        
        # Step-specific thinking instructions
        thinking_instructions = {
            "analysis": """Think step-by-step about:
• What entities/objects are mentioned or implied in the text?
• What are their types/categories?
• What is the initial situation described?
• What is the goal or objective?
• What capabilities/actions would be needed to solve this?""",
            
            "types": """Think step-by-step about:
• What categories of objects exist in this domain?
• Is there a natural hierarchy (some types being subtypes of others)?
• What types are needed for the predicates and actions we'll define later?""",
            
            "predicates": """Think step-by-step about:
• What relationships exist between objects? (e.g., location, containment, ordering)
• What properties can objects have? (e.g., status flags, states)
• What will need to be checked in action preconditions?
• What will change as effects of actions?""",
            
            "objects": """Think step-by-step about:
• What specific instances/objects exist in THIS problem?
• What type is each object?
• Are all objects mentioned in the problem description included?""",
            
            "actions": """Think step-by-step about this specific action:
• What does this action accomplish in the real world?
• What must be TRUE before this action can execute? (preconditions)
• What becomes TRUE after this action? (add effects)
• What becomes FALSE after this action? (delete effects)
• What parameters (objects) are involved?""",
            
            "tasks": """Think step-by-step about:
• What high-level goals need to be accomplished?
• How do these differ from primitive actions?
• What parameters does each task need?""",
            
            "methods": """Think step-by-step about this specific method:
• When should this method be used? (preconditions that distinguish it)
• What sequence of subtasks/actions accomplishes the task?
• What additional parameters are needed beyond the task's parameters?
• How do method parameters map to task parameters (task_args)?""",
            
            "initial_state": """Think step-by-step about:
• What facts are TRUE at the start of the problem?
• For each object: where is it? What properties does it have?
• What relationships exist between objects initially?""",
            
            "goal": """Think step-by-step about:
• What must be TRUE at the end for the problem to be solved?
• What relationships must exist in the final state?
• Are there any facts that must be FALSE? (negated goals)""",
            
            "objects": """Think step-by-step about:
• What concrete instances exist in THIS problem?
• What type is each object?
• Are all mentioned entities included?"""
        }
        
        step_instructions = thinking_instructions.get(step_key, """Think step-by-step about:
• What exactly is being asked for?
• What constraints must be satisfied?
• What are the key elements to include?""")
        
        # Build thinking prompt: include base_prompt so LLM knows the full task
        return f"""{base_prompt}

{'=' * 80}
REASONING STEP: Before generating JSON, think through your approach
{'=' * 80}

{step_instructions}

{'=' * 80}
INSTRUCTION:
{'=' * 80}

Think step-by-step in natural language. Explain your reasoning for each element 
you will generate. Write your detailed thinking below, then you'll use it to 
generate the JSON output that satisfies all requirements above."""

    def _build_json_prompt(self, step_name: str, base_prompt: str, thinking_text: str, context_info: str = "") -> str:
        """
        Build the JSON generation prompt that incorporates the thinking result.
        
        This prompt:
        1. Repeats the base_prompt so LLM has full task context
        2. Shows the reasoning from the thinking step
        3. Asks for JSON output matching the schema
        
        Args:
            base_prompt: Original generation prompt with TASK, EXAMPLE, REQUIREMENTS, CONTEXT
            thinking_text: The reasoning generated in the thinking step
            context_info: (unused - context is in base_prompt) kept for API compatibility
            step_name: Name of the step (for logging/clarity)
            
        Returns:
            Complete JSON generation prompt with thinking integrated
        """
        # Truncate thinking if too long to avoid context overflow
        thinking_summary = thinking_text[:1500] + ("\n...(truncated)..." if len(thinking_text) > 3000 else "") + thinking_text[-500:] if thinking_text else "(no prior thinking)"
        
        return f"""{base_prompt}

{'=' * 80}
YOUR REASONING FROM THIS STEP:
{'=' * 80}

{thinking_summary}

{'=' * 80}
NOW GENERATE THE JSON:
{'=' * 80}

Based on your reasoning above and all the requirements, generate the complete JSON output.
Ensure it matches the schema shown in the EXAMPLE OUTPUT section above.
Use ONLY names and types from the CONTEXT section.
Be thorough and follow all REQUIREMENTS."""

    def _summarize_result(self, result: BaseModel, step_name: str) -> str:
        """
        Generate a concise single-line summary of what was generated.
        
        Extracts names from the JSON and displays them.
        """
        data = result.model_dump()
        
        # Find first list field with dicts containing 'name'
        for key, value in data.items():
            if isinstance(value, list) and value and isinstance(value[0], dict) and 'name' in value[0]:
                names = [item.get('name', '') for item in value if isinstance(item, dict) and 'name' in item]
                if names:
                    names_display = ", ".join(names)
                    return f"  ✓ {len(names)} {key}: {names_display}"
        
        return f"  ✓ Summary generated successfully"


    @staticmethod
    def _normalize_step_key(step_name: str) -> str:
        """
        Normalize a step name to a key that matches TEMPERATURE_CONFIG and CONTEXT_RELEVANCE.
        
        Maps various step name formats to canonical keys:
        - "Problem Analysis" → "analysis"
        - "Types" → "types"
        - "Actions_Names", "Action_pickup" → "actions"
        - "Methods_Names_for_task" → "methods"
        - "HTN Tasks" → "tasks"
        - "Initial State" → "initial_state"
        - "HTN Goal", "Goal State" → "goal"
        
        Args:
            step_name: Step name as passed to _generate_and_validate
            
        Returns:
            Normalized key suitable for TEMPERATURE_CONFIG and CONTEXT_RELEVANCE
        """
        step_key = step_name.lower().replace(" ", "_")
        
        # Normalize to known keys
        if "problem_analysis" in step_key or "analysis" in step_key:
            return "analysis"
        elif "type" in step_key:
            return "types"
        elif "predicate" in step_key:
            return "predicates"
        elif "action" in step_key:
            return "actions"
        elif "task" in step_key and "method" not in step_key:
            return "tasks"
        elif "method" in step_key:
            return "methods"
        elif "object" in step_key:
            return "objects"
        elif "initial" in step_key:
            return "initial_state"
        elif "goal" in step_key:
            return "goal"
        else:
            # Fallback: return lowercase version without spaces
            return step_key
    
    def _generate_and_validate(
        self, 
        base_prompt: str, 
        response_model: Type[T], 
        validator_function: Callable,
        context_info: str = "",
        step_name: str = "generation"
    ) -> T:
        """
        Core generation method: Call LLM, validate result, retry on failure.
        
        This method integrates structured output generation with semantic validation.
        It handles the retry loop, building enhanced prompts with error feedback
        when validation fails. Also tracks metrics for each step.
        
        Args:
            base_prompt: The initial prompt to send to the LLM
            response_model: Pydantic model class defining expected output structure
            validator_function: Function that validates the result by calling PDDLBuilder methods.
                               Should raise an exception if validation fails.
            context_info: Additional context to include in error feedback prompts
            step_name: Human-readable name for this generation step (for logging/errors)
            
        Returns:
            Validated instance of response_model
            
        Raises:
            PDDLGenerationError: If validation fails after all retry attempts
        """
        error_history = []
        error_messages = []  # Track only error messages (not full entries)
        generation_prompt = base_prompt  # Prompt for generation (may include retry feedback)
        last_result = None  # For interactive mode
        thinking_text = ""
        thinking_prompt = ""
        
        # Start tracking metrics for this step (if enabled)
        if self.track_metrics:
            self.metrics.start_step(step_name)
        
        # Determine temperature based on step type
        step_key = self._normalize_step_key(step_name)
        temperature = get_temperature(step_key)
        
        self._log(LogLevel.DEBUG, f"  → '{step_name}': max_retries={self.max_retries}, temp={temperature:.2f}")
        
        for attempt in range(self.max_retries):
            try:
                # If think_mode enabled, do two-step generation
                if self.think_mode:
                    # Step 1: Get model to think - prompt is SPECIFIC to the step
                    thinking_prompt = self._build_thinking_prompt(step_name, generation_prompt, context_info)
                    
                    self._log(LogLevel.INFO, f"    💭 Thinking...")
                    self._log(LogLevel.DEBUG, f"    Attempt {attempt + 1}/{self.max_retries} (THINK)...", end=" ")
                    self._log_prompt(thinking_prompt, f"'{step_name}' thinking (attempt {attempt + 1})")

                    # Get thinking response (raw text, no structured output)
                    try:
                        # messages_think = [{"role": "user", "content": thinking_prompt}]
                        thinking_text = self.client.chat(thinking_prompt, temperature=get_temperature("think"))
                    except Exception as e:
                        self._log(LogLevel.INFO, f"(thinking fallback: {e})")
                        thinking_text = ""
                    
                    self._log(LogLevel.DEBUG, "✓")
                    self._log(LogLevel.INFO, f"    📝 Generating JSON...")
                    # Step 2: Generate JSON based on thinking - incorporates thinking and context into generation prompt
                    final_json_prompt = self._build_json_prompt(step_name, generation_prompt, thinking_text, context_info)
                else:
                    # No thinking mode - use generation prompt directly
                    final_json_prompt = generation_prompt

                # Prepare JSON generation prompt
                messages = [{"role": "user", "content": final_json_prompt}]
                self._log(LogLevel.DEBUG, f"    Attempt {attempt + 1}/{self.max_retries} (JSON)...", end=" ")
                self._log_prompt(final_json_prompt, f"'{step_name}' (attempt {attempt + 1})")
                
                # Call LLM
                result = self.client.create(
                    messages=messages, 
                    response_model=response_model, 
                    temperature=temperature
                )
                last_result = result
                
                # Log response at TRACE level
                self._log_response(result, f"'{step_name}' (attempt {attempt + 1})")
                
                # Validate
                validator_function(result)
                
                self._log(LogLevel.DEBUG, "✓")
                
                # Record success metrics (if enabled)
                if self.track_metrics:
                    self.metrics.end_step(step_name, success=True, attempts=attempt + 1, errors=error_history)
                
                # Print concise summary before interactive inspection
                summary = self._summarize_result(result, step_name)
                self._log(LogLevel.INFO, summary + (f" ({attempt + 1} attempts)" if attempt > 1 else ""))
                
                # Interactive inspection after success (with thinking context if available, but only if interactive mode)
                # Pass error_history so user can review retries if interested
                self._interactive_inspection(step_name, error_history, error_messages, final_json_prompt, result, 
                                            thinking_prompt if self.think_mode else None,
                                            thinking_text if self.think_mode else None)

                
                return result
                
            except Exception as e:
                error_message = str(e)
                
                # Capture output that caused the error
                output_str = ""
                if last_result is not None:
                    try:
                        # Try to dump as JSON if it's a Pydantic model
                        if hasattr(last_result, 'model_dump_json'):
                            output_str = last_result.model_dump_json(indent=2)
                        elif hasattr(last_result, 'model_dump'):
                            import json
                            output_str = json.dumps(last_result.model_dump(), indent=2)
                        else:
                            output_str = str(last_result)
                    except Exception:
                        output_str = str(last_result)
                
                # Store error message for retry prompt
                error_messages.append(error_message)
                
                # Store full error context for debugging
                error_entry = f'''
→ Reasoning trace:
    {thinking_text if thinking_text else "(no thinking process done)"}
                
→ Generated output:
    {output_str}

→ Error:
    {error_message}'''
                error_history.append(error_entry)
                
                self._log(LogLevel.DEBUG, f"✗ {error_message[:1000]}{'...' if len(error_message) > 1000 else ''}")
                
                if attempt < self.max_retries - 1:
                    # Build enhanced retry prompt with detailed error feedback and current error history
                    generation_prompt = self._build_retry_prompt(
                        base_prompt, error_history, context_info, step_name, error_messages, last_result
                    )
                    self._log(LogLevel.DEBUG, "    Retrying with error feedback...")
                else:
                    self._log(LogLevel.INFO, f"  ✗ '{step_name}': FAILED after {self.max_retries} attempts")
                    self._log(LogLevel.DEBUG, f"    Last error: {error_message}")
                    
                    # Record failure metrics (if enabled)
                    if self.track_metrics:
                        self.metrics.end_step(step_name, success=False, attempts=attempt + 1, errors=error_history)
                    
                    # Capture traceback
                    import traceback as tb_module
                    tb_str = tb_module.format_exc()
                    
                    # Force interactive inspection on fatal error (all retries exhausted)
                    # This will pause even outside interactive mode to allow error debugging
                    self._interactive_inspection(
                        step_name,
                        error_history,
                        error_messages,
                        final_json_prompt,
                        last_result,
                        thinking_prompt if self.think_mode else None,
                        thinking_text if self.think_mode else None,
                        tb_str,
                        is_fatal=True
                    )
                    
                    raise PDDLGenerationError(step_name, self.max_retries, error_messages, tb_str) from e
                
            except KeyboardInterrupt:
                self._log(LogLevel.INFO, f"\n⏹ [EXCEPTION] Generation aborted by user during '{step_name}'\n\n")
                print(f"\n{'─'*40} LAST PROMPT {'─'*40}")
                print(final_json_prompt if final_json_prompt else "  No prompt available for this step")
                print()
                print(f"\n{'─'*40} LAST THINKING {'─'*40}")
                print(thinking_text if thinking_text else "  No thinking text available for this step")
                print()
                print(f"\n{'─'*40} LAST RESPONSE {'─'*40}")
                if last_result:
                    if hasattr(last_result, 'model_dump_json'):
                        print(last_result.model_dump_json(indent=2))
                    else:
                        print(str(last_result))
                    print("─" * 90)
                else:
                    print("  No response available for this step")
   
                self.metrics.end_generation()
                self._save_log_file()
                
                # Capture traceback and raise with full context
                import traceback as tb_module
                tb_str = tb_module.format_exc()
                raise PDDLGenerationError(step_name, self.max_retries, error_messages, tb_str) from e
    
    # -------------------------------------------------------------------------
    # Generation Methods - Each generates one component of the PDDL model
    # -------------------------------------------------------------------------
    
    def analyze_problem(self, task_description: str) -> ProblemAnalysis:
        """
        Analyze the task description to extract entities and relationships.
        
        This initial analysis helps guide the subsequent generation steps by
        identifying all objects, their types, and what needs to happen.
        """
        prompt = f"""{'=' * 80}
TASK: Analyze the planning problem to identify entities, types, conditions, and capabilities
{'=' * 80}

{task_description}

{'=' * 80}
EXAMPLE OUTPUT:
{'=' * 80}

For a blocksworld problem "Move block A from B to the table":
{{
  "description": "Blocksworld: rearrange blocks by moving them between positions",
  "entities": ["block_a", "block_b", "table"],
  "entity_types": {{"block_a": "block", "block_b": "block", "table": "surface"}},
  "initial_conditions": ["block_a is on block_b", "block_b is on table"],
  "goal_conditions": ["block_a is on table"],
  "required_capabilities": ["pick up blocks", "put down blocks"]
}}

{'=' * 80}
REQUIREMENTS:
{'=' * 80}

1. ENTITIES: All concrete objects mentioned or implied (lowercase with underscores)
2. ENTITY_TYPES: Map each entity to its type category (general categories)
3. INITIAL_CONDITIONS: Natural language facts true at the start
4. GOAL_CONDITIONS: Natural language descriptions of what must be achieved
5. REQUIRED_CAPABILITIES: Actions/capabilities needed to solve the problem
6. DESCRIPTION: Brief summary of the planning problem"""
        
        def validate_analysis(result: ProblemAnalysis):
            entity_types_keys = set(result.entity_types.keys())
            for entity in result.entities:
                if _normalize_name(entity) not in entity_types_keys:
                    raise ValueError(f"Entity '{entity}' is listed in entities but missing from entity_types mapping")
        
        self.raw_analysis = self._generate_and_validate(
            prompt, ProblemAnalysis, validate_analysis, step_name="Problem Analysis"
        )
        return self.raw_analysis
    
    def generate_types(self, task_description: str) -> TypeList:
        """
        Generate the type hierarchy for the PDDL domain.
        
        Types define categories of objects in the domain. They can have parent types
        to form a hierarchy (e.g., 'truck' could be a subtype of 'vehicle').
        """
        dynamic_ctx = self._dynamic_context("types")
        
        prompt = f"""{'=' * 80}
TASK: Define the type hierarchy for the PDDL domain
{'=' * 80}

{task_description}

{'=' * 80}
EXAMPLE OUTPUT:
{'=' * 80}

{{
  "types": [
    {{"name": "location", "parent": null, "description": "A place where objects can be"}},
    {{"name": "city", "parent": "location", "description": "A city, which is a type of location"}},
    {{"name": "package", "parent": null, "description": "An item to be transported"}}
  ]
}}

{'=' * 80}
CONTEXT:
{'=' * 80}
{dynamic_ctx}

{'=' * 80}
REQUIREMENTS:
{'=' * 80}

- Use lowercase names with underscores (e.g., 'block', 'location')
- Order parent types BEFORE their children in the list
- Only set parent if there's a clear is-a relationship
- Include description for each type explaining its purpose"""
        
        def validate_types(result: TypeList):
            for type_def in result.types:
                self.builder.add_type(type_def.name, type_def.parent)
            
            # Verify all types from analysis are covered
            if self.raw_analysis:
                required_types = set(self.raw_analysis.entity_types.values())
                defined_types = set(self.builder.type_names)
                missing_types = required_types - defined_types
                if missing_types:
                    raise ValueError(f"Missing types that were identified in analysis: {list(missing_types)}")
        
        self.raw_types = self._generate_and_validate(
            prompt, TypeList, validate_types, 
            context_info=dynamic_ctx, 
            step_name="Types"
        )
        return self.raw_types
    
    def generate_predicates(self, task_description: str) -> PredicateList:
        """
        Generate predicates (relations) for the PDDL domain.
        
        Predicates describe relationships between objects and properties of objects.
        They are the building blocks used in preconditions and effects of actions.
        """
        dynamic_ctx = self._dynamic_context("predicates")
        
        prompt = f"""{'=' * 80}
TASK: Define predicates (relationships and properties) for the PDDL domain
{'=' * 80}

{task_description}

{'=' * 80}
EXAMPLE OUTPUT:
{'=' * 80}

{{
  "predicates": [
    {{
      "name": "on",
      "parameters": [{{"name": "?upper", "type": "block"}}, {{"name": "?lower", "type": "block"}}],
      "description": "Block ?upper is directly on top of block ?lower"
    }},
    {{
      "name": "clear",
      "parameters": [{{"name": "?x", "type": "block"}}],
      "description": "Nothing is on top of block ?x"
    }}
  ]
}}

{'=' * 80}
CONTEXT:
{'=' * 80}
{dynamic_ctx}

{'=' * 80}
REQUIREMENTS:
{'=' * 80}

- Predicates describe relationships (e.g., 'on') or properties (e.g., 'clear')
- Use lowercase names with underscores
- Each parameter must have a type from the available types shown in context
- Parameter names start with '?' (e.g., '?block', '?location')
- Include clear description of what each predicate means when true"""
        
        def validate_predicates(result: PredicateList):
            """Full validation."""
            for predicate in result.predicates:
                if _normalize_name(predicate.name) not in self.builder.predicate_names:
                    self.builder.add_predicate(predicate.name, predicate.parameters)
        
        self.raw_predicates = self._generate_and_validate(
            prompt, PredicateList, validate_predicates, 
            context_info=dynamic_ctx, 
            step_name="Predicates"
        )
        return self.raw_predicates
    
    def generate_actions(self, task_description: str, action_names: Optional[List[str]] = None) -> ActionList:
        """
        Generate actions for the PDDL domain.
        
        This is a convenience method that combines generate_action_names() and
        generate_action() for simple use cases. For more control, use those
        methods directly.
        
        Args:
            task_description: Natural language description of the domain
            action_names: Optional list of action names to generate. If None,
                         will first call generate_action_names() to get them.
        
        Returns:
            ActionList containing all generated actions
        
        Example (simple):
            modeler.generate_actions(task_desc)  # Auto-discovers and generates all
        
        Example (controlled):
            names = modeler.generate_action_names(task_desc)
            names = [n for n in names if n != 'unwanted_action']  # Filter
            modeler.generate_actions(task_desc, action_names=names)
        """
        # Step 1: Get action names (if not provided)
        if action_names is None:
            action_names = self.generate_action_names(task_description)
        
        # Step 2: Generate each action individually
        for action_name in action_names:
            self.generate_action(task_description, action_name)
        
        return self.raw_actions
    
    def generate_action_names(self, task_description: str) -> List[str]:
        """
        Ask the LLM to identify what actions are needed for this domain.
        
        This is step 1 of the action generation process. It returns a list of
        action names that should be generated. You can then filter, modify, or
        extend this list before calling generate_action() for each one.
        
        Args:
            task_description: Natural language description of the domain
            
        Returns:
            List of action names (e.g., ['pick_up', 'put_down', 'stack'])
        
        Example:
            names = modeler.generate_action_names(task_desc)
            print(f"LLM suggests: {names}")
            
            # Add a custom action
            names.append('my_custom_action')
            
            # Generate each one
            for name in names:
                modeler.generate_action(task_desc, name)
        """
        dynamic_ctx = self._dynamic_context("actions")
        
        names_prompt = f"""{'=' * 80}
TASK: Identify all primitive ACTIONS needed for this domain
{'=' * 80}

{task_description}

{'=' * 80}
EXAMPLE OUTPUT:
{'=' * 80}

{{
  "names": ["pick_up", "put_down", "stack", "unstack"]
}}

{'=' * 80}
CONTEXT:
{'=' * 80}
{dynamic_ctx}

{'=' * 80}
REQUIREMENTS:
{'=' * 80}

- List all primitive actions (directly executable operations)
- Use lowercase names with underscores
- Actions should match the required_capabilities from analysis"""
        
        def validate_action_names(result: ActionNameList):
            """Just validate format - no state changes yet."""
            if not result.names:
                raise ValueError("No action names provided")
            for name in result.names:
                if not name or not name.strip():
                    raise ValueError("Empty action name provided")
        
        action_names_result = self._generate_and_validate(
            names_prompt, ActionNameList, validate_action_names,
            context_info=dynamic_ctx,
            step_name="Actions_Names"
        )
        
        action_names = action_names_result.names
        self._log(LogLevel.DEBUG, f"  → Sequential generation: {len(action_names)} actions: {action_names}")
        
        return action_names
    
    def generate_action(self, task_description: str, action_name: str) -> ActionDef:
        """
        Generate a single action definition given its name.
        
        This is step 2 of the action generation process. Call this for each
        action you want to generate. The action will be validated and added
        to the builder automatically.
        
        Args:
            task_description: Natural language description of the domain
            action_name: Name of the action to generate (e.g., 'pick_up')
            
        Returns:
            ActionDef containing the generated action definition
            
        Raises:
            PDDLGenerationError: If generation fails after max retries
            ValueError: If action name is invalid
        
        Example:
            # Generate a specific action
            action = modeler.generate_action(task_desc, 'pick_up')
            print(f"Generated: {action.name} with {len(action.parameters)} params")
            
            # Regenerate if not satisfied
            action = modeler.generate_action(task_desc, 'pick_up')  # Try again
        """
        dynamic_ctx = self._dynamic_context("actions")
        
        action_prompt = f"""{'=' * 80}
TASK: Define the action '{action_name}' with its parameters, preconditions, and effects
{'=' * 80}

{task_description}

{'=' * 80}
EXAMPLE OUTPUT:
{'=' * 80}

{{
  "name": "pick_up",
  "parameters": [{{"name": "?b", "type": "block"}}],
  "preconditions": [
    {{"predicate": "clear", "args": ["?b"], "negated": false}},
    {{"predicate": "arm_empty", "args": [], "negated": false}}
  ],
  "effects": [
    {{"predicate": "holding", "args": ["?b"], "negated": false}},
    {{"predicate": "arm_empty", "args": [], "negated": true}}
  ],
  "description": "Pick up a block from its current position"
}}

{'=' * 80}
CONTEXT:
{'=' * 80}
{dynamic_ctx}

{'=' * 80}
REQUIREMENTS:
{'=' * 80}

1. Action name MUST be: '{action_name}'
2. Predicate names must EXACTLY match the predicates shown in context
3. Argument count must match predicate signatures
4. All arguments must be action parameters (use ?param_name format)
5. Effects: negated=false → becomes TRUE, negated=true → becomes FALSE
6. Include description explaining what the action does"""
        
        def validate_single_action(result: ActionDef):
            """Validate and add this single action."""
            action = result
            # Verify name matches
            if _normalize_name(action.name) != _normalize_name(action_name):
                raise ValueError(f"Expected action name '{action_name}', got '{action.name}'")
            # Add to builder (will validate predicates, types, etc.)
            if _normalize_name(action.name) not in self.builder.action_names:
                self.builder.add_action(
                    action.name, action.parameters,
                    action.preconditions, action.effects
                )
        
        self._log(LogLevel.DEBUG, f"  Generating action '{action_name}'...")
        
        result = self._generate_and_validate(
            action_prompt, ActionDef, validate_single_action,
            context_info=dynamic_ctx,
            step_name=f"Action {action_name}"
        )
        
        # Add to raw_actions tracking
        self._merge_actions([result])
        
        return result
    
    def generate_tasks(self, task_description: str) -> TaskList:
        """
        Generate HTN tasks (compound/abstract tasks).
        
        HTN tasks represent high-level goals that need to be decomposed into
        subtasks or primitive actions via methods. Unlike actions, tasks are
        not directly executable - they must be refined by methods.
        """
        if not self.is_htn:
            raise ValueError("Tasks only available for HTN domains")
        
        dynamic_ctx = self._dynamic_context("tasks")
        
        prompt = f"""{'=' * 80}
TASK: Define HTN compound tasks (abstract goals) for this domain
{'=' * 80}

{task_description}

{'=' * 80}
EXAMPLE OUTPUT:
{'=' * 80}

{{
  "tasks": [
    {{
      "name": "deliver_package",
      "parameters": [
        {{"name": "?pkg", "type": "package"}},
        {{"name": "?dest", "type": "location"}}
      ],
      "description": "Deliver a package to a destination location"
    }},
    {{
      "name": "transport",
      "parameters": [{{"name": "?obj", "type": "object"}}, {{"name": "?to", "type": "location"}}],
      "description": "Transport an object to a location"
    }}
  ]
}}

{'=' * 80}
CONTEXT:
{'=' * 80}
{dynamic_ctx}

{'=' * 80}
REQUIREMENTS:
{'=' * 80}

1. Tasks are ABSTRACT goals decomposed by methods - NOT primitive actions
2. Use lowercase names with underscores
3. Parameter types must exist in the available types shown in context
4. Include description explaining the high-level goal of each task"""
        
        def validate_tasks(result: TaskList):
            for task in result.tasks:
                self.builder.add_task(task.name, task.parameters)
        
        self.raw_tasks = self._generate_and_validate(
            prompt, TaskList, validate_tasks,
            context_info=dynamic_ctx, 
            step_name="HTN Tasks"
        )
        return self.raw_tasks
    
    def generate_methods(self, task_description: str, task_name: str, 
                         method_names: Optional[List[str]] = None) -> MethodList:
        """
        Generate methods for decomposing a specific HTN task.
        
        This is a convenience method that combines generate_method_names() and
        generate_method() for simple use cases. For more control, use those
        methods directly.
        
        Args:
            task_description: Natural language description of the domain
            task_name: Name of the task to decompose (must be defined)
            method_names: Optional list of method names to generate. If None,
                         will first call generate_method_names() to get them.
        
        Returns:
            MethodList containing all generated methods for this task
        
        Example (simple):
            modeler.generate_methods(task_desc, 'deliver')  # Auto-discovers methods
        
        Example (controlled):
            names = modeler.generate_method_names(task_desc, 'deliver')
            names.append('deliver_by_air')  # Add custom method
            modeler.generate_methods(task_desc, 'deliver', method_names=names)
        """
        if not self.is_htn:
            raise ValueError("Methods only available for HTN domains")
        
        # Step 1: Get method names (if not provided)
        if method_names is None:
            method_names = self.generate_method_names(task_description, task_name)
        
        # Step 2: Generate each method individually
        generated_methods = []
        for method_name in method_names:
            method = self.generate_method(task_description, task_name, method_name)
            generated_methods.append(method)
        
        return MethodList(methods=generated_methods)
    
    def generate_method_names(self, task_description: str, task_name: str) -> List[str]:
        """
        Ask the LLM to identify what methods are needed to decompose a task.
        
        This is step 1 of the method generation process. It returns a list of
        method names that should be generated for the given task. You can then
        filter, modify, or extend this list before calling generate_method().
        
        Args:
            task_description: Natural language description of the domain
            task_name: Name of the task to find decomposition methods for
            
        Returns:
            List of method names (e.g., ['deliver_by_truck', 'deliver_by_drone'])
        
        Example:
            names = modeler.generate_method_names(task_desc, 'deliver')
            print(f"Methods for 'deliver': {names}")
            
            # Generate only truck-based methods
            truck_methods = [n for n in names if 'truck' in n]
            for name in truck_methods:
                modeler.generate_method(task_desc, 'deliver', name)
        """
        dynamic_ctx = self._dynamic_context("methods")
        task_signature = self.builder.get_task_signature(task_name)
        
        existing_methods = list(self.builder.method_names)
        
        names_prompt = f"""{'=' * 80}
TASK: Identify alternative METHODS to decompose the task '{task_name}'
{'=' * 80}

{task_description}

{'=' * 80}
EXAMPLE OUTPUT:
{'=' * 80}

{{
  "names": ["deliver_by_truck", "deliver_by_drone", "deliver_by_train"]
}}

{'=' * 80}
CONTEXT:
{'=' * 80}
{dynamic_ctx}

TASK TO DECOMPOSE: {self._format_task(task_name)}
EXISTING METHODS (don't duplicate): {existing_methods}

{'=' * 80}
REQUIREMENTS:
{'=' * 80}

- List different ways to accomplish the task (different strategies/methods)
- Use lowercase names with underscores
- Don't duplicate existing methods shown above"""
        
        def validate_method_names(result: MethodNameList):
            """Just validate format - no state changes yet."""
            if not result.names:
                raise ValueError("No method names provided")
            for name in result.names:
                if not name or not name.strip():
                    raise ValueError("Empty method name provided")
        
        method_names_result = self._generate_and_validate(
            names_prompt, MethodNameList, validate_method_names,
            context_info=dynamic_ctx,
            step_name=f"Methods_Names_for_{task_name}"
        )
        
        method_names = method_names_result.names
        self._log(LogLevel.DEBUG, f"  → Sequential generation for '{task_name}': {len(method_names)} methods: {method_names}")
        
        return method_names
    
    def generate_method(self, task_description: str, task_name: str, method_name: str) -> MethodDef:
        """
        Generate a single method definition given its name.
        
        This is step 2 of the method generation process. Call this for each
        method you want to generate. The method will be validated and added
        to the builder automatically.
        
        Args:
            task_description: Natural language description of the domain
            task_name: Name of the task this method decomposes
            method_name: Name of the method to generate (e.g., 'deliver_by_truck')
            
        Returns:
            MethodDef containing the generated method definition
            
        Raises:
            PDDLGenerationError: If generation fails after max retries
            ValueError: If task_name or method_name is invalid
        
        Example:
            # Generate a specific method
            method = modeler.generate_method(task_desc, 'deliver', 'deliver_by_truck')
            print(f"Subtasks: {[s.name for s in method.subtasks]}")
            
            # Regenerate if not satisfied
            method = modeler.generate_method(task_desc, 'deliver', 'deliver_by_truck')
        """
        dynamic_ctx = self._dynamic_context("methods")
        task_signature = self.builder.get_task_signature(task_name)
        
        existing_methods = list(self.builder.method_names)
        
        method_prompt = f"""{'=' * 80}
TASK: Define the method '{method_name}' to decompose task '{task_name}'
{'=' * 80}

{task_description}

{'=' * 80}
EXAMPLE OUTPUT:
{'=' * 80}

{{
{{
  "name": "deliver_by_truck",
  "task": "deliver_package",
  "task_args": ["?pkg", "?dest"],
  "parameters": [
    {{"name": "?pkg", "type": "package"}},
    {{"name": "?dest", "type": "location"}},
    {{"name": "?truck", "type": "truck"}}
  ],
  "preconditions": [{{"predicate": "available", "args": ["?truck"], "negated": false}}],
  "subtasks": [
    {{"name": "load_truck", "args": ["?pkg", "?truck"]}},
    {{"name": "drive_truck", "args": ["?truck", "?dest"]}},
    {{"name": "unload_truck", "args": ["?pkg", "?truck"]}}
  ],
  "description": "Deliver package using a truck: load, drive, unload"
}}

{'=' * 80}
CONTEXT:
{'=' * 80}
{dynamic_ctx}

TASK TO DECOMPOSE: {self._format_task(task_name)}
EXISTING METHODS (don't duplicate): {existing_methods}

{'=' * 80}
REQUIREMENTS:
{'=' * 80}

1. Method name MUST be: '{method_name}'
2. "task" field MUST be: '{task_name}'
3. task_args count MUST match task signature shown above
4. Subtask names must be from ACTIONS or TASKS shown in context
5. All arguments in subtasks must be method parameters
6. Include description explaining the decomposition strategy"""
        
        def validate_single_method(result: MethodDef):
            """Validate and add this single method."""
            method = result
            # Verify name matches
            if _normalize_name(method.name) != _normalize_name(method_name):
                raise ValueError(f"Expected method name '{method_name}', got '{method.name}'")
            # Verify task matches
            if _normalize_name(method.task) != _normalize_name(task_name):
                raise ValueError(f"Expected task '{task_name}', got '{method.task}'")
            # Add to builder (will validate predicates, actions, tasks, etc.)
            if _normalize_name(method.name) not in self.builder.method_names:
                self.builder.add_method(
                    method.name, method.task, method.task_args,
                    method.parameters, method.preconditions, method.subtasks
                )
        
        self._log(LogLevel.DEBUG, f"  Generating method '{method_name}' for task '{task_name}'...")
        
        result = self._generate_and_validate(
            method_prompt, MethodDef, validate_single_method,
            context_info=dynamic_ctx,
            step_name=f"Method_{method_name}_for_{task_name}"
        )
        
        # Add to raw_methods tracking
        self.raw_methods.append(result)
        
        return result
    
    def generate_objects(self, task_description: str) -> ObjectList:
        """
        Generate problem objects (instances of types).
        
        Objects are the concrete entities that exist in the problem instance.
        Each object has a name and must be assigned a type from the domain.
        """
        dynamic_ctx = self._dynamic_context("objects")
        
        prompt = f"""{'=' * 80}
TASK: Define all concrete OBJECTS (instances) for this problem
{'=' * 80}

{task_description}

{'=' * 80}
EXAMPLE OUTPUT:
{'=' * 80}

{{
  "objects": [
    {{"name": "block_a", "type": "block"}},
    {{"name": "block_b", "type": "block"}},
    {{"name": "table", "type": "surface"}},
    {{"name": "robot_arm", "type": "manipulator"}}
  ]
}}

{'=' * 80}
CONTEXT:
{'=' * 80}
{dynamic_ctx}

{'=' * 80}
REQUIREMENTS:
{'=' * 80}

1. Object types must match the available types shown in context
2. Include ALL entities mentioned in problem description
3. Use lowercase names with underscores (e.g., 'block_a', 'truck_1')"""
        
        def validate_objects(result: ObjectList):
            """Full validation."""
            for obj in result.objects:
                if _normalize_name(obj.name) not in self.builder.object_names:
                    self.builder.add_object(obj.name, obj.type)
            
            # Warn about missing entities from analysis
            if self.raw_analysis:
                generated_names = {obj.name for obj in result.objects}
                missing = [entity for entity in self.raw_analysis.entities 
                          if entity not in generated_names]
                if missing:
                    sample = missing[:DEFAULT_WARNING_SAMPLE_SIZE]
                    print(f"  ⚠ Warning: {len(missing)} entities from analysis not generated as objects: {sample}...")
        
        self.raw_objects = self._generate_and_validate(
            prompt, ObjectList, validate_objects,
            context_info=dynamic_ctx, 
            step_name="Objects"
        )
        return self.raw_objects
    
    def generate_initial_state(self, task_description: str) -> InitialState:
        """
        Generate the initial state of the problem.
        
        The initial state describes all facts that are TRUE at the start.
        Facts not listed are assumed FALSE (closed-world assumption).
        """
        dynamic_ctx = self._dynamic_context("initial_state")
        
        prompt = f"""{'=' * 80}
TASK: Define the INITIAL STATE (facts that are TRUE at the start)
{'=' * 80}

{task_description}

{'=' * 80}
EXAMPLE OUTPUT:
{'=' * 80}

{{
  "facts": [
    {{"predicate": "on", "args": ["block_a", "block_b"], "negated": false}},
    {{"predicate": "on", "args": ["block_b", "table"], "negated": false}},
    {{"predicate": "clear", "args": ["block_a"], "negated": false}},
    {{"predicate": "arm_empty", "args": [], "negated": false}}
  ]
}}

{'=' * 80}
CONTEXT:
{'=' * 80}
{dynamic_ctx}

{'=' * 80}
REQUIREMENTS (CLOSED-WORLD ASSUMPTION):
{'=' * 80}

1. Only list facts that are TRUE at the start
2. Facts NOT listed are assumed FALSE (don't need to specify)
3. Do NOT include negated facts (negated must be false for all)
4. Use only predicates and objects shown in context"""
        
        def validate_initial(result: InitialState):
            for fact in result.facts:
                if fact.negated:
                    raise ValueError(
                        f"Initial state fact '{fact.predicate}' cannot be negated. "
                        "Only list facts that are TRUE."
                    )
                self.builder.add_initial_fact(fact.predicate, fact.args, value=True)
        
        self.raw_initial = self._generate_and_validate(
            prompt, InitialState, validate_initial,
            context_info=dynamic_ctx, 
            step_name="Initial State"
        )
        return self.raw_initial
    
    def generate_goal(self, task_description: str) -> Union[GoalState, HTNGoal]:
        """
        Generate the goal state (classical PDDL) or HTN goal (task-based).
        
        Delegates to appropriate implementation based on domain type.
        """
        if self.is_htn:
            return self.generate_htn_goal(task_description)
        else:
            return self.generate_classical_goal(task_description)
    
    def generate_htn_goal(self, task_description: str) -> HTNGoal:
        """
        Generate HTN goal specification (tasks to accomplish).
        
        For HTN domains: list of high-level tasks that must be accomplished.
        """
        dynamic_ctx = self._dynamic_context("goal")
        
        prompt = f"""{'=' * 80}
TASK: Define the HTN GOAL (high-level tasks to accomplish)
{'=' * 80}

{task_description}

{'=' * 80}
EXAMPLE OUTPUT:
{'=' * 80}

{{
  "tasks": [
    {{"name": "deliver_package", "args": ["package_1", "destination_city"]}},
    {{"name": "return_vehicle", "args": ["truck_1", "depot"]}}
  ]
}}

{'=' * 80}
CONTEXT:
{'=' * 80}
{dynamic_ctx}

{'=' * 80}
REQUIREMENTS:
{'=' * 80}

1. Use only task names shown in context (check exact names)
2. Arguments must be object names from available objects
3. These are TOP-LEVEL goals that will be decomposed by methods"""
        
        def validate_htn_goal(result: HTNGoal):
            self.builder.set_htn_goal([(task.name, task.args) for task in result.tasks])
        
        self.raw_goal = self._generate_and_validate(
            prompt, HTNGoal, validate_htn_goal,
            context_info=dynamic_ctx, 
            step_name="HTN Goal"
        )
        return self.raw_goal
    
    def generate_classical_goal(self, task_description: str) -> GoalState:
        """
        Generate classical PDDL goal state.
        
        For classical PDDL: list of facts that must be true in the final state.
        """
        dynamic_ctx = self._dynamic_context("goal")
        
        # Include sample of initial state to help differentiate
        initial_sample = self._format_initial_state_sample()
        
        prompt = f"""{'=' * 80}
TASK: Define the GOAL STATE (desired facts at the end)
{'=' * 80}

{task_description}

{'=' * 80}
EXAMPLE OUTPUT:
{'=' * 80}

{{
  "facts": [
    {{"predicate": "on", "args": ["block_a", "table"], "negated": false}},
    {{"predicate": "clear", "args": ["block_a"], "negated": false}},
    {{"predicate": "holding", "args": ["block_b"], "negated": true}}
  ]
}}

{'=' * 80}
CONTEXT:
{'=' * 80}
{dynamic_ctx}{initial_sample}

{'=' * 80}
REQUIREMENTS:
{'=' * 80}

1. Use only predicates and objects shown in context
2. Goal must be DIFFERENT from initial state
3. Use negated: true if a fact must be FALSE in the goal
4. Only include facts that matter for the goal (not all facts)"""
        
        def validate_goal(result: GoalState):
            for fact in result.facts:
                self.builder.add_goal_fact(fact.predicate, fact.args, fact.negated)
        
        self.raw_goal = self._generate_and_validate(
            prompt, GoalState, validate_goal,
            context_info=dynamic_ctx, 
            step_name="Goal State"
        )
        return self.raw_goal
    
    def _format_initial_state_sample(self) -> str:
        """Format a sample of initial facts for context in goal generation."""
        if not self.raw_initial:
            return ""
        sample_facts = [f"({f.predicate} {' '.join(f.args)})" 
                       for f in self.raw_initial.facts[:5]]
        return f"\nSAMPLE OF INITIAL STATE (for reference): {sample_facts}"
    
    def generate_full_model(self, task_description: str, show_metrics: bool = DEFAULT_SHOW_METRICS) -> Tuple[str, int]:
        """
        Generate complete PDDL/HTN model - EXAMPLE USE CASE of the framework.
        
        This method demonstrates a standard workflow for generating a complete model.
        It uses the modular API (generate_action_names, generate_action, etc.) internally.
        
        For custom workflows, you can use the individual methods directly:
        
        EXAMPLE - Custom workflow with action filtering:
            modeler = PDDLModeler(client, is_htn=False)
            modeler.analyze_problem(task_desc)
            modeler.generate_types(task_desc)
            modeler.generate_predicates(task_desc)
            modeler.generate_objects(task_desc)
            
            # Get action names, filter, then generate
            action_names = modeler.generate_action_names(task_desc)
            for name in action_names:
                if name != 'unwanted_action':
                    modeler.generate_action(task_desc, name)
            
            modeler.generate_initial_state(task_desc)
            modeler.generate_goal(task_desc)
        
        EXAMPLE - HTN with custom method generation:
            modeler = PDDLModeler(client, is_htn=True)
            # ... generate types, predicates, objects, actions ...
            modeler.generate_tasks(task_desc)
            
            for task_name in modeler.builder.task_names:
                method_names = modeler.generate_method_names(task_desc, task_name)
                for method_name in method_names:
                    try:
                        modeler.generate_method(task_desc, task_name, method_name)
                    except Exception as e:
                        print(f"Skipping {method_name}: {e}")
            
            modeler.generate_initial_state(task_desc)
            modeler.generate_goal(task_desc)
        
        Args:
            task_description: Natural language description of the planning problem
            show_metrics: Whether to print metrics summary at the end (default True)
        
        Returns:
            Tuple of (status_message, exit_code) where exit_code 0 means success
        """
        # Start metrics tracking (if enabled)
        if self.track_metrics:
            self.metrics.start_generation()
        
        self._log(LogLevel.INFO,  "\n============= Model Generation Started =============")
        self._log(LogLevel.INFO, f"Using the full modular generation workflow")
        
        try:
            # ═══════════════════════════════════════════════════════════════════
            # PHASE 0: Problem Analysis (optional but recommended)
            # ═══════════════════════════════════════════════════════════════════
            if self.use_analysis:
                self._log(LogLevel.INFO, "\n📋 Phase 0: Analyzing problem...")
                self.analyze_problem(task_description)
                self._log(LogLevel.INFO, f"  ✓ {len(self.raw_analysis.entities)} entities identified")
            
            # ═══════════════════════════════════════════════════════════════════
            # PHASE 1: Domain Structure (types → predicates → objects)
            # ═══════════════════════════════════════════════════════════════════
            self._log(LogLevel.INFO, "\nPhase 1. Generating Domain:")
            
            self._log(LogLevel.INFO, "\n🏷️  Step 1: Generating Types...")
            self.generate_types(task_description)
            self._log(LogLevel.INFO, f"  ✓ {len(self.builder.type_names)} types: {self.builder.type_names}")
            
            self._log(LogLevel.INFO, "\n🔗 Step 2: Generating predicates...")
            self.generate_predicates(task_description)
            self._log(LogLevel.INFO, f"   {self.builder.predicate_names}")
            
            self._log(LogLevel.INFO, "\n📦 Step 3: Generating objects...")
            self.generate_objects(task_description)
            self._log(LogLevel.INFO, f"  ✓ {len(self.builder.object_names)} objects")
            
            # ═══════════════════════════════════════════════════════════════════
            # PHASE 2: Actions (using modular API)
            # ═══════════════════════════════════════════════════════════════════
            self._log(LogLevel.INFO, "\nPhase 2. Generating actions:")
            
            # Step 1: Ask LLM what actions are needed
            self._log(LogLevel.INFO, "\n⚡ Step 1: Generating names...")
            action_names = self.generate_action_names(task_description)
            self._log(LogLevel.INFO, f"  → Action names: {action_names}")
            
            # Step 2: Generate each action individually
            self._log(LogLevel.INFO, "\n🚀 Step 2: Generating action definitions...")
            for action_name in action_names:
                self._log(LogLevel.INFO, f"  • Generating action '{action_name}'...")
                self.generate_action(task_description, action_name)
                self._log(LogLevel.INFO, f"    ✓ Action '{action_name}' generated")
            self._log(LogLevel.INFO, f"  ✓ {len(self.builder.action_names)} actions: {list(self.builder.action_names)}")
            
            # ═══════════════════════════════════════════════════════════════════
            # PHASE 3: HTN Tasks & Methods (if HTN mode)
            # ═══════════════════════════════════════════════════════════════════
            if self.is_htn:
                self._log(LogLevel.INFO, "\nPhase 3. Generating HTN Tasks & Methods:")
                
                self._log(LogLevel.INFO, "\n🎯 Step 1: Generating HTN tasks...")
                self.generate_tasks(task_description)
                self._log(LogLevel.INFO, f"  ✓ {len(self.builder.task_names)} tasks: {list(self.builder.task_names)}")
                
                self._log(LogLevel.INFO, "\n🔧 Step 2: Generating methods for each task...")
                for task_name in self.builder.task_names:
                    self._log(LogLevel.INFO, f"\n🛠 Generating methods names for task '{task_name}'...")
                    method_names = self.generate_method_names(task_description, task_name)
                    self._log(LogLevel.INFO, f"  → Methods for '{task_name}': {method_names}")
                    
                    for method_name in method_names:
                        self._log(LogLevel.INFO, f"  • Generating method '{method_name}' for task '{task_name}'...")
                        self.generate_method(task_description, task_name, method_name)
                        self._log(LogLevel.INFO, f"    ✓ Method '{method_name}' generated")
                
                self._log(LogLevel.INFO, f"  ✓ {len(self.raw_methods)} methods total")
            
            # ═══════════════════════════════════════════════════════════════════
            # PHASE 4: Problem Instance (initial state → goal)
            # ═══════════════════════════════════════════════════════════════════
            self._log(LogLevel.INFO, "\nPhase 4. Generating Problem Instance:")
            
            self._log(LogLevel.INFO, "\n🏁 Step 1: Generating initial state...")
            self.generate_initial_state(task_description)
            self._log(LogLevel.INFO, f"  ✓ {len(self.raw_initial.facts)} initial facts")
            
            self._log(LogLevel.INFO, "\n🎯 Step 2: Generating goal...")
            self.generate_goal(task_description)
            goal_count = len(self.raw_goal.tasks) if self.is_htn else len(self.raw_goal.facts)
            self._log(LogLevel.INFO, f"  ✓ {'tasks' if self.is_htn else 'facts'}: {goal_count}")
            
            # ═══════════════════════════════════════════════════════════════════
            # PHASE 5: Final Validation
            # ═══════════════════════════════════════════════════════════════════
            self._log(LogLevel.INFO, "\n✅ Phase 5. Final validation with Unified Planning...")
            is_valid, msg = self.builder.validate()
            self._log(LogLevel.INFO, f"  {'✓' if is_valid else '✗'} {msg}")
            
            # End metrics tracking (if enabled)
            if self.track_metrics:
                self.metrics.end_generation()
            
            # Show metrics summary if enabled and requested
            if self.track_metrics and show_metrics:
                self._log(LogLevel.INFO, "\n" + self.metrics.summary())
            
            # Save log file
            self._save_log_file()
            
            return ("Success", 0) if is_valid else (f"Validation failed: {msg}", 1)
                
        except KeyboardInterrupt:
            self._log(LogLevel.INFO, "\n⚠ Generation aborted by user")
            self.metrics.end_generation()
            self._save_log_file()
            return "Aborted by user", 2
            
        except Exception as e:
            # End metrics tracking even on failure (if enabled)
            if self.track_metrics:
                self.metrics.end_generation()
            
            if self.track_metrics and show_metrics:
                self._log(LogLevel.INFO, "\n" + self.metrics.summary())
            
            self._save_log_file()
            return f"Error: {e}\n\n{traceback.format_exc()}", 1
    
    def get_pddl(self, domain_name: str = DEFAULT_DOMAIN_NAME, problem_name: str = DEFAULT_PROBLEM_NAME) -> Tuple[str, str]:
        """Get PDDL domain and problem strings."""
        return self.builder.to_pddl(domain_name, problem_name)
    
    # -------------------------------------------------------------------------
    # Metrics - Access generation performance data
    # -------------------------------------------------------------------------
    
    def get_metrics_summary(self) -> Optional[str]:
        """
        Get a formatted summary of generation metrics.
        
        Returns:
            String summary of metrics if tracking is enabled, None otherwise.
            
        Example:
            ```python
            modeler = PDDLModeler(client, track_metrics=True)
            modeler.generate_full_model(task_desc)
            print(modeler.get_metrics_summary())
            ```
        """
        if not self.track_metrics:
            return None
        return self.metrics.summary()
    
    def get_metrics_dict(self) -> Optional[dict]:
        """
        Get metrics as a dictionary for programmatic access.
        
        Returns:
            Dictionary with metrics data if tracking is enabled, None otherwise.
            
        Example:
            ```python
            metrics_data = modeler.get_metrics_dict()
            if metrics_data:
                print(f"Total time: {metrics_data['total_duration_seconds']}s")
                print(f"Success rate: {metrics_data['overall_success_rate']:.1%}")
            ```
        """
        if not self.track_metrics:
            return None
        return self.metrics.to_dict()
    
    # -------------------------------------------------------------------------
    # Serialization - Export/Import model data
    # -------------------------------------------------------------------------
    
    def to_dict(self) -> dict:
        """
        Export complete model as a dictionary.
        
        The output includes all model components plus metadata about the model type
        and generation settings. This can be used for:
        - Saving to JSON/YAML files
        - Transferring between systems
        - Debugging and inspection
        - Reconstructing the model later via from_dict()
        
        Returns:
            Dictionary with all model components and metadata
        """
        def dump_list(obj, attr):
            return [x.model_dump() for x in getattr(obj, attr, [])] if obj else []
        
        return {
            # Metadata
            "metadata": {
                "is_htn": self.is_htn,
                "use_analysis": self.use_analysis,
                "model_name": self.client.model_name if self.client else None,
                "structured_output_mode": self.client.mode.name if self.client else None,
            },
            # Domain components
            "domain": {
                "types": dump_list(self.raw_types, "types"),
                "predicates": dump_list(self.raw_predicates, "predicates"),
                "actions": dump_list(self.raw_actions, "actions"),
                "tasks": dump_list(self.raw_tasks, "tasks") if self.is_htn else [],
                "methods": [m.model_dump() for m in self.raw_methods] if self.is_htn else [],
            },
            # Problem components
            "problem": {
                "objects": dump_list(self.raw_objects, "objects"),
                "initial_state": self.raw_initial.model_dump() if self.raw_initial else None,
                "goal": self.raw_goal.model_dump() if self.raw_goal else None,
            },
            # Analysis (optional)
            "analysis": self.raw_analysis.model_dump() if self.raw_analysis else None,
            # Generation metrics
            "metrics": self.metrics.to_dict() if self.metrics.steps else None,
        }
    
    def to_json(self, indent: int = DEFAULT_JSON_INDENT) -> str:
        """
        Export model as formatted JSON string.
        
        Args:
            indent: Number of spaces for indentation (default 2)
            
        Returns:
            JSON string representation of the model
        """
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    def save_json(self, filepath: str, indent: int = DEFAULT_JSON_INDENT) -> None:
        """
        Save model to a JSON file.
        
        Args:
            filepath: Path to the output JSON file
            indent: Number of spaces for indentation (default 2)
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=indent, ensure_ascii=False)
    
    def save_pddl(self, domain_path: str, problem_path: str, 
                  domain_name: str = DEFAULT_DOMAIN_NAME, 
                  problem_name: str = DEFAULT_PROBLEM_NAME) -> None:
        """
        Save PDDL domain and problem to separate files.
        
        Args:
            domain_path: Path to save domain PDDL file
            problem_path: Path to save problem PDDL file
            domain_name: Name for the domain (default: 'domain')
            problem_name: Name for the problem (default: 'problem')
        """
        domain_pddl, problem_pddl = self.get_pddl(domain_name, problem_name)
        
        with open(domain_path, 'w', encoding='utf-8') as f:
            f.write(domain_pddl)
        
        with open(problem_path, 'w', encoding='utf-8') as f:
            f.write(problem_pddl)
    
    @classmethod
    def from_dict(cls, data: dict, client: Optional[StructuredOutputClient] = None) -> "PDDLModeler":
        """
        Reconstruct a PDDLModeler from a dictionary (e.g., loaded from JSON).
        
        This allows you to reload a previously saved model without regenerating
        it from natural language. The model will be fully validated through
        Unified Planning during reconstruction.
        
        Args:
            data: Dictionary in the format produced by to_dict()
            client: Optional client for future generation (not needed for reconstruction)
            
        Returns:
            Reconstructed and validated PDDLModeler instance
            
        Raises:
            ValueError: If the data is invalid or fails validation
        """
        # Extract metadata
        metadata = data.get("metadata", {})
        is_htn = metadata.get("is_htn", False)
        use_analysis = metadata.get("use_analysis", False)
        
        # Create modeler instance
        modeler = cls(
            client=client,
            is_htn=is_htn,
            use_analysis=use_analysis,
            max_retries=1  # No retries needed for reconstruction
        )
        
        # Get domain and problem data
        domain = data.get("domain", data)  # Support both old and new format
        problem = data.get("problem", data)
        
        # Reconstruct types
        types_data = domain.get("types", [])
        if types_data:
            modeler.raw_types = TypeList(types=[TypeDef(**t) for t in types_data])
            for t in modeler.raw_types.types:
                modeler.builder.add_type(t.name, t.parent)
        
        # Reconstruct predicates
        predicates_data = domain.get("predicates", [])
        if predicates_data:
            modeler.raw_predicates = PredicateList(predicates=[PredicateDef(**p) for p in predicates_data])
            for p in modeler.raw_predicates.predicates:
                params = [ParameterDef(**param) for param in p.model_dump()["parameters"]]
                modeler.builder.add_predicate(p.name, params)
        
        # Reconstruct actions
        actions_data = domain.get("actions", [])
        if actions_data:
            modeler.raw_actions = ActionList(actions=[ActionDef(**a) for a in actions_data])
            for a in modeler.raw_actions.actions:
                dump = a.model_dump()
                params = [ParameterDef(**p) for p in dump["parameters"]]
                preconds = [FactDef(**f) for f in dump["preconditions"]]
                effects = [FactDef(**f) for f in dump["effects"]]
                modeler.builder.add_action(a.name, params, preconds, effects)
        
        # Reconstruct HTN components
        if is_htn:
            tasks_data = domain.get("tasks", [])
            if tasks_data:
                modeler.raw_tasks = TaskList(tasks=[TaskDef(**t) for t in tasks_data])
                for t in modeler.raw_tasks.tasks:
                    params = [ParameterDef(**p) for p in t.model_dump()["parameters"]]
                    modeler.builder.add_task(t.name, params)
            
            methods_data = domain.get("methods", [])
            for m_data in methods_data:
                m = MethodDef(**m_data)
                modeler.raw_methods.append(m)
                dump = m.model_dump()
                params = [ParameterDef(**p) for p in dump["parameters"]]
                preconds = [FactDef(**f) for f in dump["preconditions"]]
                subtasks = [SubtaskDef(**s) for s in dump["subtasks"]]
                modeler.builder.add_method(
                    m.name, m.task, m.task_args, params, preconds, subtasks
                )
        
        # Reconstruct objects
        objects_data = problem.get("objects", [])
        if objects_data:
            modeler.raw_objects = ObjectList(objects=[ObjectDef(**o) for o in objects_data])
            for o in modeler.raw_objects.objects:
                modeler.builder.add_object(o.name, o.type)
        
        # Reconstruct initial state
        initial_data = problem.get("initial_state")
        if initial_data:
            modeler.raw_initial = InitialState(**initial_data)
            for fact in modeler.raw_initial.facts:
                modeler.builder.add_initial_fact(fact.predicate, fact.args, not fact.negated)
        
        # Reconstruct goal
        goal_data = problem.get("goal")
        if goal_data:
            modeler.raw_goal = GoalState(**goal_data)
            for fact in modeler.raw_goal.facts:
                modeler.builder.add_goal_fact(fact.predicate, fact.args, fact.negated)
        
        # Reconstruct analysis (just store, no builder action needed)
        analysis_data = data.get("analysis")
        if analysis_data:
            modeler.raw_analysis = ProblemAnalysis(**analysis_data)
        
        # Validate the reconstructed model
        is_valid, msg = modeler.builder.validate()
        if not is_valid:
            raise ValueError(f"Reconstructed model failed validation: {msg}")
        
        return modeler
    
    @classmethod
    def load_json(cls, filepath: str, client: Optional[StructuredOutputClient] = None) -> "PDDLModeler":
        """
        Load a PDDLModeler from a JSON file.
        
        Args:
            filepath: Path to the JSON file
            client: Optional client for future generation
            
        Returns:
            Reconstructed PDDLModeler instance
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data, client)


# =============================================================================
# BACKWARD COMPATIBILITY ALIAS
# =============================================================================

# Alias for backward compatibility
UltimateModeler = PDDLModeler


# =============================================================================
# MAIN - Interactive experimentation with NL tasks
# =============================================================================

def _parse_cli_arguments():
    """
    Parse command-line arguments with support for both flags and defaults.
    
    Returns:
        tuple: (use_defaults, model_name, mode, host, log_level, is_htn, use_analysis, think_mode, interactive_mode, make_plan)
    
    Supported flags (long and short versions):
        -d, --defaults:      Use defaults for all unspecified values
        -m, --model NAME:    Ollama model name
        -M, --mode NUM:      Backend mode (1=instructor, 2=outlines, 3=raw_ollama)
        --host URL:          Ollama server URL (no short flag to avoid conflict)
        -l, --log-level LEVEL: Logging level (0|quiet, 1|info, 2|debug, 3|trace)
        -h, --htn:           Enable HTN mode
        -nh, --no-htn:       Disable HTN mode
        -a, --analysis:      Enable problem analysis
        -na, --no-analysis:  Disable problem analysis
        -t, --think:         Enable think mode (two-step reasoning)
        -nt, --no-think:     Disable think mode
        -i, --iterative:     Enable interactive mode (pauses at each step)
        -ni, --no-iterative: Disable interactive mode
        -p, --plan:          Execute planner after generation
        -np, --no-plan:      Don't execute planner after generation
    
    Examples:
        python UltimateModeler.py -d
        python UltimateModeler.py -d -m llama3.1
        python UltimateModeler.py -m llama3.1 -H
        python UltimateModeler.py -m qwen2.5 -l 3 -a
        python UltimateModeler.py -d -t -i
        python UltimateModeler.py -d -M 2 -l info -H -nt -na -i
        python UltimateModeler.py -d -M 2 -l trace -H -t -nh -na -ni
        python UltimateModeler.py -d -p
    """
    import sys
    
    # Defaults
    use_defaults = False
    model_name = None
    mode = None
    host = None
    log_level = None
    is_htn = None
    use_analysis = None
    think_mode = None
    interactive_mode = None
    make_plan = None
    
    # Parse flags
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        
        if arg in ["--defaults", "-d"]:
            use_defaults = True
            i += 1
        elif arg in ["--model", "-m"]:
            if i + 1 < len(sys.argv):
                model_name = sys.argv[i + 1]
                i += 2
            else:
                print("⚠ --model requires a value")
                i += 1
        elif arg in ["--mode", "-M"]:
            if i + 1 < len(sys.argv):
                try:
                    mode = int(sys.argv[i + 1])
                    i += 2
                except ValueError:
                    print(f"⚠ --mode requires a number (1-3), got '{sys.argv[i + 1]}'")
                    i += 2
            else:
                print("⚠ --mode requires a value")
                i += 1
        elif arg == "--host":
            if i + 1 < len(sys.argv):
                host = sys.argv[i + 1]
                i += 2
            else:
                print("⚠ --host requires a value")
                i += 1
        elif arg in ["--log-level", "-l"]:
            if i + 1 < len(sys.argv):
                level_input = sys.argv[i + 1]
                try:
                    # Try numeric first
                    log_level = int(level_input)
                    i += 2
                except ValueError:
                    # Try name (QUIET, INFO, DEBUG, TRACE)
                    level_map = {"quiet": 0, "info": 1, "debug": 2, "trace": 3}
                    normalized = level_input.lower()
                    if normalized in level_map:
                        log_level = level_map[normalized]
                        i += 2
                    else:
                        print(f"⚠ --log-level requires a number (0-3) or name (quiet, info, debug, trace), got '{level_input}'")
                        i += 2
            else:
                print("⚠ --log-level requires a value")
                i += 1
        elif arg in ["--htn", "-h"]:
            is_htn = True
            i += 1
        elif arg in ["--no-htn", "-nh"]:
            is_htn = False
            i += 1
        elif arg in ["--analysis", "-a"]:
            use_analysis = True
            i += 1
        elif arg in ["--no-analysis", "-na"]:
            use_analysis = False
            i += 1
        elif arg in ["--think", "-t"]:
            think_mode = True
            i += 1
        elif arg in ["--no-think", "-nt"]:
            think_mode = False
            i += 1
        elif arg in ["--iterative", "-i"]:
            interactive_mode = True
            i += 1
        elif arg in ["--no-iterative", "-ni"]:
            interactive_mode = False
            i += 1  
        elif arg in ["--plan", "-p"]:
            make_plan = True
            i += 1
        elif arg in ["--no-plan", "-np"]:
            make_plan = False
            i += 1
        else:
            i += 1
    
    return use_defaults, model_name, mode, host, log_level, is_htn, use_analysis, think_mode, interactive_mode, make_plan


# =============================================================================
# PLANNER EXECUTION - Call a planner to solve the generated model
# =============================================================================

def execute_planner(modeler: "PDDLModeler") -> Tuple[bool, str]:
    """
    Execute a planner to solve the generated PDDL/HTN model.
    
    Uses Unified Planning's OneshotPlanner to find a solution to the problem.
    For classical PDDL problems, it uses PDDL-based planners (e.g., FastDownward).
    For HTN problems, it uses HTN planners.
    
    Args:
        modeler: PDDLModeler instance with a generated model
        
    Returns:
        Tuple[success: bool, message: str]:
        - If planning succeeds: (True, "Plan found with N steps:\n...")
        - If planning fails: (False, "No plan found: reason")
        - If error: (False, "Error: details")
    """
    try:
        print("\n" + "="*80)
        print("EXECUTING PLANNER")
        print("="*80)
        
        problem = modeler.builder.problem
        
        # Try to find a plan
        print(f"Problem type: {'HTN' if modeler.is_htn else 'Classical PDDL'}")
        print("Searching for a plan...")
        
        with OneshotPlanner(problem_kind=problem.kind) as planner:
            result = planner.solve(problem)
            
            # Add planner info
            print(f"\tPlanner used: {planner}")
            print(f"\tEngine: {result.engine_name if hasattr(result, 'engine_name') else '(UNKNOWN)'}\n")
            # message += f"Planning time: {result.planning_time:.2f} seconds\n\n"
            
            if 'SOLVED' in result.status.name:
                plan = result.plan
                
                if modeler.is_htn:
                    # For HTN, show the hierarchical plan
                    print(f"\t✓ PLAN FOUND ({result.status.name})\n\n")
                    print(f"\tPlan type: Hierarchical\n")
                    print(f"\tStatus: {result.status.name}\n")
                    print(f"\tPlan length: {len(plan.actions) if hasattr(plan, 'actions') else 'N/A'} actions\n\n")
                    
                    print("\tAction sequence:\n")
                    if hasattr(plan, 'actions'):
                        for i, action in enumerate(plan.actions, 1):
                            print(f"  {i}. {action}")
                    elif hasattr(plan, '__str__'):
                        print(str(plan))
                    else:
                        print("  (Plan details not available)\n")
                        
                else:
                    # For classical PDDL, show step-by-step actions
                    print(f"\t✓ PLAN FOUND ({result.status.name})\n\n")
                    print(f"\tPlan type: Sequential\n")
                    print(f"\tStatus: {result.status.name}\n")
                    print(f"\tPlan length: {len(plan.actions)} steps\n\n")
                    
                    print("\tAction sequence:\n")
                    for i, action in enumerate(plan.actions, 1):
                        print(f"  {i}. {action}")

                return True, f"Plan found with {len(plan.actions)} steps"
                
            elif 'UNSOLVABLE' in result.status.name:
                print(f"\t❌ NO PLAN FOUND (Problem is unsolvable)\n")
                print(f"\tStatus: {result.status.name}")
                print("The problem has no solution given the current domain.")
                print("Check:")
                print("  - Initial state has all required facts")
                print("  - Goal is achievable with available actions")
                print("  - Preconditions are satisfiable")
                return False, 'No plan found: Problem is unsolvable'
                
            else:
                print(f"\t⚠ PLANNING FAILED\n\n")
                print(f"\tStatus: {result.status.name}\n")
                if hasattr(result, 'reason'):
                    print(f"\tReason: {result.reason}\n")
                return False , f'No plan found. Unknown status: {result.status.name}'
                
    except Exception as e:
        error_msg += f"❌ PLANNER ERROR\n\n"
        error_msg += f"Error: {str(e)}\n\n"
        error_msg += "This may be due to:\n"
        error_msg += "  - Missing planner (e.g., FastDownward not installed)\n"
        error_msg += "  - Invalid problem formulation\n"
        error_msg += "  - Unsupported problem type\n"
        error_msg += f"\nFull error:\n{traceback.format_exc()}\n"
        return False, error_msg


def main():
    """Interactive PDDL generator for experimenting with natural language task descriptions.
    
    Provides:
    - Multiple structured output backends (instructor, outlines, raw ollama)
    - Classical PDDL and HTN/HDDL generation
    - Validation via Unified Planning
    
    Usage:
        python UltimateModeler.py [FLAGS]
    
    FLAGS (all optional, can be combined):
        -d, --defaults:        Use defaults for all unspecified values
        -m, --model NAME:      Ollama model (e.g., llama3.1, qwen2.5)
        -M, --mode NUM:        Backend: 1=instructor, 2=outlines, 3=raw_ollama
        --host URL:            Ollama server URL (no short flag)
        -l, --log-level LEVEL: Logging: 0|quiet, 1|info, 2|debug, 3|trace
        -h, --htn:             Enable HTN mode
        -nh, --no-htn:         Disable HTN mode
        -a, --analysis:        Enable problem analysis
        -na, --no-analysis:    Disable problem analysis
        -t, --think:           Enable think mode (two-step reasoning)
        -nt, --no-think:       Disable think mode
        -i, --iterative:       Enable interactive mode (pauses at each step)
        -ni, --no-iterative:   Disable interactive mode
    
    Examples:
        python UltimateModeler.py                    # Full interactive prompts
        python UltimateModeler.py -d                 # All defaults, no prompts
        python UltimateModeler.py -d -m llama3.1    # Defaults + custom model
        python UltimateModeler.py -m qwen2.5 -H -t  # Custom model, HTN + think mode
        python UltimateModeler.py -d -m llama3.1 -l 2 -t  # Defaults with overrides
        python UltimateModeler.py -d -a -H          # Defaults with analysis + HTN
        python UltimateModeler.py -d --no-iterative # Single task only

    Requirements:
        - Ollama running: ollama serve
        - Model available: ollama pull llama3.1
        - Base: pip install ollama unified-planning pydantic
        - For instructor mode: pip install instructor
        - For outlines mode: pip install outlines openai
    """
    try:
        print("\n" + "=" * 80)
        print("PDDL Model Generator")
        print("=" * 80)
        print("\nGenerates PDDL/HDDL from natural language using Ollama + Unified Planning.\n")
            
        # Check base dependencies
        try:
            import ollama as _ollama_check  # noqa: F401
            import unified_planning as _up_check  # noqa: F401
            del _ollama_check, _up_check
        except ImportError:
            print("❌ Missing base dependency. Install with:")
            print("   pip install ollama unified-planning pydantic")
            return 1
        
        # Parse CLI arguments
        use_defaults, cli_model, cli_mode, cli_host, cli_log_level, cli_is_htn, cli_use_analysis, cli_think_mode, cli_interactive, cli_make_plan = _parse_cli_arguments()
        
        model_name = cli_model
        host = cli_host or DEFAULT_OLLAMA_HOST
        selected_mode = None
        if cli_mode is not None:
            selected_mode = StructuredOutputMode(cli_mode)
        elif use_defaults:
            selected_mode = StructuredOutputMode[DEFAULT_STRUCTURED_OUTPUT_MODE.upper()]
            print(f" - Using default output mode: {selected_mode.name}")
        else:
            print("Select structured output backend:")
            print("  1. INSTRUCTOR - Pydantic validation + automatic retry (requires: instructor)")
            print("  2. OUTLINES   - Grammar-constrained generation (requires: outlines)")
            print("  3. RAW_OLLAMA - Native Ollama JSON schema (no extra dependencies)")
            mode_choice = input(f"\nSelect (1-3, default={DEFAULT_STRUCTURED_OUTPUT_MODE}): ").strip() or None
            
            if mode_choice in [str(member.value) for member in StructuredOutputMode]:
                selected_mode = StructuredOutputMode(int(mode_choice))
            else:
                selected_mode = StructuredOutputMode[DEFAULT_STRUCTURED_OUTPUT_MODE.upper()]
                print(f"Invalid mode choice '{mode_choice}', using {selected_mode.name}")
        
        # Check mode-specific dependencies
        if selected_mode == StructuredOutputMode.INSTRUCTOR:
            try:
                import instructor as _instructor_check  # noqa: F401
                del _instructor_check
            except ImportError:
                print("❌ INSTRUCTOR mode requires: pip install instructor")
                return 1
        elif selected_mode == StructuredOutputMode.OUTLINES:
            try:
                import outlines as _outlines_check  # noqa: F401
                del _outlines_check
            except ImportError:
                print("❌ OUTLINES mode requires: pip install outlines openai")
                return 1
        
        # Select log level
        if cli_log_level is not None:
            selected_log_level = LogLevel(cli_log_level)
        elif use_defaults:
            selected_log_level = LogLevel[DEFAULT_LOG_LEVEL.upper()]
            print(f" - Using default log level: {selected_log_level.name}")
        else:
            print("\nSelect logging level:")
            print("  0. QUIET  - Only errors and final result")
            print("  1. INFO   - Phase progress (default)")
            print("  2. DEBUG  - Detailed step info")
            print("  3. TRACE  - Full prompts and responses")
            log_choice = input(f"\nSelect (0-3, default={DEFAULT_LOG_LEVEL.upper()}): ").strip() or None
            
            if log_choice in [str(member.value) for member in LogLevel]:
                selected_log_level = LogLevel(int(log_choice))
            else:
                selected_log_level = LogLevel[DEFAULT_LOG_LEVEL.upper()]
                print(f"Invalid log choice '{log_choice}', using default: {selected_log_level.name}")
    
        # Select model
        if cli_model is not None:
            model_name = cli_model
        elif use_defaults:
            model_name = DEFAULT_OLLAMA_MODEL
            print(f" - Using default model: {model_name}")
        else:
            model_name = input(f"\nOllama model (default={DEFAULT_OLLAMA_MODEL}): ").strip() or DEFAULT_OLLAMA_MODEL
        
        # Create client
        print(f"\nConnecting to Ollama ({selected_mode.name} mode)...")
        try:
            client = create_structured_client(
                model=model_name,
                host=host,
                mode=selected_mode,
                max_retries=3
            )
            print(f"✓ Connected using {selected_mode.name} backend")
        except Exception as e:
            print(f"❌ Cannot connect: {e}")
            print("\nMake sure Ollama is running: ollama serve")
            return 1
        
        # Main interaction loop
        while True: 
            print("\n" + "="*80)
            print("STARTING NEW TASK GENERATION")
            print("="*80)                       
            print("Options:")
            print("  1. Generate from input task description")
            print("  2. Use example task (PlanBench blocksworld-1)")
            print("  3. Use debug task description that is predefined for testing")
            print("  4. Use custom task from file")
            print("  Q. Quit")
            
            choice = input(f"\nSelect (default={DEFAULT_TASK_INPUT_OPTION}): ").strip() or DEFAULT_TASK_INPUT_OPTION
            
            if choice.lower() == "q":
                print(f"[Option {choice}] Exiting the program.")
                print("\nGoodbye!")
                break
            
            if choice == "1":
                print(f"[Option {choice}] Input custom task description.")
                print("\nEnter task description (Enter twice to finish):")
                lines = []
                while True:
                    line = input()
                    if line == "":
                        if lines and lines[-1] == "":
                            break
                        lines.append("")
                    else:
                        lines.append(line)
                task_description = "\n".join(lines[:-1])
            
            elif choice == "2":
                task_description = """
I am playing with a set of blocks where I need to arrange the blocks into stacks. Here are the actions I can do

Pick up a block
Unstack a block from on top of another block
Put down a block
Stack a block on top of another block

I have the following restrictions on my actions:
I can only pick up or unstack one block at a time.
I can only pick up or unstack a block if my hand is empty.
I can only pick up a block if the block is on the table and the block is clear. A block is clear if the block has no other blocks on top of it and if the block is not picked up.
I can only unstack a block from on top of another block if the block I am unstacking was really on top of the other block.
I can only unstack a block from on top of another block if the block I am unstacking is clear.
Once I pick up or unstack a block, I am holding the block.
I can only put down a block that I am holding.
I can only stack a block on top of another block if I am holding the block being stacked.
I can only stack a block on top of another block if the block onto which I am stacking the block is clear.
Once I put down or stack a block, my hand becomes empty.
Once you stack a block on top of a second block, the second block is no longer clear.

As initial conditions I have that, the red block is clear, the yellow block is clear, the hand is empty, the red block is on top of the blue block, the yellow block is on top of the orange block, the blue block is on the table and the orange block is on the table.
My goal is to have that the orange block is on top of the red block.
                """.strip()
                print(f"\n[Option {choice}] Example blocksworld task from PlanBench.")
            elif choice == "3":
                task_description = """
I am defining a simple planning model (domain and problem) for a HTN simple version of the classical PDDL problem blocksworld.
I want you to help me create the HTN representation for this domain and a specific problem instance.
I already have defined all the components of the domain and problem that I need, so I dont want you to invent anything new.
I want you to generate the PDDL domain and problem exactly based on the following specifications:

The Types are:
- block
The Predicates are: 
- on(x, y) meaning block x is on block y
- ontable(x) meaning block x is on the table
- clear(x) meaning block x has nothing on top of it
- handempty meaning the hand is empty.
- holding(x) meaning the hand is holding block x.
The Actions are:
- pick-up(x - block): Precondition: clear(x), ontable(x), handempty. Effects: not ontable(x), not clear(x), not handempty, holding(x).
- put-down(x - block): Precondition: holding(x). Effects: ontable(x), clear(x), handempty, not holding(x).
- stack(x - block, y - block): Precondition: holding(x), clear(y). Effects: on(x, y), clear(x), handempty, not holding(x), not clear(y).
- unstack(x - block, y - block): Precondition: on(x, y), clear(x), handempty. Effects: holding(x), clear(y), not on(x, y), not clear(x), not handempty.
The Tasks are:
- move(x - block, y - block): This is a compound task that represents moving block x to block y.
The Methods are:
- move-from-block(x - block, y - block, z - block): 
    Task: move(x, y) 
    Precondition: on(x, y), clear(x), clear(z) 
    Subtasks:
    1. unstack(x, z)
    2. stack(x, y)
- move-from-table(x - block, y - block): 
    Task: move(x, y)
    Precondition: ontable(x), clear(x), clear(y)
    Subtasks:
    1. pick-up(x)
    2. stack(x, y)
The Objects are:
- A of type block
- B of type block
- C of type block
The Initial State is:
- A is on the table
- B is on the table
- C is on top of A
The Goal Task is:
- move(C, B)
                """.strip()
                print(f"[Option {choice}] Debug task description for testing.")
            elif choice == "4":
                print(f"[Option {choice}] Custom task description from file.")
                file_path = input("\nEnter the path to the file: ").strip()
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        task_description = f.read()
                    print(f"\n[INFO] Using task description from {file_path}.")
                except Exception as e:
                    print(f"❌ Cannot read file: {e}")
                    continue
            else:
                print("❌ Invalid choice.")
                continue
            
            if not task_description.strip():
                print("❌ No task description provided.")
                continue
            
            # HTN choice
            # Use CLI args if provided, otherwise use defaults or ask
            if cli_is_htn is not None:
                is_htn = cli_is_htn
            elif use_defaults:
                is_htn = DEFAULT_IS_HTN
            else:
                is_htn = input(f"\nGenerate HTN model? (y/n, default={DEFAULT_IS_HTN}): ").strip().lower()
                if is_htn == "y":
                    is_htn = True
                elif is_htn == "n":
                    is_htn = False
                else:
                    is_htn = DEFAULT_IS_HTN
                    print(f"Invalid input. Using default HTN setting: {is_htn}")
            
            if cli_use_analysis is not None:
                use_analysis = cli_use_analysis
            elif use_defaults:
                use_analysis = DEFAULT_USE_ANALYSIS
            else:
                use_analysis = input(f"Use problem analysis? (y/n, default={DEFAULT_USE_ANALYSIS}): ").strip().lower()
                if use_analysis == "y":
                    use_analysis = True
                elif use_analysis == "n":
                    use_analysis = False
                else:
                    use_analysis = DEFAULT_USE_ANALYSIS
                    print(f"Invalid input. Using default analysis setting: {use_analysis}")
            
            # Think mode
            if cli_think_mode is not None:
                think_mode = cli_think_mode
            elif use_defaults:
                think_mode = DEFAULT_THINK_MODE
            else:
                think_mode = input(f"Enable think mode (two-step reasoning)? (y/n, default={DEFAULT_THINK_MODE}): ").strip().lower()
                if think_mode == "y":
                    think_mode = True
                elif think_mode == "n":
                    think_mode = False
                else:
                    think_mode = DEFAULT_THINK_MODE
                    print(f"Invalid input. Using default think mode setting: {think_mode}")
            
            # Interactive mode
            if cli_interactive is not None:
                interactive = cli_interactive
            elif use_defaults:
                interactive = DEFAULT_INTERACTIVE_MODE
            else:
                interactive = input(f"Enable interactive mode (pauses at each step)? (y/n, default={DEFAULT_INTERACTIVE_MODE}): ").strip().lower()
                if interactive == "y":
                    interactive = True
                elif interactive == "n":
                    interactive = False
                else:
                    interactive = DEFAULT_INTERACTIVE_MODE
                    print(f"Invalid input. Using default interactive mode setting: {interactive}")
            
            if use_defaults:
                print(f" - Using defaults: HTN={is_htn}, Analysis={use_analysis}, ThinkMode={think_mode}, Interactive={interactive}")
            
            # Generate
            print("\n" + "=" * 80)
            print(f"Generating {'HTN' if is_htn else 'PDDL'} ({selected_mode.name} backend)")
            if use_analysis:
                print(" - Using problem analysis for better context")
            if think_mode:
                print(" - Using think mode (two-step reasoning)")
            print("=" * 80)
            print()
            
            modeler = PDDLModeler(client, is_htn=is_htn, max_retries=3, log_level=selected_log_level, use_analysis=use_analysis, think_mode=think_mode, interactive=interactive)
            
            try:
                status_message, exit_code = modeler.generate_full_model(task_description)
                if exit_code == 0:
                    status_prompt = " ✓ Generation Completed "
                elif exit_code == 1:
                    status_prompt = " ❌ Generation Failed "
                elif exit_code == 2:
                    status_prompt = " ⚠ Generation Aborted "
                else:
                    status_prompt = f"❓ Unknown Exit Code: {exit_code} "
                print()
                print()
                print("=" * 30 + status_prompt + "=" * 30)
                print(status_message)
                
                domain_pddl, problem_pddl = modeler.get_pddl()
                   
                # View options
                print("\nView:")
                print("  1. Full PDDL (domain + problem)")
                print("  2. Domain only")
                print("  3. Problem only")
                print("  4. Summary only")
                view = input("Select (1-4, default=4): ").strip() or "4"
                
                if view in ["1", "2"]:
                    print("\n" + "-" * 80)
                    print("DOMAIN PDDL")
                    print("-" * 80)
                    print(domain_pddl)
                
                if view in ["1", "3"]:
                    print("\n" + "-" * 80)
                    print("PROBLEM PDDL")
                    print("-" * 80)
                    print(problem_pddl)
                
                # Summary
                d = modeler.to_dict()
                print("\n" + "-" * 80)
                print("Summary")
                print("-" * 80)
                
                # Metadata
                metadata = d.get('metadata', {})
                mode_type = "HTN (Hierarchical)" if metadata.get('is_htn') else "Classical PDDL"
                model_name = metadata.get('model_name', 'Unknown')
                output_mode = metadata.get('structured_output_mode', 'Unknown')
                print(f"Model: {model_name} | Type: {mode_type} | Response Mode: {output_mode}")
                
                # Analysis
                analysis = d.get('analysis')
                if analysis is not None:
                    print(f"Analysis: {len(analysis.get('entities', []))} entities, {len(analysis.get('entity_types', []))} types identified")
                
                # Domain components
                domain = d.get('domain', {})
                print(f"Types: {len(domain.get('types', []))}, Predicates: {len(domain.get('predicates', []))}, Actions: {len(domain.get('actions', []))}")
                if is_htn:
                    print(f"Tasks: {len(domain.get('tasks', []))}, Methods: {len(domain.get('methods', []))}")
                
                # Problem components
                problem = d.get('problem', {})
                objects = problem.get('objects') if problem.get('objects') is not None else []
                initial_state = problem.get('initial_state') if problem.get('initial_state') is not None else {}
                initial_facts = initial_state.get('facts') if initial_state.get('facts') is not None else []
                goal = problem.get('goal') if problem.get('goal') is not None else {}
                goal_facts = goal.get('facts') if goal.get('facts') is not None else []
                goal_tasks = goal.get('tasks') if goal.get('tasks') is not None else []
                if is_htn:
                    print(f"Objects: {len(objects)}, Initial State facts: {len(initial_facts)}, Goal tasks: {len(goal_tasks)}")
                    if len(goal_facts) > 0:
                        print(f"[WARNING] Goal facts present in HTN mode: {len(goal_facts)}")
                else:
                    print(f"Objects: {len(objects)}, Initial State facts: {len(initial_facts)}, Goal facts: {len(goal_facts)}")
                    if len(goal_tasks) > 0:
                        print(f"[WARNING] Goal tasks present in PDDL mode: {len(goal_tasks)}")
                        
                # Save
                if input(f"\nSave to file? (y/n, default=n): ").strip().lower() == "y":
                    df = input("Domain file (default=domain.pddl): ").strip() or "domain.pddl"
                    pf = input("Problem file (default=problem.pddl): ").strip() or "problem.pddl"
                    with open(df, "w", encoding="utf-8") as f:
                        f.write(domain_pddl)
                    with open(pf, "w", encoding="utf-8") as f:
                        f.write(problem_pddl)
                    print(f"✓ Saved to {df} and {pf}")
                
                # Execute planner if generation was successful
                if exit_code == 0:
                    if cli_make_plan is not None:
                        make_plan = cli_make_plan
                    elif use_defaults:
                        make_plan = DEFAULT_EXECUTE_PLANNER
                    else:
                        make_plan = input(f"\nExecute planner to solve the problem? (y/n, default={DEFAULT_EXECUTE_PLANNER}): ").strip().lower()
                        if make_plan == "y":
                            make_plan = True
                        elif make_plan == "n":
                            make_plan = False
                        else:
                            make_plan = DEFAULT_EXECUTE_PLANNER
                            print(f"Invalid input. Using default execute planner setting: {make_plan}")
                    
                    if make_plan:
                        plan_success, plan_message = execute_planner(modeler)
                        print("\n" + plan_message)
                        
                        if plan_success:
                            print("✓ Planning succeeded!")
                        else:
                            print("⚠ Planning did not find a solution.")
            
            except PDDLGenerationError as e:
                # PDDLGenerationError includes traceback in its message
                print(f"\n{e}")
                if input("\nTry again? (y/n, default=y): ").strip().lower() != "n":
                    continue
                else:
                    raise
            except Exception as e:
                print(f"\n❌ Unexpected error: {e}")
                print(traceback.format_exc())
                if input("\nTry again? (y/n, default=y): ").strip().lower() != "n":
                    continue
                else:
                    raise
    
    except KeyboardInterrupt:
        print("\n\nInterrupted.")
        return 0
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print(traceback.format_exc())
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())