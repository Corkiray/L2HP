# L2HP : Framework to Hierachical Planning (HTN) Model Generation Driven by LLMs

## General Information

This framework is built upon [**l2p: LLM-driven Planning Model library kit**](https://github.com/AI-Planning/l2p) to support LLM-driven model generation for Hierarchical Planning (HP).  
We designed **L2HP** with a focus on **generality and extensibility**, to facilitate and encourage the implementation of future LLM-driven HP systems. 

For more details, please refer to the paper: [**Towards a General Framework for HTN Modeling with LLMs**](https://openreview.net/pdf?id=0PZQxcAQfF).


📌 Full library documentation will be provided in the future. Sorry for the inconvenience.  

**Note**: This framework incorporates several modules from the original [l2p](https://github.com/AI-Planning/l2p) library. We highly recommend visiting their [repository](https://github.com/AI-Planning/l2p), as well as checking their [documentation](https://marcustantakoun.github.io/l2p.github.io/) and [survey](https://arxiv.org/abs/2503.18971v1), to deepen your knowledge of Modeling-via-LLM and better understand the specific library usage.

## Usage Examples and Preliminary Experimental Results

### Building an Architecture

See [agents/nl2htn.py](https://github.com/Corkiray/L2HP/blob/main/agents/nl2htn.py) for an example.  
`NL2HTN` is a simple architecture that illustrates how to use L2HP modules to create a pipeline that transforms natural language task descriptions into automated planning models — both **classical** and **hierarchical** (i.e., domain and problem files).

### Running Experiments

See [run_experiments.py](https://github.com/Corkiray/L2HP/blob/main/run_experiments.py) for an illustrative experiment using `NL2HTN` with the **PlanBench** dataset, whose interface is included in the framework.  
This script demonstrates how to instantiate aN architecture and highlights the flexibility of the modular implementation. 

### Experimental Setup

Experiments were conducted on commit `95d5be6ce430b5516ac6041fd3b3e2917d32d712` of L2HP, with the following specifications:

- **LLM**: Google GenAI API — Package: `google-generativeai`, version: `0.8.5`  
- **Planner**: Unified Planning library — Package: `up-aries`, version: `0.4.0`  

**Note**: Further experimental considerations are available in the referenced paper: [**Towards a General Framework for HTN Modeling with LLMs**](https://openreview.net/pdf?id=0PZQxcAQfF).

## Contact
`israelpm01@ugr.es`