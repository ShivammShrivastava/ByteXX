"""
AI Agents Module — SatQuery AI
================================
Provides a multi-tool orchestration agent for satellite image analysis.

Components:
  - Router   : decides fast CNN vs heavy VLM
  - Planner  : breaks queries into execution steps
  - Tools    : standard callable tool interface
  - Executor : parallel/sequential step execution
  - Verifier : low-confidence re-analysis
  - Merger   : structured report assembly
  - Agent    : top-level SatAgent orchestrator

Usage:
    from AI_Agents.agent import sat_agent
    report = await sat_agent.run(image, question)
"""

from AI_Agents.agent import SatAgent, sat_agent

__all__ = ["SatAgent", "sat_agent"]
