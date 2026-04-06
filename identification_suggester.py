from pydantic import BaseModel, Field
from typing import List
from .llm_backend import LLMBackend

_SYSTEM = """You are an expert in causal identification theory.
You apply the backdoor criterion, frontdoor criterion, and instrumental variable
conditions to suggest valid identification strategies for causal effects."""


class BackdoorResponse(BaseModel):
    backdoor_set: List[str] = Field(
        description="Variables satisfying the backdoor criterion for this treatment-outcome pair"
    )
    rationale: str


class IVResponse(BaseModel):
    ivs: List[str] = Field(
        description="Valid instrumental variables"
    )
    rationale: str


class MediatorResponse(BaseModel):
    mediators: List[str] = Field(
        description="Variables on the causal path from treatment to outcome"
    )


class FrontdoorResponse(BaseModel):
    frontdoor_set: List[str] = Field(
        description="Variables satisfying the frontdoor criterion"
    )
    rationale: str


class IdentificationSuggester:
    """
    Suggests causal identification strategies: backdoor sets, IVs, mediators, frontdoor sets.
    Drop-in equivalent of pywhyllm.suggesters.identification_suggester.IdentificationSuggester.
    """

    def __init__(self, model: str = "gpt-4o", **kwargs):
        self.llm = LLMBackend(model, **kwargs)

    def suggest_backdoor(
        self,
        treatment: str,
        outcome: str,
        all_factors: List[str],
        domain_expertises: List[str],
    ) -> List[str]:
        prompt = f"""Treatment: {treatment}, Outcome: {outcome}
All variables: {all_factors}
Domain expertise: {domain_expertises}

Suggest a valid backdoor adjustment set: variables that block all backdoor paths
from {treatment} to {outcome} without opening new paths.
Do NOT include descendants of {treatment}.
Do NOT include variables that are effects of both {treatment} and {outcome} (colliders),
as conditioning on colliders opens spurious paths."""
        return self.llm.complete_structured(
            prompt, BackdoorResponse, _SYSTEM
        ).backdoor_set

    def suggest_ivs(
        self,
        treatment: str,
        outcome: str,
        all_factors: List[str],
        domain_expertises: List[str],
    ) -> List[str]:
        prompt = f"""Treatment: {treatment}, Outcome: {outcome}
All variables: {all_factors}
Domain expertise: {domain_expertises}

Suggest valid instrumental variables (IVs). Each IV must satisfy ALL three conditions:
1. Relevance: The IV is correlated with {treatment}
2. Exclusion restriction: The IV affects {outcome} ONLY through {treatment} — no direct effect
3. Independence: The IV shares no unmeasured common causes with {outcome}

Only include variables that clearly satisfy all three conditions."""
        return self.llm.complete_structured(prompt, IVResponse, _SYSTEM).ivs

    def suggest_mediators(
        self,
        treatment: str,
        outcome: str,
        all_factors: List[str],
        domain_expertises: List[str],
    ) -> List[str]:
        prompt = f"""Treatment: {treatment}, Outcome: {outcome}
All variables: {all_factors}
Domain expertise: {domain_expertises}

Which variables lie on the causal path from {treatment} to {outcome}
(i.e., are mediators of the causal effect)?"""
        return self.llm.complete_structured(
            prompt, MediatorResponse, _SYSTEM
        ).mediators

    def suggest_frontdoor(
        self,
        treatment: str,
        outcome: str,
        all_factors: List[str],
        domain_expertises: List[str],
    ) -> List[str]:
        prompt = f"""Treatment: {treatment}, Outcome: {outcome}
All variables: {all_factors}
Domain expertise: {domain_expertises}

Suggest a frontdoor adjustment set: mediators that intercept all directed paths
from {treatment} to {outcome}, with no unblocked backdoor paths from {treatment}
to the set, and all backdoor paths from the set to {outcome} blockable by {treatment}."""
        return self.llm.complete_structured(
            prompt, FrontdoorResponse, _SYSTEM
        ).frontdoor_set
