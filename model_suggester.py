from pydantic import BaseModel, Field
from typing import List, Optional
from .llm_backend import LLMBackend
from .enums import RelationshipStrategy

_SYSTEM = """You are an expert in causal inference and causal graphical models.
You help domain experts construct accurate causal DAGs by suggesting relationships,
confounders, and domain expertise based on variable names and context."""


class DomainExpertiseResponse(BaseModel):
    expertise: List[str] = Field(description="Relevant domain expertise areas")


class RelationshipsResponse(BaseModel):
    edges: List[List[str]] = Field(
        description="List of [cause, effect] pairs representing directed causal edges"
    )


class ConfoundersResponse(BaseModel):
    confounders: List[str] = Field(
        description="Variables that causally affect both treatment and outcome"
    )


class PairwiseRelationshipResponse(BaseModel):
    cause: Optional[str] = Field(
        description="The causing variable, or null if no direct relationship"
    )
    effect: Optional[str] = Field(
        description="The effect variable, or null if no direct relationship"
    )
    has_relationship: bool
    rationale: str


class ModelSuggester:
    """
    Suggests causal structure: relationships between variables and confounders.
    Drop-in equivalent of pywhyllm.suggesters.model_suggester.ModelSuggester.

    Usage:
        modeler = ModelSuggester("claude-sonnet-4-20250514")
        modeler = ModelSuggester("gpt-4o")
        modeler = ModelSuggester("ollama/llama3.2")
    """

    def __init__(self, model: str = "gpt-4o", **kwargs):
        self.llm = LLMBackend(model, **kwargs)

    def suggest_domain_expertise(
        self,
        all_factors: List[str],
    ) -> List[str]:
        prompt = f"""Given these variables in a causal analysis: {all_factors}
What domain expertise areas would be most helpful for understanding
causal relationships among them? Be specific and actionable."""
        result = self.llm.complete_structured(prompt, DomainExpertiseResponse, _SYSTEM)
        return result.expertise

    def suggest_relationships(
        self,
        all_factors: List[str],
        domain_expertises: List[str],
        strategy: RelationshipStrategy = RelationshipStrategy.Pairwise,
    ) -> List[List[str]]:
        prompt = f"""Variables: {all_factors}
Domain expertise context: {domain_expertises}
Strategy: {strategy.value}

Suggest directed causal relationships (edges) between these variables.
Only include edges where there is a plausible direct causal mechanism.
Do NOT include spurious correlations or reverse causation.
Do NOT include variables that are effects of both treatment and outcome (colliders)."""
        result = self.llm.complete_structured(prompt, RelationshipsResponse, _SYSTEM)
        return result.edges

    def suggest_confounders(
        self,
        treatment: str,
        outcome: str,
        all_factors: List[str],
        domain_expertises: List[str],
    ) -> List[str]:
        prompt = f"""Treatment variable: {treatment}
Outcome variable: {outcome}
All available variables: {all_factors}
Domain expertise: {domain_expertises}

Which variables are confounders — i.e., causally affect BOTH {treatment} and {outcome}?
Only include variables from the provided list.
Do NOT include mediators (variables on the causal path from treatment to outcome)."""
        result = self.llm.complete_structured(prompt, ConfoundersResponse, _SYSTEM)
        return result.confounders

    def suggest_pairwise_relationship(
        self,
        var_a: str,
        var_b: str,
        domain_expertises: List[str],
    ) -> PairwiseRelationshipResponse:
        prompt = f"""Variable A: {var_a}
Variable B: {var_b}
Domain expertise: {domain_expertises}

Does {var_a} directly cause {var_b}, does {var_b} directly cause {var_a},
or is there no direct causal relationship between them?
Provide a brief rationale."""
        return self.llm.complete_structured(
            prompt, PairwiseRelationshipResponse, _SYSTEM
        )
