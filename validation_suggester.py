from pydantic import BaseModel, Field
from typing import List, Literal, Optional
from .llm_backend import LLMBackend
from .enums import RelationshipStrategy

_SYSTEM = """You are a rigorous causal model critic. You identify structural flaws,
missing confounders, implausible edges, and suggest empirical falsification strategies
for proposed causal DAGs."""


class EdgeCritique(BaseModel):
    edge: Optional[List[str]] = Field(
        description="[cause, effect] or null for graph-level issues"
    )
    issue: str
    severity: Literal["low", "medium", "high"]
    suggestion: str


class CritiqueResponse(BaseModel):
    critiques: List[EdgeCritique]
    overall_assessment: str


class LatentConfounder(BaseModel):
    name: str
    affects: List[str] = Field(
        description="Which observed variables this latent factor affects"
    )
    rationale: str


class LatentConfoundersResponse(BaseModel):
    latent_confounders: List[LatentConfounder]


class NegativeControl(BaseModel):
    variable: str
    type: Literal["exposure", "outcome"]
    rationale: str


class NegativeControlsResponse(BaseModel):
    negative_controls: List[NegativeControl]


class ValidationSuggester:
    """
    Critiques causal DAGs and suggests validation/falsification strategies.
    Drop-in equivalent of pywhyllm.suggesters.validation_suggester.ValidationSuggester.
    """

    def __init__(self, model: str = "gpt-4o", **kwargs):
        self.llm = LLMBackend(model, **kwargs)

    def critique_graph(
        self,
        all_factors: List[str],
        dag_edges: List[List[str]],
        domain_expertises: List[str],
        strategy: RelationshipStrategy = RelationshipStrategy.Pairwise,
    ) -> CritiqueResponse:
        prompt = f"""Variables: {all_factors}
Proposed DAG edges (cause -> effect): {dag_edges}
Domain expertise: {domain_expertises}
Review strategy: {strategy.value}

Critique this causal graph. For each problematic edge or structural issue, provide:
- The specific edge (or null for graph-level issues)
- What is wrong or suspicious
- Severity (low/medium/high)
- How to fix it"""
        return self.llm.complete_structured(prompt, CritiqueResponse, _SYSTEM)

    def suggest_latent_confounders(
        self,
        treatment: str,
        outcome: str,
        all_factors: List[str],
        domain_expertises: List[str],
    ) -> List[LatentConfounder]:
        prompt = f"""Treatment: {treatment}, Outcome: {outcome}
Observed variables: {all_factors}
Domain expertise: {domain_expertises}

What unmeasured or latent variables likely confound this treatment-outcome relationship?
For each, specify which observed variables it affects."""
        return self.llm.complete_structured(
            prompt, LatentConfoundersResponse, _SYSTEM
        ).latent_confounders

    def suggest_negative_controls(
        self,
        treatment: str,
        outcome: str,
        all_factors: List[str],
        domain_expertises: List[str],
    ) -> List[NegativeControl]:
        prompt = f"""Treatment: {treatment}, Outcome: {outcome}
Variables: {all_factors}
Domain expertise: {domain_expertises}

Suggest negative control variables for falsification testing.

Negative control EXPOSURES: variables that share confounding structure with {treatment}
but have NO causal pathway to {outcome}. Example: if studying effect of smoking on
lung cancer, a negative control exposure might be "left-handedness" (shares some
confounders but cannot cause cancer).

Negative control OUTCOMES: outcomes that share confounding structure with {outcome}
but cannot be caused by {treatment}. Example: studying effect of a new drug on a disease;
a negative control outcome might be an injury unrelated to the drug's mechanism.

These are used to detect unmeasured confounding."""
        return self.llm.complete_structured(
            prompt, NegativeControlsResponse, _SYSTEM
        ).negative_controls
