# Causal Suggesters

LLM-agnostic causal inference suggester library. Replicates [pywhyllm](https://github.com/py-why/pywhyllm)'s four suggester classes but works with **any LLM** supported by LiteLLM — Claude, GPT, Gemini, Llama via Ollama, and others.

## Architecture

```
causal_suggesters/
├── llm_backend.py              # LiteLLM + instructor adapter
├── enums.py                    # RelationshipStrategy enum
├── model_suggester.py          # Suggest causal DAG structure
├── identification_suggester.py # Suggest identification strategies
├── validation_suggester.py     # Critique and validate DAGs
└── augmented_model_suggester.py # RAG-enhanced model suggester
```

All intelligence lives in prompts. The original pywhyllm uses `guidance` to call OpenAI + parse structured output. This library replaces that with **LiteLLM + instructor** (Pydantic-validated structured output), which works across all providers without fragile JSON string parsing.

## Installation

### Core (required)

```bash
pip install litellm instructor pydantic
```

### For AugmentedModelSuggester only (optional)

```bash
pip install sentence-transformers chromadb
```

## API Key Setup

Set your provider's API key as an environment variable before running.

**Windows (Command Prompt):**
```cmd
set ANTHROPIC_API_KEY=sk-ant-...
set OPENAI_API_KEY=sk-...
set GEMINI_API_KEY=...
```

**Windows (PowerShell):**
```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."
$env:OPENAI_API_KEY = "sk-..."
$env:GEMINI_API_KEY = "..."
```

**Linux / macOS:**
```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
export GEMINI_API_KEY=...
```

**Or set in Python directly:**
```python
import os
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."
```

Ollama runs locally and needs no API key — just have the Ollama server running.

## Supported Models

| Provider | Model string example | API key env var |
|---|---|---|
| Anthropic | `"claude-opus-4-6"` | `ANTHROPIC_API_KEY` |
| OpenAI | `"gpt-5.4"` | `OPENAI_API_KEY` |
| Google | `"gemini/gemini-3.1-pro-preview"` | `GEMINI_API_KEY` |
| Ollama (local) | `"ollama/llama4"` | None |

See [LiteLLM docs](https://docs.litellm.ai/docs/providers) for the full list of 100+ supported providers.

## The Four Suggesters

### 1. ModelSuggester

Suggests causal DAG structure: domain expertise, directed edges, confounders, and pairwise relationship analysis.

**Methods:**

| Method | Returns | Purpose |
|---|---|---|
| `suggest_domain_expertise(all_factors)` | `List[str]` | Relevant expertise areas for these variables |
| `suggest_relationships(all_factors, domain_expertises, strategy)` | `List[List[str]]` | Directed causal edges `[[cause, effect], ...]` |
| `suggest_confounders(treatment, outcome, all_factors, domain_expertises)` | `List[str]` | Variables that causally affect both treatment and outcome |
| `suggest_pairwise_relationship(var_a, var_b, domain_expertises)` | `PairwiseRelationshipResponse` | Direction and rationale for a single variable pair |

### 2. IdentificationSuggester

Suggests causal identification strategies using the backdoor criterion, instrumental variables, mediators, and the frontdoor criterion.

**Methods:**

| Method | Returns | Purpose |
|---|---|---|
| `suggest_backdoor(treatment, outcome, all_factors, domain_expertises)` | `List[str]` | Valid backdoor adjustment set |
| `suggest_ivs(treatment, outcome, all_factors, domain_expertises)` | `List[str]` | Valid instrumental variables |
| `suggest_mediators(treatment, outcome, all_factors, domain_expertises)` | `List[str]` | Variables on the causal path |
| `suggest_frontdoor(treatment, outcome, all_factors, domain_expertises)` | `List[str]` | Valid frontdoor adjustment set |

### 3. ValidationSuggester

Critiques proposed causal DAGs and suggests falsification strategies.

**Methods:**

| Method | Returns | Purpose |
|---|---|---|
| `critique_graph(all_factors, dag_edges, domain_expertises, strategy)` | `CritiqueResponse` | Edge-level critiques with severity ratings |
| `suggest_latent_confounders(treatment, outcome, all_factors, domain_expertises)` | `List[LatentConfounder]` | Unmeasured confounders and which variables they affect |
| `suggest_negative_controls(treatment, outcome, all_factors, domain_expertises)` | `List[NegativeControl]` | Negative control exposures and outcomes for falsification |

### 4. AugmentedModelSuggester

Extends `ModelSuggester` with RAG (Retrieval-Augmented Generation). Retrieves relevant causal knowledge from a local vector store before prompting the LLM. Uses sentence-transformers for embeddings (fully local, no API dependency).

**Additional methods:**

| Method | Purpose |
|---|---|
| `load_knowledge_base(entries)` | Populate the vector store with causal knowledge |

Works without any knowledge base loaded (falls back to standard ModelSuggester behavior).

## Complete Example: Titanic Dataset

This example walks through the full causal inference pipeline — from building a DAG to identifying estimation strategies to validating the graph — using the Titanic survival dataset.

```python
import os
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."  # or your provider's key

from causal_suggesters import (
    ModelSuggester,
    IdentificationSuggester,
    ValidationSuggester,
    AugmentedModelSuggester,
    RelationshipStrategy,
)

# Choose your model
MODEL = "claude-opus-4-6"

# --- Define the causal analysis problem ---
all_factors = [
    "Pclass",     # Passenger class (1st / 2nd / 3rd)
    "Sex",        # Male / Female
    "Age",        # Age in years
    "SibSp",      # Number of siblings/spouses aboard
    "Parch",      # Number of parents/children aboard
    "Fare",       # Ticket price paid
    "Embarked",   # Port of embarkation (Cherbourg / Queenstown / Southampton)
]
treatment = "Pclass"
outcome = "Survived"


# ================================================================
# STEP 1: Build the Causal DAG with ModelSuggester
# ================================================================
print("=" * 60)
print("STEP 1: Building Causal DAG")
print("=" * 60)

modeler = ModelSuggester(MODEL)

# 1a. What domain expertise is relevant?
expertise = modeler.suggest_domain_expertise(all_factors + [outcome])
print("\nDomain Expertise Areas:")
for e in expertise:
    print(f"  - {e}")

# 1b. Suggest directed causal edges
edges = modeler.suggest_relationships(
    all_factors + [outcome],
    expertise,
    RelationshipStrategy.Pairwise,
)
print("\nProposed Causal Edges:")
for cause, effect in edges:
    print(f"  {cause} -> {effect}")

# 1c. Identify confounders for our treatment-outcome pair
confounders = modeler.suggest_confounders(
    treatment, outcome, all_factors, expertise
)
print(f"\nConfounders of {treatment} -> {outcome}:")
for c in confounders:
    print(f"  - {c}")

# 1d. Analyze a specific variable pair
pw = modeler.suggest_pairwise_relationship("Sex", "Survived", expertise)
print(f"\nPairwise Analysis: Sex vs Survived")
print(f"  Has relationship: {pw.has_relationship}")
print(f"  Direction: {pw.cause} -> {pw.effect}")
print(f"  Rationale: {pw.rationale}")


# ================================================================
# STEP 2: Identify Estimation Strategies with IdentificationSuggester
# ================================================================
print("\n" + "=" * 60)
print("STEP 2: Causal Identification")
print("=" * 60)

identifier = IdentificationSuggester(MODEL)

# 2a. Backdoor adjustment set
backdoor = identifier.suggest_backdoor(
    treatment, outcome, all_factors, expertise
)
print(f"\nBackdoor Adjustment Set for {treatment} -> {outcome}:")
print(f"  Condition on: {backdoor}")

# 2b. Instrumental variables
ivs = identifier.suggest_ivs(treatment, outcome, all_factors, expertise)
print(f"\nInstrumental Variables:")
print(f"  IVs: {ivs}")

# 2c. Mediators
mediators = identifier.suggest_mediators(
    treatment, outcome, all_factors, expertise
)
print(f"\nMediators on {treatment} -> {outcome} path:")
print(f"  Mediators: {mediators}")

# 2d. Frontdoor adjustment set
frontdoor = identifier.suggest_frontdoor(
    treatment, outcome, all_factors, expertise
)
print(f"\nFrontdoor Adjustment Set:")
print(f"  Variables: {frontdoor}")


# ================================================================
# STEP 3: Validate the Proposed DAG with ValidationSuggester
# ================================================================
print("\n" + "=" * 60)
print("STEP 3: DAG Validation")
print("=" * 60)

validator = ValidationSuggester(MODEL)

# 3a. Critique the graph structure
critique = validator.critique_graph(
    all_factors + [outcome], edges, expertise
)
print(f"\nOverall Assessment: {critique.overall_assessment}")
print("\nEdge-Level Critiques:")
for c in critique.critiques:
    print(f"  [{c.severity.upper()}] {c.edge}: {c.issue}")
    print(f"    Fix: {c.suggestion}")

# 3b. Identify potential latent confounders
latent = validator.suggest_latent_confounders(
    treatment, outcome, all_factors, expertise
)
print(f"\nLatent (Unmeasured) Confounders:")
for lc in latent:
    print(f"  {lc.name}")
    print(f"    Affects: {lc.affects}")
    print(f"    Rationale: {lc.rationale}")

# 3c. Suggest negative controls for falsification
neg_controls = validator.suggest_negative_controls(
    treatment, outcome, all_factors, expertise
)
print(f"\nNegative Controls for Falsification:")
for nc in neg_controls:
    print(f"  [{nc.type.upper()}] {nc.variable}")
    print(f"    Rationale: {nc.rationale}")


# ================================================================
# STEP 4 (Optional): RAG-Enhanced Suggestions
# ================================================================
print("\n" + "=" * 60)
print("STEP 4: RAG-Enhanced Model Suggestion")
print("=" * 60)

# AugmentedModelSuggester works even without a knowledge base,
# but is more useful when you load domain-specific causal knowledge.
augmented = AugmentedModelSuggester(MODEL)

# Load Titanic-specific causal knowledge
augmented.load_knowledge_base([
    {
        "id": "titanic_class_survival",
        "text": "Higher passenger class (1st class) had priority access to lifeboats, "
                "directly increasing survival probability on the Titanic.",
        "metadata": {"source": "domain_knowledge"},
    },
    {
        "id": "titanic_women_children",
        "text": "The 'women and children first' protocol during Titanic evacuation "
                "meant Sex and Age directly caused differences in survival rates.",
        "metadata": {"source": "domain_knowledge"},
    },
    {
        "id": "titanic_class_fare",
        "text": "Passenger class determined ticket fare. First class tickets cost "
                "significantly more than third class. Fare did not independently "
                "cause survival — it was a proxy for class.",
        "metadata": {"source": "domain_knowledge"},
    },
    {
        "id": "titanic_embarkation_class",
        "text": "Port of embarkation (Embarked) was associated with passenger class "
                "due to geographic and socioeconomic factors. Cherbourg had more "
                "first-class passengers.",
        "metadata": {"source": "domain_knowledge"},
    },
    {
        "id": "titanic_deck_location",
        "text": "Cabin location (deck) was determined by passenger class and affected "
                "proximity to lifeboats, mediating the class-survival relationship.",
        "metadata": {"source": "domain_knowledge"},
    },
])

# RAG-augmented relationship suggestions
augmented_edges = augmented.suggest_relationships(
    all_factors + [outcome],
    expertise,
    RelationshipStrategy.Pairwise,
)
print("\nRAG-Augmented Causal Edges:")
for cause, effect in augmented_edges:
    print(f"  {cause} -> {effect}")
```

### Expected Output Structure

The exact output depends on the LLM, but you can expect results like:

```
STEP 1: Building Causal DAG

Domain Expertise Areas:
  - Maritime disaster survival analysis
  - Socioeconomic stratification in early 20th century transatlantic travel
  - Evacuation protocol and lifeboat access patterns
  - Demographics and family structure effects on survival behavior

Proposed Causal Edges:
  Pclass -> Fare
  Pclass -> Survived
  Sex -> Survived
  Age -> Survived
  SibSp -> Survived
  Parch -> Survived
  Embarked -> Pclass

Confounders of Pclass -> Survived:
  - Sex
  - Age

Pairwise Analysis: Sex vs Survived
  Has relationship: True
  Direction: Sex -> Survived
  Rationale: Women were given priority in lifeboat boarding under the
  "women and children first" protocol, creating a direct causal path
  from Sex to Survived.

STEP 2: Causal Identification

Backdoor Adjustment Set for Pclass -> Survived:
  Condition on: ['Sex', 'Age']

Instrumental Variables:
  IVs: ['Embarked']

Mediators on Pclass -> Survived path:
  Mediators: ['Fare']

STEP 3: DAG Validation

Overall Assessment: The proposed DAG captures the main causal structure
reasonably well but has some issues with missing edges and potential
confounding paths.

Edge-Level Critiques:
  [MEDIUM] ['Embarked', 'Pclass']: Direction may be reversed — embarkation
  port was chosen based on class/geography, not the other way around.
    Fix: Consider Embarked as a common effect of geographic origin rather
    than a cause of Pclass.

Latent (Unmeasured) Confounders:
  Wealth / Socioeconomic Status
    Affects: ['Pclass', 'Fare', 'Survived']
    Rationale: Wealth determines class, fare, and potentially survival
    through connections and influence beyond ticket class alone.
```

## RelationshipStrategy Enum

Controls how `suggest_relationships` and `critique_graph` approach the analysis:

| Value | Description |
|---|---|
| `RelationshipStrategy.Pairwise` | Determine causal direction between variable pairs |
| `RelationshipStrategy.Parent` | Find causes of a specific variable |
| `RelationshipStrategy.Child` | Find effects of a specific variable |
| `RelationshipStrategy.Confounder` | Find common causes of treatment + outcome |

## Response Schemas

All structured outputs are Pydantic models. Key ones:

```python
# From model_suggester.py
class PairwiseRelationshipResponse(BaseModel):
    cause: Optional[str]       # The causing variable, or null
    effect: Optional[str]      # The effect variable, or null
    has_relationship: bool
    rationale: str

# From validation_suggester.py
class EdgeCritique(BaseModel):
    edge: Optional[List[str]]  # [cause, effect] or null for graph-level
    issue: str
    severity: Literal["low", "medium", "high"]
    suggestion: str

class CritiqueResponse(BaseModel):
    critiques: List[EdgeCritique]
    overall_assessment: str

class LatentConfounder(BaseModel):
    name: str
    affects: List[str]         # Which observed variables it affects
    rationale: str

class NegativeControl(BaseModel):
    variable: str
    type: Literal["exposure", "outcome"]
    rationale: str
```

## Comparison with pywhyllm

| Feature | pywhyllm | causal_suggesters |
|---|---|---|
| LLM backend | `guidance` + OpenAI only | LiteLLM (100+ providers) |
| Structured output | String parsing | Pydantic + instructor (validated, retries) |
| Embedding for RAG | OpenAI embeddings API | Local sentence-transformers |
| Vector store | OpenAI-dependent | ChromaDB (local, no API) |
| Model support | GPT-3.5 / GPT-4 | Claude, GPT, Gemini, Mistral, Ollama, etc. |
| API compatibility | Original | Drop-in equivalent method signatures |
