# Output Validation Patterns (Reference Document)

> **Note:** This is a **research/reference document** describing general output validation patterns
> for LLM applications. It catalogs Pydantic validation techniques, external libraries, and
> validation strategies that can be used with or alongside spellcrafting.
>
> For the actual spellcrafting implementation, see:
> - **Guards:** `@guard.input()` and `@guard.output()` decorators in `guard.py`
> - **Failure Strategies:** `OnFail.escalate()`, `OnFail.fallback()`, etc. in `on_fail.py`
> - **Design Decisions:** `guards.md` for what we chose to build vs. defer

## Overview

This document catalogs output validation patterns for LLM applications: structural enforcement via Pydantic, semantic validation techniques, safety filtering approaches, and quality metrics. These patterns can be used with spellcrafting's `@spell` decorator or with external validation libraries.

## Pattern Categories

This document covers patterns for:

1. **Structural validation**: Pydantic schema enforcement, retries with feedback
2. **Semantic validation**: Hallucination detection, factuality checking, consistency
3. **Safety filtering**: Toxicity detection, refusal detection, off-topic filtering
4. **Quality metrics**: Relevance scoring, completeness checking, confidence estimation

## What spellcrafting Provides vs. External Libraries

| Need | spellcrafting Provides | External Libraries |
|------|-------------------|-------------------|
| Structural validation | Pydantic return types + `retries` | Instructor |
| Retry with feedback | Built-in via PydanticAI | - |
| Semantic guards | `@guard.output()` decorator | Guardrails AI |
| Failure handling | `OnFail.escalate/fallback/custom` | - |
| Toxicity detection | - | Detoxify, LLM Guard |
| PII detection | - | Presidio, AWS Comprehend |
| Content moderation | - | OpenAI Moderation API |

See `guards.md` for the rationale behind these decisions.

---

## 1. Structural Validation

### 1.1 Pydantic Schema Enforcement

The `@spell` decorator already leverages Pydantic for output type validation via PydanticAI. The output type annotation becomes the schema contract:

```python
from pydantic import BaseModel, Field, field_validator
from typing import Literal, Annotated

class Analysis(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    key_points: list[str] = Field(min_length=1, max_length=10)

    @field_validator("key_points")
    @classmethod
    def validate_key_points(cls, v: list[str]) -> list[str]:
        # Remove empty strings
        return [p.strip() for p in v if p.strip()]

@spell(model="fast", retries=2)
def analyze_text(text: str) -> Analysis:
    """Analyze the sentiment and key points of the given text."""
    ...
```

**Key Patterns:**

| Pattern | Pydantic Feature | Use Case |
|---------|------------------|----------|
| Constrained values | `Literal`, `Enum` | Force specific categories |
| Numeric bounds | `Field(ge=, le=)` | Confidence scores, ratings |
| List constraints | `Field(min_length=, max_length=)` | Bounded collections |
| Custom validation | `@field_validator` | Business rules |
| Nested models | Composed `BaseModel` | Complex structures |

### 1.2 Retry Strategies with Feedback

PydanticAI automatically retries when validation fails. The `retries` parameter controls this:

```python
@spell(model="fast", retries=3)  # Up to 3 attempts on validation failure
def extract_entities(text: str) -> Entities:
    """Extract named entities from text."""
    ...
```

**Retry Flow:**
1. LLM generates response
2. Pydantic validates against schema
3. On failure, PydanticAI sends error feedback to LLM
4. LLM regenerates with corrected output
5. Repeat until success or max retries

**Best Practice:** Keep `retries` low (1-3). If validation fails repeatedly, the prompt or schema needs refinement.

### 1.3 Partial Output Recovery (Streaming)

Pydantic v2.10+ supports experimental partial validation for streaming:

```python
from pydantic import TypeAdapter

adapter = TypeAdapter(Analysis)

# Validate incomplete JSON during streaming
partial_result = adapter.validate_json(
    incomplete_json_string,
    experimental_allow_partial="trailing-strings"
)
```

**Integration with Spells:**

```python
from spellcrafting import spell, StreamingConfig

@spell(model="fast", streaming=StreamingConfig(partial_validation=True))
async def analyze_stream(text: str) -> Analysis:
    """Analyze with partial results available during streaming."""
    ...

# Consumer can access partial results
async for partial in analyze_stream.stream("Long text..."):
    print(f"Partial: {partial}")  # Valid partial Analysis object
```

**Caveats:**
- Experimental feature, API may change
- Last element errors are ignored during partial validation
- Full validation occurs on complete response

### 1.4 Custom Validators for LLM-Specific Constraints

Use `Annotated` with validators for complex rules:

```python
from pydantic import AfterValidator, BeforeValidator
from typing import Annotated

def no_competitor_mentions(text: str) -> str:
    """Block output mentioning competitors."""
    competitors = ["acme", "globex", "initech"]
    lower_text = text.lower()
    for competitor in competitors:
        if competitor in lower_text:
            raise ValueError(f"Output mentions competitor: {competitor}")
    return text

def clean_whitespace(text: str) -> str:
    """Normalize whitespace."""
    return " ".join(text.split())

CleanText = Annotated[str, BeforeValidator(clean_whitespace)]
SafeText = Annotated[str, AfterValidator(no_competitor_mentions)]

class Response(BaseModel):
    summary: Annotated[str, BeforeValidator(clean_whitespace), AfterValidator(no_competitor_mentions)]
    # Or compose types:
    # summary: SafeText
```

---

## 2. Semantic Validation

> **Status**: The patterns in this section are NOT implemented in spellcrafting.
> They describe external library patterns and techniques that can be used alongside
> spellcrafting's `@guard.output()` decorator. See `guards.md` for design rationale.

### 2.1 Hallucination Detection

Hallucination detection requires comparing LLM output against provided context or known facts.

**Approaches:**

| Technique | Description | When to Use |
|-----------|-------------|-------------|
| Entailment checking | NLI model checks if output follows from context | RAG systems |
| Self-consistency | Compare multiple generations for agreement | Critical outputs |
| Semantic entropy | Measure uncertainty across paraphrased outputs | Research/evaluation |
| Source citation | Require explicit citations, verify they exist | Document QA |

**Entailment-Based Validation:**

```python
from pydantic import model_validator

class GroundedAnswer(BaseModel):
    answer: str
    source_quotes: list[str]  # Direct quotes from context

    @model_validator(mode="after")
    def verify_grounding(self) -> "GroundedAnswer":
        # Each source_quote must appear in the original context
        # This validator would need access to context via validation_context
        return self

# Using validation context (Pydantic feature)
from pydantic import ValidationInfo

class GroundedAnswer(BaseModel):
    answer: str
    source_quotes: list[str]

    @field_validator("source_quotes")
    @classmethod
    def verify_quotes_exist(cls, quotes: list[str], info: ValidationInfo) -> list[str]:
        context = info.context.get("source_document", "") if info.context else ""
        for quote in quotes:
            if quote not in context:
                raise ValueError(f"Quote not found in source: {quote[:50]}...")
        return quotes
```

**Self-Consistency Checking:**

```python
async def check_consistency(spell_fn, input_data, n_samples: int = 3) -> tuple[Any, float]:
    """Run spell multiple times and measure agreement."""
    results = [await spell_fn(input_data) for _ in range(n_samples)]

    # For structured outputs, compare field values
    # For text, use embedding similarity
    consistency_score = calculate_agreement(results)

    # Return majority/best result and confidence
    return select_best(results), consistency_score
```

### 2.2 Factuality Checking

For factual claims, use NLI models or LLM-as-judge:

```python
from pydantic import BeforeValidator

# Using HuggingFace NLI model
def check_factual_consistency(claim: str, context: str) -> bool:
    """Check if claim is supported by context using NLI."""
    from transformers import pipeline

    nli = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    result = nli(claim, candidate_labels=["entailment", "contradiction", "neutral"])
    return result["labels"][0] == "entailment"

# Validator that requires context
class FactualClaim(BaseModel):
    claim: str
    supporting_evidence: str

    @model_validator(mode="after")
    def verify_factuality(self) -> "FactualClaim":
        if not check_factual_consistency(self.claim, self.supporting_evidence):
            raise ValueError("Claim not supported by evidence")
        return self
```

### 2.3 Consistency Validation

Detect self-contradictions within a single response:

```python
class ConsistentResponse(BaseModel):
    statements: list[str]

    @model_validator(mode="after")
    def check_internal_consistency(self) -> "ConsistentResponse":
        """Ensure statements don't contradict each other."""
        for i, s1 in enumerate(self.statements):
            for s2 in self.statements[i+1:]:
                if is_contradictory(s1, s2):
                    raise ValueError(f"Contradictory statements: '{s1}' vs '{s2}'")
        return self
```

**Using LLM Guard's Factual Consistency Scanner:**

```python
from llm_guard.output_scanners import FactualConsistency

scanner = FactualConsistency()
sanitized_output, is_valid, risk_score = scanner.scan(prompt, output)
```

### 2.4 Source Attribution Verification

Ensure LLM responses cite sources correctly:

```python
class CitedResponse(BaseModel):
    text: str
    citations: list[Citation]

    class Citation(BaseModel):
        source_id: str
        quote: str
        page: int | None = None

    @model_validator(mode="after")
    def verify_citations(self) -> "CitedResponse":
        # Check that all citation IDs reference real documents
        # Check that quotes appear in source documents
        # Validation requires access to source documents via context
        return self
```

**RAG Integration Pattern:**

```python
@spell(model="reasoning")
def answer_with_sources(question: str, documents: list[Document]) -> CitedResponse:
    """Answer the question using ONLY the provided documents.
    Every claim must include a citation to the source document.
    Use format: [source_id] for citations.
    """
    ...

# The documents are passed as context, citations verified post-generation
```

---

## 3. Safety and Content Filtering

> **Status**: The patterns in this section are NOT implemented in spellcrafting.
> They describe external library patterns (Detoxify, LLM Guard, Guardrails AI)
> that can be used alongside spellcrafting's guards. See `guards.md` for design rationale.

### 3.1 Toxicity Detection

**Layered Approach:**

```python
from pydantic import AfterValidator
from typing import Annotated

# Fast: regex-based blocklist
def check_blocklist(text: str) -> str:
    """Quick check against known bad patterns."""
    import re
    patterns = [r"\b(slur|profanity)\b"]  # Simplified
    for pattern in patterns:
        if re.search(pattern, text, re.I):
            raise ValueError("Content violates blocklist")
    return text

# Medium: Detoxify classifier
def check_toxicity_ml(text: str, threshold: float = 0.5) -> str:
    """ML-based toxicity detection."""
    from detoxify import Detoxify

    results = Detoxify("original").predict(text)
    if results["toxicity"] > threshold:
        raise ValueError(f"Content too toxic: {results['toxicity']:.2f}")
    return text

# Compose validators
SafeContent = Annotated[str,
    AfterValidator(check_blocklist),
    AfterValidator(check_toxicity_ml)
]

class SafeResponse(BaseModel):
    content: SafeContent
```

**Using Guardrails AI:**

```python
from guardrails import Guard
from guardrails.hub import ToxicLanguage

guard = Guard().use(ToxicLanguage(threshold=0.5, on_fail="exception"))

@spell(model="fast")
def generate_response(query: str) -> str:
    """Generate a helpful response."""
    ...

# Wrap with guardrail
def safe_generate(query: str) -> str:
    response = generate_response(query)
    return guard.validate(response)
```

### 3.2 Refusal Detection

Detect when the LLM refuses to help (sometimes incorrectly):

```python
from pydantic import model_validator
import re

REFUSAL_PATTERNS = [
    r"I cannot",
    r"I'm unable to",
    r"I won't",
    r"I'm not able to",
    r"I apologize, but",
    r"As an AI",
]

class ActionableResponse(BaseModel):
    content: str

    @model_validator(mode="after")
    def check_not_refusal(self) -> "ActionableResponse":
        """Ensure the model actually attempted the task."""
        for pattern in REFUSAL_PATTERNS:
            if re.search(pattern, self.content, re.I):
                raise ValueError(f"Model refused to complete task: {self.content[:100]}...")
        return self
```

**Using LLM Guard:**

```python
from llm_guard.output_scanners import NoRefusal

scanner = NoRefusal()
_, is_refusal, _ = scanner.scan(prompt, output)
if is_refusal:
    # Retry with different prompt or escalate
    pass
```

**Classifier-Based Detection:**

```python
from transformers import pipeline

# Fine-tuned refusal classifier
classifier = pipeline("text-classification",
    model="Human-CentricAI/LLM-Refusal-Classifier")

def is_refusal(text: str) -> bool:
    result = classifier(text)[0]
    return result["label"] == "refusal" and result["score"] > 0.8
```

### 3.3 Off-Topic Detection

Keep conversations within expected bounds:

```python
from pydantic import model_validator
from sentence_transformers import SentenceTransformer, util

class TopicBoundedResponse(BaseModel):
    content: str

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def check_on_topic(self) -> "TopicBoundedResponse":
        """Verify response relates to allowed topics."""
        allowed_topics = [
            "customer support",
            "product information",
            "order status",
        ]

        model = SentenceTransformer("all-MiniLM-L6-v2")
        response_embedding = model.encode(self.content)
        topic_embeddings = model.encode(allowed_topics)

        max_similarity = max(util.cos_sim(response_embedding, te).item()
                            for te in topic_embeddings)

        if max_similarity < 0.3:
            raise ValueError("Response is off-topic")
        return self
```

**NeMo Guardrails Topic Control:**

```python
# config.yml
rails:
  output:
    flows:
      - check topic

flows:
  check topic:
    - if: "off_topic($last_bot_message)"
      - stop
      - bot refuse
```

### 3.4 Competitor Mention Filtering

Enterprise use case: prevent mentioning competitors:

```python
from pydantic import AfterValidator
from typing import Annotated

COMPETITORS = {"acme", "globex", "initech", "umbrella corp"}

def filter_competitors(text: str) -> str:
    """Block any mention of competitor names."""
    words = text.lower().split()
    mentioned = COMPETITORS.intersection(words)
    if mentioned:
        raise ValueError(f"Response mentions competitors: {mentioned}")
    return text

# More sophisticated: use NER to catch variations
def filter_competitors_ner(text: str) -> str:
    """Use NER to catch competitor mentions including variations."""
    import spacy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)

    for ent in doc.ents:
        if ent.label_ == "ORG" and ent.text.lower() in COMPETITORS:
            raise ValueError(f"Response mentions competitor: {ent.text}")
    return text

CompetitorFreeText = Annotated[str, AfterValidator(filter_competitors)]
```

---

## 4. Quality Metrics

> **Status**: The patterns in this section are NOT implemented in spellcrafting.
> They describe external library patterns (embedding models, LLM-as-judge)
> that can be used alongside spellcrafting's guards. See `guards.md` for design rationale.

### 4.1 Relevance Scoring

Measure how well the output addresses the input:

```python
from pydantic import BaseModel, computed_field
from sentence_transformers import SentenceTransformer, util

class ScoredResponse(BaseModel):
    content: str
    _query: str = ""  # Set via context

    @computed_field
    @property
    def relevance_score(self) -> float:
        """Cosine similarity between query and response."""
        if not self._query:
            return 1.0
        model = SentenceTransformer("all-MiniLM-L6-v2")
        query_emb = model.encode(self._query)
        response_emb = model.encode(self.content)
        return float(util.cos_sim(query_emb, response_emb)[0][0])
```

**LLM-as-Judge for Relevance:**

```python
@spell(model="fast")
def score_relevance(query: str, response: str) -> RelevanceScore:
    """Score how well the response addresses the query.

    Consider:
    - Does it answer the question?
    - Is it complete?
    - Is it concise (no unnecessary info)?
    """
    ...

class RelevanceScore(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    reasoning: str
    missing_aspects: list[str] = []
```

### 4.2 Completeness Checking

Verify all required aspects are covered:

```python
class CompleteAnswer(BaseModel):
    """Ensures all aspects of a question are addressed."""
    main_answer: str
    aspects_covered: list[str]

    @model_validator(mode="after")
    def check_completeness(self) -> "CompleteAnswer":
        # Compare against expected aspects from context
        expected = self.model_config.get("expected_aspects", [])
        missing = set(expected) - set(self.aspects_covered)
        if missing:
            raise ValueError(f"Missing aspects: {missing}")
        return self
```

**Using LLM to Check Completeness:**

```python
@spell(model="fast")
def check_completeness(question: str, answer: str) -> CompletenessCheck:
    """Evaluate if the answer fully addresses all parts of the question.

    Identify:
    - Which parts of the question are addressed
    - Which parts are missing or incomplete
    - Overall completeness score
    """
    ...

class CompletenessCheck(BaseModel):
    score: float = Field(ge=0.0, le=1.0, description="0=incomplete, 1=fully complete")
    addressed_parts: list[str]
    missing_parts: list[str]
    suggestions: list[str] = []
```

### 4.3 Confidence Estimation

Estimate model's confidence in its output:

**Self-Reported Confidence:**

```python
class ConfidentResponse(BaseModel):
    answer: str
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Your confidence in this answer. 0=guessing, 1=certain"
    )
    confidence_reasoning: str
```

**Consistency-Based Confidence:**

```python
async def estimate_confidence(spell_fn, input_data, n_samples: int = 5) -> float:
    """Estimate confidence via self-consistency."""
    results = []
    for _ in range(n_samples):
        result = await spell_fn(input_data)
        results.append(result)

    # For structured outputs, compare field-by-field
    # Return ratio of agreeing samples
    return calculate_agreement_ratio(results)
```

**Entropy-Based Uncertainty (requires model access):**

```python
# For models that expose logprobs
def calculate_semantic_entropy(logprobs: list[float]) -> float:
    """Lower entropy = higher confidence."""
    import numpy as np
    probs = np.exp(logprobs)
    return -np.sum(probs * np.log(probs + 1e-10))
```

---

## 5. Validation Pipeline Architecture

> **Note:** This section describes **conceptual patterns** that can be implemented using
> spellcrafting's `@guard.output()` decorator or external validation libraries. The `ValidationConfig`
> shown below is a hypothetical API, not currently implemented in spellcrafting.

### 5.1 Composable Validation Layers (Pattern)

This pattern shows how to build a validation pipeline. In spellcrafting, you achieve this
with stacked `@guard.output()` decorators:

```python
# spellcrafting approach: stacked guards
@spell
@guard.output(check_toxicity)
@guard.output(check_competitors)
@guard.output(check_relevance)
def generate_response(query: str) -> Response:
    """Generate a helpful response."""
    ...
```

For more complex pipelines, here's a generic pattern you could implement:

```python
from typing import Protocol, TypeVar
from dataclasses import dataclass

T = TypeVar("T")

class OutputValidator(Protocol[T]):
    """Protocol for output validators."""

    def validate(self, output: T, context: dict) -> T:
        """Validate and potentially transform output. Raise on failure."""
        ...

    @property
    def name(self) -> str: ...


@dataclass
class ValidationResult(Generic[T]):
    output: T
    passed: bool
    scores: dict[str, float]
    errors: list[str]


class ValidationPipeline(Generic[T]):
    """Chain multiple validators."""

    def __init__(self, validators: list[OutputValidator[T]]):
        self.validators = validators

    def run(self, output: T, context: dict) -> ValidationResult[T]:
        errors = []
        scores = {}
        current = output

        for validator in self.validators:
            try:
                current = validator.validate(current, context)
                scores[validator.name] = 1.0
            except ValueError as e:
                errors.append(f"{validator.name}: {e}")
                scores[validator.name] = 0.0

        return ValidationResult(
            output=current,
            passed=len(errors) == 0,
            scores=scores,
            errors=errors,
        )
```

### 5.2 Integration with Spells

**Current spellcrafting approach** - use `@guard.output()` decorators:

```python
from spellcrafting import spell, guard

def no_competitors(output: Response, ctx: dict) -> Response:
    """Block competitor mentions."""
    competitors = {"acme", "globex"}
    if any(c in output.content.lower() for c in competitors):
        raise ValueError("Response mentions competitor")
    return output

@spell(model="fast")
@guard.output(no_competitors)
def generate_response(query: str, context: str) -> Response:
    """Generate a helpful response based on the provided context."""
    ...
```

**Hypothetical declarative API** (not implemented - shown for reference):

```python
# This is a HYPOTHETICAL API, not currently available in spellcrafting
@spell(
    model="fast",
    validation=ValidationConfig(
        toxicity_threshold=0.5,
        require_grounding=True,
        block_competitors=["acme", "globex"],
        min_relevance=0.7,
    )
)
def generate_response(query: str, context: str) -> Response:
    """Generate a helpful response based on the provided context."""
    ...
```

### 5.3 Validation Metrics in Logging

spellcrafting's `SpellExecutionLog` includes `ValidationMetrics` for tracking guard results
and retry information. The **actual implementation** in `logging.py`:

```python
@dataclass
class ValidationMetrics:
    """Tracks validation during spell execution."""

    # Retry tracking
    attempt_count: int = 1
    retry_reasons: list[str] = field(default_factory=list)

    # Guard results
    input_guards_passed: list[str] = field(default_factory=list)
    input_guards_failed: list[str] = field(default_factory=list)
    output_guards_passed: list[str] = field(default_factory=list)
    output_guards_failed: list[str] = field(default_factory=list)

    # Pydantic validation errors
    pydantic_errors: list[str] = field(default_factory=list)

    # On-fail strategy tracking
    on_fail_triggered: str | None = None  # "escalate", "fallback", "custom"
    escalated_to_model: str | None = None
```

**Extended schema** (not implemented - conceptual for semantic validation):

```python
# Hypothetical extension for semantic/safety metrics
@dataclass
class ExtendedValidationMetrics(ValidationMetrics):
    # Semantic (would require external libraries)
    hallucination_score: float | None = None
    groundedness_score: float | None = None

    # Safety (would require Detoxify, LLM Guard, etc.)
    toxicity_score: float | None = None
    is_refusal: bool = False

    # Quality (would require embedding models)
    relevance_score: float | None = None
    confidence_score: float | None = None
```

See `guards.md` for why spellcrafting defers semantic/safety scoring to external libraries.

---

## 6. Actual spellcrafting Implementation

For reference, here's what spellcrafting actually provides:

### 6.1 Guard Decorators (`guard.py`)

```python
from spellcrafting import spell, guard

def validate_input(input_args: dict, ctx: dict) -> dict:
    """Validate/transform inputs before LLM call."""
    if not input_args.get("text", "").strip():
        raise ValueError("Text cannot be empty")
    return input_args

def validate_output(output: str, ctx: dict) -> str:
    """Validate/transform output after LLM call."""
    if len(output) < 10:
        raise ValueError("Output too short")
    return output

@spell(model="fast")
@guard.input(validate_input)
@guard.output(validate_output)
def summarize(text: str) -> str:
    """Summarize the text."""
    ...
```

### 6.2 OnFail Strategies (`on_fail.py`)

```python
from spellcrafting import spell, OnFail

# Escalate to better model on validation failure
@spell(model="fast", on_fail=OnFail.escalate("reasoning"))
def complex_task(query: str) -> Analysis:
    """Complex analysis that may need a better model."""
    ...

# Return default on failure
@spell(on_fail=OnFail.fallback(default=EmptyResponse()))
def optional_enrichment(data: str) -> Enriched:
    """Optional enrichment - return empty if LLM fails."""
    ...

# Custom error handling
@spell(on_fail=OnFail.custom(my_error_handler))
def with_custom_handling(text: str) -> Result:
    """Custom error handling logic."""
    ...
```

---

## 7. Implementation Roadmap

> **Status:** This roadmap reflects the original design vision. See `guards.md`
> for what was actually built vs. deferred.

### Implemented

1. Pydantic validation via return types
2. Retry with feedback via `retries` parameter
3. `@guard.input()` and `@guard.output()` decorators
4. `OnFail` strategies (escalate, fallback, custom)
5. `ValidationMetrics` in logging

### Deferred to External Libraries

1. Toxicity detection (use Detoxify, LLM Guard)
2. Hallucination/grounding checks (use NLI models)
3. Relevance scoring (use embedding models)
4. Declarative `ValidationConfig` API

---

## 8. Tools and Libraries (External)

### 8.1 Recommended Stack

| Category | Tool | Use Case |
|----------|------|----------|
| Structured outputs | [Instructor](https://python.useinstructor.com/) | Pydantic + retry + validation |
| General guardrails | [Guardrails AI](https://github.com/guardrails-ai/guardrails) | Composable validators |
| Conversational safety | [NeMo Guardrails](https://github.com/NVIDIA-NeMo/Guardrails) | Dialog flow control |
| Toxicity | [Detoxify](https://github.com/unitaryai/detoxify) | Fast toxicity classification |
| Factuality | [LLM Guard](https://llm-guard.com/) | NLI-based consistency |
| Embeddings | [Sentence Transformers](https://www.sbert.net/) | Relevance/similarity scoring |

### 8.2 Minimal Dependencies Approach

For minimal footprint, implement core validation without heavy dependencies:

```python
# Core validation without ML dependencies
class MinimalValidator:
    """Regex and rule-based validation only."""

    @staticmethod
    def check_json_structure(text: str, schema: type) -> bool:
        """Validate JSON structure matches Pydantic schema."""
        try:
            schema.model_validate_json(text)
            return True
        except Exception:
            return False

    @staticmethod
    def check_blocklist(text: str, blocklist: set[str]) -> list[str]:
        """Find blocklisted terms in text."""
        words = set(text.lower().split())
        return list(words & blocklist)

    @staticmethod
    def check_refusal_patterns(text: str) -> bool:
        """Check for common refusal phrases."""
        patterns = ["i cannot", "i'm unable", "i won't"]
        return any(p in text.lower() for p in patterns)
```

---

## 9. Original Research Roadmap (Historical)

> **Note:** This was the original research roadmap. See section 7 for what was actually
> implemented. Most items here were intentionally deferred - see `guards.md` for rationale.

### Phase 1: Structural Foundation - COMPLETED
1. Document Pydantic validation patterns for spells
2. Add validation metrics to `SpellExecutionLog`
3. ~~Implement `ValidationResult` return type option~~ -> Implemented as `SpellResult` via `.with_metadata()`

### Phase 2: Safety Layer - DEFERRED
> Deferred to external libraries (Detoxify, LLM Guard, OpenAI Moderation)

1. ~~Integrate Detoxify for toxicity detection~~ -> Use external library
2. ~~Add refusal detection~~ -> Use external library
3. ~~Implement competitor filtering~~ -> Achievable via `@guard.output()`

### Phase 3: Semantic Validation - DEFERRED
> Deferred to external libraries (NLI models, embedding models)

1. ~~Add groundedness checking (NLI-based)~~ -> Use external library
2. ~~Implement self-consistency helper~~ -> Pattern documented in this file
3. ~~Source attribution verification~~ -> Use Pydantic validators

### Phase 4: Quality Metrics - DEFERRED
> Deferred to external libraries (embedding models, LLM-as-judge patterns)

1. ~~Relevance scoring (embedding-based)~~ -> Use Sentence Transformers
2. ~~Completeness checking (LLM-as-judge)~~ -> Pattern documented (use validator spells)
3. ~~Confidence estimation~~ -> Implemented in `SpellResult`

### Phase 5: Pipeline Integration - PARTIALLY IMPLEMENTED
1. ~~Declarative `ValidationConfig` for `@spell`~~ -> Use `@guard` decorators instead
2. ~~Composable `ValidationPipeline`~~ -> Use stacked `@guard` decorators
3. ~~Validation dashboard/reporting~~ -> Use `SpellExecutionLog` + external tools

---

## 10. Design Decisions

1. **Pydantic-first**: All validation leverages Pydantic validators where possible. This keeps validation declarative and co-located with types.

2. **Layered validation**: Cheap checks (regex, blocklist) run before expensive checks (ML models, LLM-as-judge).

3. **Fail-fast vs. collect-all**: Default to fail-fast for production. Collect-all mode for debugging/evaluation.

4. **Validation is opt-in**: Core spell execution has zero validation overhead. Validation layers are explicitly configured.

5. **Metrics over blocking**: For quality metrics, log scores rather than hard-failing. Let applications decide thresholds.

6. **Context propagation**: Validation often needs access to input context (for grounding checks). Use Pydantic's `ValidationInfo.context`.

---

## 11. References

### Structural Validation
- [Instructor Library](https://python.useinstructor.com/)
- [Pydantic LLM Validation](https://pydantic.dev/articles/llm-validation)
- [Pydantic v2.10 Partial Validation](https://jakubkrajewski.substack.com/p/pydantic-v210-processing-the-output)

### Semantic Validation
- [Detecting Hallucinations Using Semantic Entropy](https://www.nature.com/articles/s41586-024-07421-0)
- [Self-Contradictory Hallucinations](https://arxiv.org/pdf/2305.15852)
- [LLM Grounding](https://www.iguazio.com/glossary/llm-grounding/)
- [RAG Evaluation - Groundedness](https://www.deepset.ai/blog/rag-llm-evaluation-groundedness)

### Safety and Content Filtering
- [Guardrails AI](https://github.com/guardrails-ai/guardrails)
- [NVIDIA NeMo Guardrails](https://github.com/NVIDIA-NeMo/Guardrails)
- [LLM Guard](https://llm-guard.com/)
- [Detoxify](https://github.com/unitaryai/detoxify)
- [LLM Refusal Classifier](https://huggingface.co/Human-CentricAI/LLM-Refusal-Classifier)

### Quality Metrics
- [LLM Evaluation Metrics Guide](https://www.confident-ai.com/blog/llm-evaluation-metrics-everything-you-need-for-llm-evaluation)
- [LLM-as-a-Judge Guide](https://www.evidentlyai.com/llm-guide/llm-as-a-judge)
- [Uncertainty Quantification Survey](https://arxiv.org/html/2503.15850v1)
- [Confidence Estimation in LLMs](https://medium.com/@georgekar91/measuring-confidence-in-llm-responses-e7df525c283f)

### Enterprise Guardrails
- [Aporia Off-Topic Detection](https://www.aporia.com/ai-guardrails/off-topic-detection/)
- [Datadog LLM Guardrails Best Practices](https://www.datadoghq.com/blog/llm-guardrails-best-practices/)
- [Palo Alto Networks LLM Content Filtering Study](https://unit42.paloaltonetworks.com/comparing-llm-guardrails-across-genai-platforms/)
