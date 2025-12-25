# Guardrails Design for magically

> Synthesis of research into an opinionated design for the magically library.
> This document recommends what to build, what NOT to build, and why.

## Executive Summary

After analyzing four research documents covering framework comparisons, input validation, output validation, and enterprise patterns, this design takes an opinionated stance:

**Build:**
- Output validation via Pydantic (already have it - document patterns)
- Retry with validation feedback (already have it via `retries` parameter)
- A simple `@validate` decorator for composable output checks
- Validator spells - using `@spell` to validate other spell outputs

**Don't Build:**
- Input validation framework (use specialized libraries)
- PII detection/redaction (use Presidio, AWS Comprehend)
- Content moderation (use OpenAI Moderation API)
- Prompt injection detection (use Lakera, LLM Guard)
- Complex DSLs or configuration systems (NeMo-style)
- Rate limiting, budgeting, circuit breakers (use LiteLLM Proxy, Portkey)

**Philosophy:** magically is about making LLM calls feel like native Python. Guardrails should feel the same way - Pydantic validators, not YAML configs.

---

## 1. What We Already Have (and It's Enough)

### 1.1 Structural Validation via Pydantic

The `@spell` decorator already uses the return type as the output schema. Pydantic handles structural validation automatically:

```python
from pydantic import BaseModel, Field, field_validator

class Analysis(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float = Field(ge=0.0, le=1.0)
    key_points: list[str] = Field(min_length=1, max_length=10)

    @field_validator("key_points")
    @classmethod
    def no_empty_points(cls, v: list[str]) -> list[str]:
        return [p.strip() for p in v if p.strip()]

@spell(retries=2)
def analyze(text: str) -> Analysis:
    """Analyze the sentiment and key points of the given text."""
    ...
```

**This is the right approach.** Pydantic validators are:
- Familiar to Python developers
- Co-located with types
- Composable via `Annotated`
- Automatically retried by PydanticAI

### 1.2 Retry with Feedback

The `retries` parameter already provides automatic retry with validation error feedback:

```python
@spell(retries=3)  # Up to 3 attempts on validation failure
def extract_entities(text: str) -> Entities:
    """Extract named entities from text."""
    ...
```

PydanticAI sends validation errors back to the LLM, which corrects its output. This is exactly what Instructor does, and we get it for free.

**Recommendation:** Document these patterns well. No new code needed.

---

## 2. What to Add: Semantic Validation

Pydantic handles *structural* validation (types, ranges, formats). But some validation is *semantic* - it requires understanding content, not just structure.

### 2.1 The `@validate` Decorator

A lightweight decorator for composable output validation:

```python
from magically import spell, validate

@spell
@validate(lambda r: len(r.key_points) >= 3, "Must have at least 3 key points")
@validate(lambda r: r.confidence > 0.5, "Confidence too low")
def analyze(text: str) -> Analysis:
    """Analyze the sentiment and key points of the given text."""
    ...
```

**Design:**

```python
from typing import Callable, TypeVar
from functools import wraps

T = TypeVar("T")

class ValidationError(Exception):
    """Raised when output validation fails."""
    pass

def validate(
    check: Callable[[T], bool],
    message: str = "Validation failed",
    *,
    on_fail: str = "raise",  # "raise" | "warn" | "log"
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for semantic output validation.

    Args:
        check: Predicate function that returns True if valid
        message: Error message on failure
        on_fail: Action on failure - raise exception, warn, or just log
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            result = func(*args, **kwargs)
            if not check(result):
                if on_fail == "raise":
                    raise ValidationError(message)
                elif on_fail == "warn":
                    import warnings
                    warnings.warn(message, stacklevel=2)
                # "log" handled by logging system
            return result
        return wrapper
    return decorator
```

**Why this is enough:**
- Simple, Pythonic API
- Composable (stack decorators)
- Works with any callable, not just spells
- No configuration DSL

### 2.2 Validator Spells

The most powerful insight from the research: some validation is best done by another LLM. Rather than building a complex framework, we can use spells to validate spells:

```python
from pydantic import BaseModel
from magically import spell

class ValidationResult(BaseModel):
    is_valid: bool
    issues: list[str]
    suggestions: list[str]

@spell(model="fast")  # Use a fast, cheap model for validation
def check_professional_tone(content: str) -> ValidationResult:
    """Check if the content maintains a professional tone.

    Flag issues like:
    - Informal language
    - Inappropriate humor
    - Unprofessional phrasing
    """
    ...

@spell(model="fast")
def check_factual_grounding(claim: str, sources: list[str]) -> ValidationResult:
    """Verify the claim is supported by the provided sources.

    Check that:
    - Each assertion has a supporting source
    - No claims are made beyond what sources state
    - Quotes are accurate
    """
    ...

# Usage: validate one spell's output with another
@spell
def generate_report(data: str) -> Report:
    """Generate a professional report from the data."""
    ...

async def validated_report(data: str) -> Report:
    report = await generate_report(data)

    # Validate with another spell
    tone_check = await check_professional_tone(report.content)
    if not tone_check.is_valid:
        # Could retry, modify, or raise
        raise ValidationError(f"Tone issues: {tone_check.issues}")

    return report
```

**Why this is the right pattern:**
- Uses the library's core abstraction (`@spell`)
- Flexible - any validation logic expressible in natural language
- Composable - chain multiple validators
- No new concepts to learn

### 2.3 Built-in Validator Spells (Optional)

We *could* ship a few common validator spells:

```python
# magically.validators (optional module)
from magically import spell
from pydantic import BaseModel

class ToneCheck(BaseModel):
    is_appropriate: bool
    issues: list[str]

@spell(model="fast")
def check_tone(
    content: str,
    expected_tone: str = "professional",
) -> ToneCheck:
    """Check if content matches the expected tone.

    Tone should be: {expected_tone}

    Flag any content that doesn't match this tone.
    """
    ...
```

**Recommendation:** Ship 0-3 common validators initially. Let the community build more. Don't try to cover every use case.

---

## 3. What NOT to Build

### 3.1 Input Validation (PII, Injection, Moderation)

**Don't build this.** The research is clear:

1. **PII Detection** requires NER models, entity-specific logic, and compliance expertise. Presidio and AWS Comprehend exist for this.

2. **Prompt Injection** is the #1 AI security risk (OWASP). Defense-in-depth requires multiple layers. Lakera and LLM Guard are purpose-built.

3. **Content Moderation** requires trained classifiers. OpenAI's Moderation API is free and covers major categories.

**The pattern for users:**

```python
from presidio_analyzer import AnalyzerEngine
from llm_guard.input_scanners import PromptInjection
import openai

# Users compose their own input pipeline
def validate_input(text: str) -> str:
    # 1. Check for PII
    analyzer = AnalyzerEngine()
    if analyzer.analyze(text, language="en"):
        raise ValueError("PII detected")

    # 2. Check for injection
    scanner = PromptInjection()
    _, is_valid, _ = scanner.scan(text)
    if not is_valid:
        raise ValueError("Potential injection")

    # 3. Check moderation
    result = openai.moderations.create(input=text)
    if result.results[0].flagged:
        raise ValueError("Content flagged")

    return text

# Then use with spell
@spell
def process(text: str) -> Output:
    """Process the validated text."""
    ...

# Usage
safe_text = validate_input(user_input)
result = process(safe_text)
```

**Why not integrate these?**
- Each library has its own dependencies (spaCy, torch, etc.)
- Configuration is domain-specific (which PII entities? what threshold?)
- We'd be wrapping wrappers with no added value
- Users can compose exactly what they need

### 3.2 Rate Limiting, Budgeting, Circuit Breakers

**Don't build this.** These are infrastructure concerns, not library concerns.

The research documents LiteLLM Proxy, Portkey, and other gateways that handle:
- Token budgets per user/team
- Rate limiting (RPM, TPM)
- Circuit breakers and fallbacks
- Provider routing

**Why not in magically?**
- Requires persistence (Redis, database) for stateful tracking
- Multi-process/distributed coordination is complex
- Existing solutions are battle-tested
- We'd be building a gateway inside a library

**Recommendation:** Document how to use magically with LiteLLM Proxy or similar.

### 3.3 Configuration DSLs

NeMo Guardrails uses Colang, a custom DSL for dialog flows. The research shows:
- Significant learning curve
- 3-4x latency overhead
- Multiple days for integration

**Don't do this.** Python *is* our DSL:

```python
# Bad: Custom DSL
"""
define flow
    user express insult
    bot refuse to respond
"""

# Good: Python
@spell
def respond(message: str) -> Response:
    """Respond to the user message politely.
    If the message is insulting, politely decline to engage.
    """
    ...
```

### 3.4 Hub/Registry Systems

Guardrails AI has a validator hub requiring separate installation (`guardrails hub install hub://...`).

**Don't do this.** PyPI is our hub. If we ship validators, they're just Python functions:

```python
# Users install once
pip install magically

# Validators are just functions
from magically.validators import check_tone, check_grounding
```

---

## 4. Integration Patterns

### 4.1 With PII Libraries

```python
from functools import wraps
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

def anonymize_pii(func):
    """Decorator to anonymize PII in first string argument."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if args and isinstance(args[0], str):
            results = analyzer.analyze(text=args[0], language="en")
            if results:
                anon = anonymizer.anonymize(text=args[0], analyzer_results=results)
                args = (anon.text,) + args[1:]
        return func(*args, **kwargs)
    return wrapper

# Usage
@anonymize_pii
@spell
def summarize(text: str) -> Summary:
    """Summarize the text."""
    ...
```

### 4.2 With Content Moderation

```python
from openai import OpenAI

def moderate_input(func):
    """Decorator to moderate input content."""
    client = OpenAI()

    @wraps(func)
    def wrapper(*args, **kwargs):
        if args and isinstance(args[0], str):
            result = client.moderations.create(input=args[0])
            if result.results[0].flagged:
                raise ValueError("Content flagged by moderation")
        return func(*args, **kwargs)
    return wrapper
```

### 4.3 With LiteLLM Proxy

```python
# pyproject.toml
[tool.magically.models]
fast = { model = "openai:gpt-4o-mini" }  # Actually routes through proxy

# Environment
OPENAI_API_BASE=http://localhost:4000  # LiteLLM Proxy

# The proxy handles rate limiting, budgets, etc.
# magically just makes the call
@spell(model="fast")
def process(text: str) -> Output:
    ...
```

---

## 5. Implementation Plan

### Phase 1: Documentation (Week 1)

No code changes. Document existing patterns:
1. Pydantic validation patterns for spells
2. `retries` parameter usage
3. Custom validators via `Annotated` and `field_validator`
4. Integration patterns with external libraries

### Phase 2: `@validate` Decorator (Week 2)

Add the lightweight validation decorator:
1. `validate(predicate, message, on_fail)` decorator
2. `ValidationError` exception
3. Async support
4. Integration with logging system

Estimated: ~100 lines of code.

### Phase 3: Validator Spells (Week 3, Optional)

Ship 1-3 common validator spells in `magically.validators`:
1. `check_tone(content, expected_tone) -> ToneCheck`
2. `check_grounding(claim, sources) -> GroundingCheck`
3. `check_completeness(question, answer) -> CompletenessCheck`

These are just examples showing the pattern, not a comprehensive library.

---

## 6. What Success Looks Like

### For Users

```python
from pydantic import BaseModel, Field, field_validator
from magically import spell, validate

class Report(BaseModel):
    title: str = Field(min_length=5)
    sections: list[str] = Field(min_length=1)

    @field_validator("sections")
    @classmethod
    def no_empty_sections(cls, v):
        return [s for s in v if s.strip()]

@spell(retries=2)
@validate(lambda r: len(r.sections) >= 3, "Need at least 3 sections")
def generate_report(topic: str) -> Report:
    """Generate a comprehensive report on the topic."""
    ...
```

- Structural validation via Pydantic types
- Semantic validation via `@validate`
- Automatic retry on failure
- Zero new concepts to learn

### For the Library

- Small surface area (one decorator, one exception)
- No new dependencies
- Composable with existing Python patterns
- Follows the "spells are just functions" philosophy

---

## 7. Appendix: Research Summary

### Key Insights from Framework Comparison

| Framework | Lesson for magically |
|-----------|---------------------|
| Guardrails AI | Composable validators are good, but hub dependencies are bad |
| NeMo Guardrails | DSLs add complexity and latency - avoid |
| LangChain | `with_structured_output()` is the right pattern - we have it |
| Instructor | Pure Pydantic + retry is sufficient - we have it |

### What Each Framework Does Well

1. **Guardrails AI:** Composable `.use()` pattern translates to stacked decorators
2. **Instructor:** Proves that Pydantic + retry is enough for 90% of cases
3. **NeMo:** Shows what NOT to do (complex config, DSLs, high latency)
4. **LangChain:** Confirms native structured output > parsing

### Enterprise Patterns We Explicitly Defer

| Pattern | Recommendation |
|---------|---------------|
| Audit logging | Already in logging.py |
| Cost tracking | Already in logging.py |
| Rate limiting | Use LiteLLM Proxy |
| Budget enforcement | Use LiteLLM Proxy |
| Circuit breakers | Use infrastructure (Istio, Envoy) |
| Data residency | Provider-level concern |
| HIPAA/GDPR compliance | Use specialized services |

---

## 8. Decision Log

| Decision | Rationale |
|----------|-----------|
| No input validation framework | Specialized libraries (Presidio, LLM Guard, OpenAI Moderation) are better |
| No rate limiting | Infrastructure concern, not library concern |
| No configuration DSL | Python is our DSL |
| No validator hub | PyPI is our hub |
| Simple `@validate` decorator | Matches library's "Python-native" philosophy |
| Validator spells pattern | Uses core abstraction, infinitely flexible |
| Pydantic-first validation | Already have it, it's the right approach |

---

## References

- Research: `/docs/research/guardrails-frameworks.md`
- Research: `/docs/research/input_validation_guardrails.md`
- Research: `/docs/design/output_validation.md`
- Research: `/docs/research/enterprise-guardrails-patterns.md`
