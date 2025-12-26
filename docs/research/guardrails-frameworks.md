# LLM Guardrails Frameworks Research

This document analyzes four major guardrails/validation libraries for LLMs, focusing on patterns that could inform a decorator-based spell system with simple, composable guardrails.

## Implementation Status

This research document informed the design of spellcrafting's guard and validation systems. Key decisions:

**Implemented:**
- `@guard.input()` and `@guard.output()` decorators (Guardrails AI-inspired composable pattern)
- `OnFail` strategies: `RAISE`, `escalate()`, `fallback()`, `custom()` (Guardrails AI on_fail pattern)
- Pydantic-first validation via return types (Instructor-inspired)
- Built-in retry with error context via PydanticAI
- Provider-agnostic design across multiple LLM providers

**Explicitly NOT implemented (as recommended):**
- No DSL or config files (learned from NeMo Guardrails complexity)
- No hub dependencies (learned from Guardrails AI installation friction)
- No input validation building (beyond structural Pydantic validation)

**Deferred to external libraries:**
- Toxicity detection (use Detoxify, LLM Guard)
- PII detection (use Presidio, AWS Comprehend)
- Hallucination detection (use NLI models)

See `guard.py` and `on_fail.py` for the actual implementation.

## Table of Contents
1. [Guardrails AI](#guardrails-ai)
2. [NeMo Guardrails (NVIDIA)](#nemo-guardrails-nvidia)
3. [LangChain Output Parsers](#langchain-output-parsers)
4. [Instructor](#instructor)
5. [Comparative Analysis](#comparative-analysis)
6. [Patterns for Spell System](#patterns-for-spell-system)

---

## Guardrails AI

**Repository:** [guardrails-ai/guardrails](https://github.com/guardrails-ai/guardrails)
**Documentation:** [guardrailsai.com/docs](https://guardrailsai.com/docs)
**Current Version:** 0.7.2 (PyPI)

### Core Concepts

Guardrails AI provides a **Guard** abstraction that wraps LLM calls with composable **validators**. The framework operates on both input and output, with configurable failure handling policies.

**Key Components:**
- **Guard**: The main wrapper that orchestrates validation
- **Validators**: Reusable validation rules from the Guardrails Hub
- **OnFailAction**: Configurable behavior when validation fails

### API Design

```python
from guardrails import Guard
from guardrails.hub import ToxicLanguage, DetectPII, RegexMatch

# Single validator with on_fail behavior
guard = Guard().use(
    ToxicLanguage,
    threshold=0.5,
    validation_method="sentence",
    on_fail="exception"
)

# Multiple validators composed together
guard = Guard().use_many(
    ToxicLanguage(threshold=0.5),
    DetectPII(pii_entities=["EMAIL_ADDRESS", "PHONE_NUMBER"]),
    RegexMatch(regex="^[A-Z]")
)

# Validation with metadata for context
result = guard.validate(
    "some text",
    metadata={
        'pii_entities': ["EMAIL_ADDRESS"],
        'sources': ["The sun is a star."]
    }
)

# LLM integration with structured output
from pydantic import BaseModel, Field

class Pet(BaseModel):
    pet_type: str = Field(description="Species of pet")
    name: str = Field(description="A unique pet name")

guard = Guard.for_pydantic(output_class=Pet)
result = guard(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Generate a pet"}]
)
```

### OnFail Actions

The `on_fail` parameter determines behavior when validation fails:

| Action | Behavior |
|--------|----------|
| `reask` | Re-prompt the LLM with failure context |
| `fix` | Programmatically fix the output (e.g., remove hallucinated sentences) |
| `filter` | Remove the invalid field, return rest of output |
| `refrain` | Return empty/fallback response |
| `noop` | Log failure but take no action |
| `exception` | Raise `ValidationError` |
| `fix_reask` | Try fix first, then reask if still invalid |
| `custom` | Call a custom function with value and `FailResult` |

```python
# Custom on_fail handler
def custom_on_fail(value, fail_result):
    return f"Sanitized: {value[:50]}..."

guard = Guard().use(DetectPII(on_fail=custom_on_fail))
```

### What They Do Well

1. **Composable Validators**: The `.use()` and `.use_many()` pattern allows clean chaining
2. **Hub Ecosystem**: Pre-built validators for common use cases (PII, toxicity, regex, SQL validation)
3. **Flexible Failure Handling**: Multiple on_fail strategies including custom handlers
4. **Pydantic Integration**: Native support for structured output validation
5. **Metadata Context**: Validators can receive runtime context via metadata dict

### Limitations and Pain Points

1. **Installation Complexity**: Each validator requires separate hub installation (`guardrails hub install hub://...`)
2. **Latency Overhead**: Validation adds processing time, especially with multiple validators
3. **Learning Curve**: Understanding when to use which on_fail action
4. **Hub Dependency**: Many useful validators require hub access

---

## NeMo Guardrails (NVIDIA)

**Repository:** [NVIDIA-NeMo/Guardrails](https://github.com/NVIDIA-NeMo/Guardrails)
**Documentation:** [docs.nvidia.com/nemo/guardrails](https://docs.nvidia.com/nemo/guardrails/latest/index.html)
**Current Version:** Requires Python 3.10+

### Core Concepts

NeMo Guardrails uses **Colang**, a domain-specific language for defining conversational flows and rails. It provides the most comprehensive dialog management among guardrails frameworks.

**Rail Types:**
- **Input Rails**: Filter/transform user input before processing
- **Dialog Rails**: Control conversation flow and LLM prompting
- **Output Rails**: Validate/filter generated responses
- **Retrieval Rails**: Filter RAG chunks before they reach the LLM

### API Design

**config.yml Structure:**
```yaml
models:
  - type: main
    engine: openai
    model: gpt-3.5-turbo-instruct

rails:
  input:
    flows:
      - check jailbreak
      - mask sensitive data on input
  output:
    flows:
      - self check facts
      - self check hallucination

config:
  sensitive_data_detection:
    input:
      entities:
        - PERSON
        - EMAIL_ADDRESS
```

**Colang Flow Definitions:**
```colang
# Define user intent patterns
define user express insult
    "You are stupid"
    "This is garbage"

# Define response flow
define flow
    user express insult
    bot express calmly willingness to help

# Input rail with action
define flow self check input
    $allowed = execute self_check_input

    if not $allowed
        bot refuse to respond
        stop
```

**Python Usage:**
```python
from nemoguardrails import RailsConfig, LLMRails

# Load from config directory
config = RailsConfig.from_path("./config")
rails = LLMRails(config)

response = rails.generate(messages=[{
    "role": "user",
    "content": "Hello! What can you do for me?"
}])

# Or initialize from content directly
config = RailsConfig.from_content(
    colang_content=colang_content,
    yaml_content=yaml_content
)
```

### What They Do Well

1. **Dialog Management**: Unique capability to define conversation flows declaratively
2. **Comprehensive Rail Types**: Input, output, dialog, and retrieval rails
3. **Colang DSL**: Python-like syntax that's readable and maintainable
4. **RAG Integration**: Built-in support for filtering retrieval chunks
5. **Enterprise Features**: Multi-agent support, GPU acceleration, NIM integrations

### Limitations and Pain Points

1. **Significant Latency**:
   - Base queries can go from 40s to 1m45s with guardrails
   - 3-4x slower than direct LLM calls reported
   - 3.5s overhead with bare bones configuration
   - Adding vector DB increases to 10-11s

2. **Complexity**:
   - Requires learning Colang DSL
   - Config file-based setup is verbose
   - Multiple full working days reported for LangChain integration

3. **Streaming Challenges**: Output rails process synchronously by default, breaking streaming

4. **Effectiveness Issues**:
   - Simple prompts can bypass rails without tuning
   - Fact-checker rails may miss fabricated content
   - Hallucinations not always caught

5. **C++ Dependency**: Uses `annoy` library requiring C++ compiler for installation

---

## LangChain Output Parsers

**Documentation:** [python.langchain.com](https://python.langchain.com/docs/concepts/structured_outputs/)
**Package:** `langchain-core`

### Core Concepts

LangChain provides multiple approaches to structured output:
1. **`with_structured_output()`**: Native model support for structured responses
2. **PydanticOutputParser**: Schema-based parsing with validation
3. **StructuredOutputParser**: Custom schema definitions
4. **OutputFixingParser**: Error recovery wrapper

### API Design

**Using `with_structured_output()` (Preferred):**
```python
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

class Joke(BaseModel):
    setup: str
    punchline: str

model = ChatOpenAI(model="gpt-4")
structured_model = model.with_structured_output(Joke)
result = structured_model.invoke("Tell me a joke")
# result is a validated Joke instance
```

**Using PydanticOutputParser:**
```python
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field, model_validator

class Joke(BaseModel):
    setup: str = Field(description="The question in the joke")
    punchline: str = Field(description="The answer")

    @model_validator(mode="before")
    @classmethod
    def validate_setup(cls, values):
        if values.get("setup") and not values["setup"].endswith("?"):
            raise ValueError("Setup must end with a question mark")
        return values

parser = PydanticOutputParser(pydantic_object=Joke)
format_instructions = parser.get_format_instructions()

prompt = PromptTemplate(
    template="Tell me a joke.\n{format_instructions}",
    partial_variables={"format_instructions": format_instructions}
)

chain = prompt | model | parser
result = chain.invoke({})
```

**Using OutputFixingParser for Error Recovery:**
```python
from langchain.output_parsers import OutputFixingParser

# Wraps another parser with LLM-based error correction
fixing_parser = OutputFixingParser.from_llm(
    parser=parser,
    llm=model
)
```

### What They Do Well

1. **Native Model Integration**: `with_structured_output()` leverages provider APIs directly
2. **Composable with LCEL**: Parsers work in LangChain Expression Language chains
3. **Pydantic V2 Support**: Modern validation with async and better performance
4. **Multiple Approaches**: Flexibility to choose based on model capabilities
5. **Error Recovery**: OutputFixingParser provides automatic retry with LLM

### Limitations and Pain Points

1. **JSON Parsing Fragility**:
   - Fails on newline characters (`\n` not escaped as `\\n`)
   - Single quotes instead of double quotes cause failures
   - Plain `json.loads` often works when parser fails

2. **Model-Specific Issues**:
   - Format instructions may not match model behavior (gpt-4o issues reported)
   - Different models interpret instructions differently

3. **Type Handling Bugs**:
   - Date/Union type parsing problems (`date | str` returns string)
   - Pydantic v2 compatibility breaking changes between versions

4. **Retry Limitations**:
   - `RetryWithErrorOutputParser` often fails to correct format issues
   - No built-in exponential backoff or max retries

5. **Recommendation**: Documentation now suggests using native structured output when available, implying parsers are a fallback

---

## Instructor

**Repository:** [567-labs/instructor](https://github.com/567-labs/instructor)
**Documentation:** [python.useinstructor.com](https://python.useinstructor.com/)
**Author:** Jason Liu
**Current Version:** 1.7.5 (March 2025)

### Core Concepts

Instructor is the most focused library: it patches LLM client libraries to return validated Pydantic models with automatic retry. The philosophy is "good LLM validation is just good validation."

**Key Features:**
- `response_model`: Define expected output type
- `max_retries`: Automatic retry on validation failure
- `llm_validator`: Semantic validation using another LLM

### API Design

**Basic Usage:**
```python
import instructor
from pydantic import BaseModel, Field, field_validator

class User(BaseModel):
    name: str = Field(..., min_length=2)
    age: int = Field(..., ge=0, le=150)

    @field_validator('age')
    @classmethod
    def validate_age(cls, v):
        if v < 0:
            raise ValueError("Age cannot be negative")
        return v

# Patch the client
client = instructor.from_provider("openai/gpt-4")

# Get validated output with retry
user = client.create(
    response_model=User,
    max_retries=3,
    messages=[{"role": "user", "content": "John is 25 years old"}]
)
print(user.name)  # "John"
print(user.age)   # 25
```

**Semantic Validation with LLM:**
```python
from typing import Annotated
from pydantic import BaseModel, BeforeValidator
from instructor import llm_validator

client = instructor.from_provider("openai/gpt-4.1-mini")

class ContentReview(BaseModel):
    content: Annotated[
        str,
        BeforeValidator(
            llm_validator(
                "Content must be family-friendly with no profanity",
                client=client
            )
        )
    ]

# Validation happens via LLM call
review = client.create(
    response_model=ContentReview,
    messages=[{"role": "user", "content": "Generate content"}]
)
```

**Error Handling:**
```python
from instructor.exceptions import InstructorRetryException, InstructorValidationError

try:
    user = client.create(
        response_model=User,
        max_retries=3,
        messages=[...]
    )
except InstructorRetryException as e:
    print(f"All {e.n_attempts} retries failed")
    print(f"Last error: {e.last_error}")
    for attempt in e.attempts:
        print(f"Attempt {attempt.number}: {attempt.exception}")
except InstructorValidationError as e:
    print(f"Validation failed: {e}")
```

**Tenacity Integration for Advanced Retry:**
```python
from tenacity import Retrying, stop_after_attempt, wait_exponential

retrying = Retrying(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=60)
)

user = client.create(
    response_model=User,
    max_retries=retrying,  # Pass tenacity Retrying object
    messages=[...]
)
```

### What They Do Well

1. **Simplicity**: Minimal API surface - just add `response_model` and `max_retries`
2. **Pure Pydantic**: Uses standard Pydantic validation, no new concepts
3. **Automatic Retry**: Sends validation errors back to LLM for correction
4. **Provider Abstraction**: Single API works across 15+ providers
5. **Semantic Validation**: `llm_validator` for complex, subjective criteria
6. **Tenacity Integration**: Leverage battle-tested retry strategies

### Limitations and Pain Points

1. **Retry Prompt Customization**: Users can't easily modify the "Recall the function correctly" retry prompt

2. **Native Structured Output Limitations**:
   - When using OpenAI's native structured output, you lose original completion on failure
   - Can't implement targeted retry without raw response

3. **Timeout with Retries**: Total timeout applies to all retries, can be unpredictable with larger models

4. **Validation vs. Content Quality**: Ensures schema adherence but not useful content
   - Perfectly formatted yet unhelpful responses possible

5. **Complex Validation Scenarios**: For complex cases, need custom validators or field-level validation

---

## Comparative Analysis

### Feature Comparison

| Feature | Guardrails AI | NeMo Guardrails | LangChain | Instructor |
|---------|---------------|-----------------|-----------|------------|
| **Primary Focus** | Validation pipeline | Dialog management | Parsing/chaining | Structured output |
| **Learning Curve** | Medium | High (Colang) | Low-Medium | Low |
| **Setup Complexity** | Medium (hub) | High (config) | Low | Very Low |
| **Retry Logic** | Yes (reask) | Yes | Partial | Yes (best) |
| **Composability** | Good | Good | Excellent | Limited |
| **Latency Overhead** | Medium | High | Low | Low |
| **Provider Support** | Multiple | Multiple | Extensive | 15+ |
| **Pydantic Native** | Yes | No | Yes | Yes |

### Validation Flow Comparison

**Guardrails AI:**
```
Input -> [Validators] -> LLM -> [Validators] -> Output
                              -> on_fail policy
```

**NeMo Guardrails:**
```
Input -> [Input Rails] -> [Dialog Rails] -> LLM
     -> [Output Rails] -> Output
     Uses Colang flows for control
```

**LangChain:**
```
Prompt (with format_instructions) -> LLM -> Parser -> Validated Output
                                         -> OutputFixingParser (optional)
```

**Instructor:**
```
Request (with response_model) -> LLM -> Pydantic Validation
                                     -> Retry with error context (if fails)
                                     -> Validated Output
```

---

## Patterns for Spell System

Based on this research, here are patterns that could inform a decorator-based spell system:

### 1. Simple Decorator Pattern (Instructor-Inspired)

```python
from spellcrafting import spell
from pydantic import BaseModel

class Summary(BaseModel):
    title: str
    points: list[str]

@spell(output=Summary, max_retries=3)
def summarize(text: str) -> Summary:
    """Summarize the given text into key points."""
    pass  # LLM handles implementation
```

**Why:** Instructor's simplicity is its strength. The decorator hides complexity while Pydantic provides validation.

### 2. Composable Guardrails (Guardrails AI-Inspired)

```python
from spellcrafting import spell, guard
from spellcrafting.guards import no_pii, max_length, semantic

@spell
@guard(no_pii(entities=["EMAIL", "PHONE"]))
@guard(max_length(1000))
@guard(semantic("Response must be professional"))
def generate_email(topic: str) -> str:
    """Generate a professional email."""
    pass
```

**Why:** Guardrails AI's composable validators via `.use()` translates well to stacked decorators.

### 3. On-Fail Strategies (Guardrails AI Pattern)

```python
from spellcrafting import spell, OnFail

@spell(
    on_fail=OnFail.RETRY,      # retry with error context
    # on_fail=OnFail.FIX,      # attempt programmatic fix
    # on_fail=OnFail.RAISE,    # raise exception
    # on_fail=custom_handler,  # custom function
    max_retries=3
)
def extract_data(text: str) -> dict:
    pass
```

### 4. Context-Aware Validation (Guardrails AI Metadata Pattern)

```python
@spell
def answer_question(question: str, sources: list[str]) -> str:
    """Answer based on provided sources."""
    pass

# Runtime context passed to validators
result = answer_question(
    "What is the capital?",
    sources=["France's capital is Paris"],
    _context={"require_citation": True}  # Validator receives this
)
```

### 5. Semantic Validation (Instructor Pattern)

```python
from spellcrafting import spell, validate

@spell
@validate.semantic("Response must not contain medical advice")
@validate.semantic("Response must be appropriate for all ages")
def chat(message: str) -> str:
    pass
```

**Why:** Instructor's `llm_validator` shows that some rules are best expressed in natural language.

### 6. Avoid: Dialog DSL (NeMo Lesson)

NeMo's Colang is powerful but adds significant complexity. For a decorator-based system:

**Don't do:**
```python
# Avoid separate DSL files
# Avoid complex flow definitions
# Avoid file-based configuration
```

**Do:**
```python
# Keep everything in Python
# Use decorators for flow control
# Configuration via function parameters
```

### Key Design Principles

1. **Pydantic First**: Use Pydantic for all validation - it's familiar and powerful
2. **Decorator Composition**: Stack decorators for multiple guardrails
3. **Sensible Defaults**: Work with zero configuration, customize when needed
4. **Transparent Retry**: Built-in retry with error context passed to LLM
5. **Low Latency**: Avoid NeMo's multi-step CoT overhead
6. **Provider Agnostic**: Single API across providers (like Instructor)
7. **Python Native**: No DSLs, no config files, no hub dependencies

### Recommended Minimal API

```python
from spellcrafting import spell
from pydantic import BaseModel, Field

class Output(BaseModel):
    result: str = Field(..., min_length=10)

# Simplest case - just works
@spell
def simple(prompt: str) -> str:
    """Generate a response."""
    pass

# With structured output and retry
@spell(output=Output, max_retries=3)
def structured(data: str) -> Output:
    """Extract structured data."""
    pass

# With composable guards
@spell
@guard.no_pii()
@guard.max_tokens(500)
def guarded(text: str) -> str:
    """Generate with guardrails."""
    pass
```

---

## Sources

### Guardrails AI
- [Guardrails AI Documentation](https://guardrailsai.com/docs)
- [Validators Concept](https://guardrailsai.com/docs/concepts/validators)
- [OnFail Actions](https://www.guardrailsai.com/docs/concepts/validator_on_fail_actions)
- [GitHub Repository](https://github.com/guardrails-ai/guardrails)
- [PyPI Package](https://pypi.org/project/guardrails-ai/)

### NeMo Guardrails
- [NVIDIA NeMo Guardrails Docs](https://docs.nvidia.com/nemo/guardrails/latest/index.html)
- [GitHub Repository](https://github.com/NVIDIA-NeMo/Guardrails)
- [Pinecone Tutorial](https://www.pinecone.io/learn/nemo-guardrails-intro/)
- [Configuration Guide](https://docs.nvidia.com/nemo/guardrails/latest/user-guides/configuration-guide.html)
- [Performance Issues Discussion](https://github.com/NVIDIA-NeMo/Guardrails/issues/154)

### LangChain
- [Structured Output Docs](https://docs.langchain.com/oss/python/langchain/structured-output)
- [PydanticOutputParser Reference](https://python.langchain.com/api_reference/core/output_parsers/langchain_core.output_parsers.pydantic.PydanticOutputParser.html)
- [Output Parser Fixing Guide](https://python.langchain.com/docs/how_to/output_parser_fixing/)

### Instructor
- [Instructor Documentation](https://python.useinstructor.com/)
- [Why Instructor](https://python.useinstructor.com/why/)
- [Validation Concepts](https://python.useinstructor.com/concepts/validation/)
- [Retry Mechanisms](https://python.useinstructor.com/learning/validation/retry_mechanisms/)
- [PyPI Package](https://pypi.org/project/instructor/)
