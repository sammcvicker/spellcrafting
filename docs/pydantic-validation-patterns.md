# Pydantic Validation Patterns for Spell Outputs

When you use a Pydantic model as your spell's return type, you get powerful validation capabilities for free. This guide shows common patterns for robust output validation.

## How It Works

The `@spell` decorator uses [PydanticAI](https://ai.pydantic.dev/) under the hood. Your return type annotation becomes the output schema:

```python
from spellcrafting import spell
from pydantic import BaseModel

class Analysis(BaseModel):
    sentiment: str
    confidence: float

@spell(model="anthropic:claude-sonnet-4-20250514")
def analyze(text: str) -> Analysis:
    """Analyze the sentiment of the text."""
    ...
```

When the LLM's response doesn't match your schema, PydanticAI automatically:
1. Captures the validation error
2. Sends the error back to the LLM as feedback
3. Asks the LLM to regenerate a corrected response
4. Repeats up to `retries` times (default: 1)

This means your Pydantic validators don't just validate—they teach the LLM to produce correct outputs.

---

## 1. Constrained Values

### Literal Types

Use `Literal` to restrict a field to specific values:

```python
from typing import Literal
from pydantic import BaseModel

class SentimentResult(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    text: str

@spell(model="fast", retries=2)
def classify_sentiment(text: str) -> SentimentResult:
    """Classify the sentiment as positive, negative, or neutral."""
    ...
```

The LLM can only return one of these three values. If it tries `"happy"` or `"Positive"` (wrong case), Pydantic rejects it and the LLM corrects itself.

### Enums

Use `Enum` for typed options with additional semantics:

```python
from enum import Enum
from pydantic import BaseModel

class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class Task(BaseModel):
    title: str
    priority: Priority

@spell(model="fast")
def create_task(description: str) -> Task:
    """Create a task from the description with appropriate priority."""
    ...
```

### Constrained Primitives

Use Pydantic's `constr`, `conint`, and `confloat` for constrained primitives:

```python
from pydantic import BaseModel, constr, conint, confloat

class UserProfile(BaseModel):
    username: constr(min_length=3, max_length=20, pattern=r"^[a-z0-9_]+$")
    age: conint(ge=0, le=150)
    rating: confloat(ge=0.0, le=5.0)
```

---

## 2. Numeric Bounds

### Confidence Scores

The most common pattern—bounded floats for confidence:

```python
from pydantic import BaseModel, Field

class Answer(BaseModel):
    response: str
    confidence: float = Field(ge=0.0, le=1.0)

@spell(model="fast")
def answer_question(question: str) -> Answer:
    """Answer the question and rate your confidence from 0 to 1."""
    ...
```

### Ratings and Scores

```python
class Review(BaseModel):
    summary: str
    rating: int = Field(ge=1, le=5, description="Star rating from 1-5")
    pros: list[str]
    cons: list[str]
```

### Using Annotated

The `Annotated` syntax is often cleaner:

```python
from typing import Annotated
from pydantic import Field

Confidence = Annotated[float, Field(ge=0.0, le=1.0)]
Rating = Annotated[int, Field(ge=1, le=5)]

class Response(BaseModel):
    answer: str
    confidence: Confidence
    quality_rating: Rating
```

---

## 3. List Constraints

### Bounded Lists

Require a minimum or maximum number of items:

```python
from pydantic import BaseModel, Field

class Summary(BaseModel):
    title: str
    key_points: list[str] = Field(min_length=1, max_length=5)

@spell(model="fast", retries=2)
def summarize(text: str) -> Summary:
    """Summarize the text with 1-5 key bullet points."""
    ...
```

### Non-Empty Lists

```python
class Analysis(BaseModel):
    topics: list[str] = Field(min_length=1)  # At least one topic required
```

### Fixed-Length Lists

```python
class Comparison(BaseModel):
    # Exactly 3 pros and 3 cons
    pros: list[str] = Field(min_length=3, max_length=3)
    cons: list[str] = Field(min_length=3, max_length=3)
```

---

## 4. Field Validators

### Cleaning and Transforming

Use `@field_validator` to clean up LLM outputs:

```python
from pydantic import BaseModel, field_validator

class CleanResponse(BaseModel):
    content: str
    tags: list[str]

    @field_validator("content")
    @classmethod
    def clean_whitespace(cls, v: str) -> str:
        """Normalize whitespace in content."""
        return " ".join(v.split()).strip()

    @field_validator("tags")
    @classmethod
    def clean_tags(cls, v: list[str]) -> list[str]:
        """Remove empty tags and normalize."""
        return [tag.strip().lower() for tag in v if tag.strip()]
```

### Validation Rules

```python
class Email(BaseModel):
    subject: str
    body: str

    @field_validator("subject")
    @classmethod
    def subject_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Subject cannot be empty")
        if len(v) > 100:
            raise ValueError("Subject too long (max 100 chars)")
        return v.strip()
```

### Using BeforeValidator and AfterValidator

For reusable validators, use `Annotated` with validators:

```python
from typing import Annotated
from pydantic import BeforeValidator, AfterValidator

def strip_whitespace(v: str) -> str:
    return " ".join(v.split()).strip()

def must_not_be_empty(v: str) -> str:
    if not v:
        raise ValueError("Cannot be empty")
    return v

CleanStr = Annotated[str, BeforeValidator(strip_whitespace)]
NonEmptyStr = Annotated[str, AfterValidator(must_not_be_empty)]
CleanNonEmptyStr = Annotated[str, BeforeValidator(strip_whitespace), AfterValidator(must_not_be_empty)]

class Document(BaseModel):
    title: CleanNonEmptyStr
    body: CleanStr
```

---

## 5. Model Validators

### Cross-Field Validation

Use `@model_validator` when fields depend on each other:

```python
from pydantic import BaseModel, model_validator
from datetime import date

class DateRange(BaseModel):
    start_date: date
    end_date: date

    @model_validator(mode="after")
    def validate_range(self) -> "DateRange":
        if self.end_date < self.start_date:
            raise ValueError("end_date must be after start_date")
        return self

@spell(model="fast")
def extract_dates(text: str) -> DateRange:
    """Extract the date range mentioned in the text."""
    ...
```

### Conditional Requirements

```python
class SearchResult(BaseModel):
    found: bool
    result: str | None = None
    error_message: str | None = None

    @model_validator(mode="after")
    def check_consistency(self) -> "SearchResult":
        if self.found and not self.result:
            raise ValueError("result required when found=True")
        if not self.found and not self.error_message:
            raise ValueError("error_message required when found=False")
        return self
```

### Complex Business Rules

```python
class Order(BaseModel):
    items: list[str]
    total: float
    discount_percent: float = Field(ge=0, le=100)
    final_total: float

    @model_validator(mode="after")
    def validate_totals(self) -> "Order":
        expected = self.total * (1 - self.discount_percent / 100)
        if abs(self.final_total - expected) > 0.01:
            raise ValueError(
                f"final_total {self.final_total} doesn't match "
                f"total {self.total} with {self.discount_percent}% discount"
            )
        return self
```

---

## 6. Nested Models

### Structured Hierarchies

```python
class Address(BaseModel):
    street: str
    city: str
    country: str
    postal_code: str | None = None

class Person(BaseModel):
    name: str
    email: str
    address: Address

class Company(BaseModel):
    name: str
    employees: list[Person]
    headquarters: Address

@spell(model="reasoning")
def extract_company_info(document: str) -> Company:
    """Extract structured company information from the document."""
    ...
```

### Optional Nested Fields

```python
class Article(BaseModel):
    title: str
    author: str
    metadata: "ArticleMetadata | None" = None

class ArticleMetadata(BaseModel):
    published_date: date | None = None
    word_count: int | None = None
    tags: list[str] = []
```

### Lists of Models

```python
class Step(BaseModel):
    number: int
    instruction: str
    duration_minutes: int | None = None

class Recipe(BaseModel):
    name: str
    ingredients: list[str] = Field(min_length=1)
    steps: list[Step] = Field(min_length=1)

@spell(model="fast")
def parse_recipe(text: str) -> Recipe:
    """Parse a recipe from the text."""
    ...
```

---

## 7. Common Patterns

### URL Validation

```python
from pydantic import BaseModel, HttpUrl

class WebPage(BaseModel):
    title: str
    url: HttpUrl  # Validates URL format
    description: str | None = None
```

### Email Validation

```python
from pydantic import BaseModel, EmailStr

class Contact(BaseModel):
    name: str
    email: EmailStr  # Validates email format
```

### Date and Time

```python
from datetime import date, datetime, time
from pydantic import BaseModel

class Event(BaseModel):
    name: str
    date: date
    start_time: time
    created_at: datetime
```

### Optional with Defaults

```python
class Config(BaseModel):
    name: str
    enabled: bool = True
    max_retries: int = 3
    timeout_seconds: float | None = None
```

---

## 8. Retry Behavior

### How Retries Work

When validation fails, PydanticAI sends the error back to the LLM:

```python
@spell(model="fast", retries=3)  # Up to 3 attempts
def analyze(text: str) -> Analysis:
    """Analyze the text."""
    ...
```

**Retry flow:**
1. LLM generates response
2. Pydantic validates against schema
3. On failure: error message sent to LLM as feedback
4. LLM regenerates with correction
5. Repeat until success or max retries

### When to Increase Retries

Increase retries when:
- Schema is complex (nested models, many constraints)
- LLM sometimes makes format errors (JSON syntax)
- Validation rules are nuanced (cross-field dependencies)

```python
# Complex output needs more retries
@spell(model="fast", retries=3)
def complex_extraction(doc: str) -> ComplexModel:
    ...

# Simple output needs fewer
@spell(model="fast", retries=1)
def simple_classification(text: str) -> SimpleLabel:
    ...
```

### Signs Your Schema Is Too Strict

If you're seeing repeated failures even with retries:

1. **Constraints too tight**: Relax bounds (e.g., `le=1.0` → `le=1.1` to allow for floating-point)
2. **Ambiguous field names**: Rename to be clearer in the schema
3. **Add descriptions**: Use `Field(description="...")` to guide the LLM
4. **Simplify validators**: Complex logic confuses LLMs

```python
# Help the LLM understand what you want
class Response(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="The overall sentiment. Use 'neutral' if mixed or unclear."
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="How confident you are, from 0 (guessing) to 1 (certain)"
    )
```

---

## 9. Using on_fail Strategies

When validation fails after all retries, you can specify what happens next:

### Escalate to a Better Model

```python
from spellcrafting import spell, OnFail

@spell(model="fast", retries=2, on_fail=OnFail.escalate("reasoning"))
def analyze(text: str) -> Analysis:
    """Analyze the text."""
    ...
```

If the fast model fails validation, it automatically retries with the `reasoning` model.

### Return a Fallback Value

```python
default = Analysis(sentiment="neutral", confidence=0.0)

@spell(model="fast", retries=2, on_fail=OnFail.fallback(default))
def analyze(text: str) -> Analysis:
    """Analyze the text."""
    ...
```

### Custom Error Handler

```python
def handle_failure(error: Exception, attempt: int, context: dict) -> Analysis:
    # Log error, return safe default, etc.
    return Analysis(sentiment="error", confidence=0.0)

@spell(model="fast", retries=2, on_fail=OnFail.custom(handle_failure))
def analyze(text: str) -> Analysis:
    """Analyze the text."""
    ...
```

---

## 10. LLM-Powered Validators

For validation that requires understanding, use `llm_validator`:

```python
from pydantic import BaseModel, BeforeValidator
from typing import Annotated
from spellcrafting import llm_validator

# The LLM validates that content is family-friendly
FamilyFriendly = Annotated[str, BeforeValidator(llm_validator("Must be family-friendly"))]

class Comment(BaseModel):
    content: FamilyFriendly
    author: str

@spell(model="fast")
def summarize_discussion(posts: list[str]) -> Comment:
    """Summarize the discussion."""
    ...
```

The `llm_validator` can also auto-fix values:

```python
# Auto-correct values that fail validation
ProfessionalTone = Annotated[str, BeforeValidator(
    llm_validator("Must use professional business language", on_fail="fix")
)]
```

---

## Quick Reference

| Pattern | Pydantic Feature | Example |
|---------|------------------|---------|
| Fixed options | `Literal["a", "b"]` | `status: Literal["open", "closed"]` |
| Typed options | `Enum` | `priority: Priority` |
| Bounded numbers | `Field(ge=, le=)` | `score: float = Field(ge=0, le=1)` |
| String length | `constr(min_length=, max_length=)` | `name: constr(min_length=1)` |
| String pattern | `constr(pattern=)` | `code: constr(pattern=r"^[A-Z]{3}$")` |
| List bounds | `Field(min_length=, max_length=)` | `items: list[str] = Field(min_length=1)` |
| Field validation | `@field_validator` | Custom cleaning/rules |
| Cross-field rules | `@model_validator` | Dependent field validation |
| Nested structure | Composed `BaseModel` | `address: Address` |
| Optional fields | `field: Type \| None = None` | `bio: str \| None = None` |
| Default values | `field: Type = default` | `enabled: bool = True` |
| URL validation | `HttpUrl` | `website: HttpUrl` |
| Email validation | `EmailStr` | `email: EmailStr` |

---

## Further Reading

- [Pydantic v2 Documentation](https://docs.pydantic.dev/)
- [PydanticAI Structured Outputs](https://ai.pydantic.dev/)
- [Pydantic Field Types](https://docs.pydantic.dev/latest/concepts/types/)
- [Pydantic Validators](https://docs.pydantic.dev/latest/concepts/validators/)
