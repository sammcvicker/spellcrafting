# Enterprise Guardrails Patterns for LLM Applications

> Research compiled for production-grade LLM library development
> Last updated: December 2024

## Implementation Status

This research document informed the design of spellcrafting's observability and configuration systems. Key decisions:

**Implemented:**
- `SpellExecutionLog` dataclass for structured audit logging (section 4.1 pattern)
- `ValidationMetrics` tracking guard passes/failures and retry attempts
- Model configuration via `pyproject.toml` with hierarchical resolution
- Default timeout handling (2 minutes) to prevent indefinite hangs

**Partially Implemented:**
- Basic retry logic via PydanticAI's built-in retry mechanism
- Observability via `SpellExecutionLog` (simpler than full OpenTelemetry integration)

**Deferred to external tools (as recommended):**
- Rate limiting and token budgets (use API gateway or liteLLM proxy)
- Circuit breakers (use tenacity or external service mesh)
- Cost estimation and caps (use provider dashboards or specialized tools)
- HIPAA/GDPR compliance checks (use specialized compliance tools)
- Human-in-the-loop workflows (application-specific, not library-level)

**Rationale:** These enterprise patterns are better handled at the infrastructure/gateway level rather than embedded in the spell library. See `logging.py` and `config.py` for the implemented patterns.

## Executive Summary

Enterprise LLM deployments require comprehensive guardrails spanning compliance, cost control, reliability, and observability. This document synthesizes production patterns and best practices from industry leaders to guide the development of an enterprise-grade LLM library.

---

## 1. Compliance and Governance

### 1.1 Audit Logging Requirements

Audit logs are foundational for enterprise LLM deployments, serving security, compliance, and operational purposes.

#### What to Log

| Category | Data Points | Purpose |
|----------|-------------|---------|
| **Request Metadata** | Timestamp, user ID, session ID, API key hash | Traceability |
| **Input/Output** | Prompt hash/content, response hash/content, token counts | Compliance audit |
| **Model Details** | Model ID, version, provider, temperature, parameters | Reproducibility |
| **Guardrail Events** | Triggered rules, block reasons, confidence scores | Security analysis |
| **Performance** | Latency, token throughput, queue time | Optimization |
| **Cost** | Token usage, estimated cost, budget remaining | Financial control |

#### Retention Requirements by Framework

| Regulation | Retention Period | Special Requirements |
|------------|------------------|---------------------|
| **HIPAA** | 6 years minimum | PHI access tracking, breach notification logs |
| **GDPR** | Purpose-based (often 3-5 years) | Right to erasure, data minimization |
| **SOC 2** | 1 year minimum | Change management, access control logs |
| **Financial (SOX)** | 7 years | Immutable audit trail |

**Best Practice**: Implement tiered retention with 30-90 day TTLs for detailed traces and longer retention for aggregated compliance data.

```python
# Example: Structured audit log entry
@dataclass
class AuditLogEntry:
    timestamp: datetime
    request_id: str
    user_id: str
    team_id: str

    # Request details
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    input_hash: str  # For PII-safe logging

    # Guardrail events
    guardrails_triggered: List[str]
    guardrail_action: Literal["allow", "block", "warn", "redact"]
    confidence_scores: Dict[str, float]

    # Cost tracking
    estimated_cost_usd: float
    budget_remaining_usd: float

    # Compliance metadata
    data_classification: str
    retention_policy: str
```

### 1.2 Data Residency Concerns

Data residency is critical for regulated industries and organizations operating across jurisdictions.

#### Key Concepts

- **Data Residency**: Physical location where data is stored
- **Data Sovereignty**: Legal authority a government has over data based on location
- **Four Tenets of Digital Sovereignty**: Data residency, data privacy, security/resiliency, legal controls

#### Implementation Patterns

```python
class DataResidencyConfig:
    """Configuration for data residency requirements."""

    region: str  # e.g., "eu-west-1", "us-east-1"
    allowed_regions: List[str]
    blocked_regions: List[str]

    # Provider routing based on residency
    provider_preferences: Dict[str, str]  # region -> provider

    # Data handling
    log_residency: str  # Where logs must be stored
    cache_residency: str  # Where cache can exist

    # Legal
    subject_to_cloud_act: bool
    requires_eu_legal_entity: bool
```

#### Enterprise Patterns

1. **Regional Provider Routing**: Route requests to providers with data centers in required regions
2. **Local LLM Fallback**: Fall back to self-hosted models when residency requirements cannot be met
3. **Data Minimization**: Strip PII before sending to external providers
4. **Geo-fencing**: Enforce policies that restrict data processing and movement

**Sovereign Cloud Options**:
- Azure OpenAI with Microsoft for Sovereignty
- AWS Bedrock with regional isolation
- Private LLM deployments (on-premise or dedicated cloud)

### 1.3 Regulatory Frameworks

#### GDPR Implications

| Requirement | LLM Implementation |
|-------------|-------------------|
| **Lawful Basis** | Consent or legitimate interest for AI processing |
| **Data Minimization** | Only send necessary data to models |
| **Purpose Limitation** | Define and enforce allowed use cases |
| **Right to Erasure** | Ability to delete user data from logs/caches |
| **Data Protection Impact Assessment** | Required for high-risk AI processing |
| **DPA with Providers** | Business Associate Agreements for data processors |

#### HIPAA Implications

| Requirement | LLM Implementation |
|-------------|-------------------|
| **BAA Requirement** | Provider must sign Business Associate Agreement |
| **PHI Protection** | Encryption in transit and at rest |
| **Access Controls** | Role-based access to PHI-containing prompts |
| **Audit Logging** | Track all PHI access with user attribution |
| **Breach Notification** | Detect and report PHI exposure |
| **Minimum Necessary** | Limit PHI in prompts to what's required |

```python
class ComplianceGuardrail:
    """Guardrail for regulatory compliance."""

    def check_hipaa_compliance(self, request: LLMRequest) -> GuardrailResult:
        """Check HIPAA compliance before processing."""
        issues = []

        # Check for PHI in prompt
        if self.contains_phi(request.prompt):
            if not request.provider.has_baa:
                issues.append("Provider lacks BAA for PHI processing")
            if not request.user.hipaa_authorized:
                issues.append("User not authorized for PHI access")

        # Check data residency
        if not self.is_hipaa_compliant_region(request.provider.region):
            issues.append(f"Region {request.provider.region} not HIPAA compliant")

        return GuardrailResult(
            action="block" if issues else "allow",
            reasons=issues,
            audit_required=True
        )
```

### 1.4 Human-in-the-Loop Patterns

HITL is essential for high-stakes decisions and regulatory compliance.

#### Pattern Types

| Pattern | Use Case | Implementation |
|---------|----------|----------------|
| **Approval Flows** | High-risk actions, PII handling | Pause workflow until human approves |
| **Confidence-Based Routing** | Uncertain responses | Route to human when confidence < threshold |
| **Breakpoints** | Critical decision points | Static or dynamic pause points in workflow |
| **Async Review** | Non-blocking review | Queue for later human review via Slack/email |
| **Sampling Review** | Quality assurance | Review random sample of responses |

```python
class HITLConfig:
    """Human-in-the-loop configuration."""

    # Trigger conditions
    confidence_threshold: float = 0.7  # Route to human if below
    cost_threshold_usd: float = 10.0   # Require approval above
    pii_detected: bool = True          # Always require review

    # Review channels
    sync_approval: bool = False        # Block until approved
    async_channel: str = "slack"       # slack, email, dashboard
    escalation_timeout: timedelta = timedelta(hours=4)

    # Fallback behavior
    on_timeout: Literal["block", "allow_with_warning", "use_fallback"]

    # Audit
    log_all_decisions: bool = True
    require_justification: bool = True
```

#### Implementation Example

```python
class HITLGuardrail:
    """Human-in-the-loop guardrail implementation."""

    async def check(self, request: LLMRequest, response: LLMResponse) -> HITLResult:
        # Check if HITL is required
        triggers = []

        if response.confidence < self.config.confidence_threshold:
            triggers.append(f"Low confidence: {response.confidence:.2f}")

        if response.estimated_cost > self.config.cost_threshold_usd:
            triggers.append(f"High cost: ${response.estimated_cost:.2f}")

        if self.pii_detector.scan(response.content):
            triggers.append("PII detected in response")

        if not triggers:
            return HITLResult(action="proceed", reviewed=False)

        # Route to human review
        if self.config.sync_approval:
            decision = await self.request_sync_approval(request, response, triggers)
        else:
            await self.queue_async_review(request, response, triggers)
            decision = self.config.on_timeout  # Proceed with default action

        return HITLResult(
            action=decision,
            triggers=triggers,
            reviewed=True
        )
```

---

## 2. Cost and Rate Controls

### 2.1 Token Budget Enforcement

Token budgets prevent runaway costs and ensure fair resource allocation.

#### Hierarchical Budget Model

```
Organization Budget ($10,000/month)
├── Team A Budget ($3,000/month)
│   ├── Project 1 ($1,000/month)
│   └── Project 2 ($2,000/month)
├── Team B Budget ($5,000/month)
│   ├── API Key 1 ($2,000/month)
│   └── API Key 2 ($3,000/month)
└── Team C Budget ($2,000/month)
```

**Key Principle**: All applicable budgets are checked for every request. A request only proceeds if ALL levels have sufficient remaining balance.

```python
@dataclass
class BudgetConfig:
    """Token and cost budget configuration."""

    # Budget limits
    max_tokens_per_request: int = 4096
    max_tokens_per_minute: int = 100_000
    max_tokens_per_day: int = 1_000_000
    max_cost_per_request_usd: float = 1.0
    max_cost_per_day_usd: float = 100.0
    max_cost_per_month_usd: float = 1000.0

    # Budget reset
    reset_period: Literal["never", "daily", "weekly", "monthly"] = "monthly"

    # Soft limits for warnings
    warning_threshold_percent: float = 0.8

    # Actions when exceeded
    on_soft_limit: Literal["warn", "throttle"] = "warn"
    on_hard_limit: Literal["block", "queue", "use_fallback"] = "block"


class BudgetEnforcer:
    """Enforce token and cost budgets."""

    def check_budget(self, request: LLMRequest) -> BudgetResult:
        """Check all applicable budgets before request."""

        # Estimate cost
        estimated_tokens = self.estimate_tokens(request)
        estimated_cost = self.estimate_cost(estimated_tokens, request.model)

        # Check hierarchical budgets
        budgets_to_check = [
            self.get_org_budget(request.org_id),
            self.get_team_budget(request.team_id),
            self.get_user_budget(request.user_id),
            self.get_key_budget(request.api_key),
        ]

        for budget in budgets_to_check:
            remaining = budget.remaining_usd

            if remaining < estimated_cost:
                return BudgetResult(
                    allowed=False,
                    reason=f"Budget exceeded for {budget.name}",
                    remaining_usd=remaining,
                    estimated_cost_usd=estimated_cost
                )

            if remaining < estimated_cost / (1 - budget.warning_threshold):
                self.emit_warning(budget, remaining, estimated_cost)

        return BudgetResult(allowed=True, estimated_cost_usd=estimated_cost)

    def deduct_cost(self, request: LLMRequest, actual_cost: float):
        """Deduct cost from all applicable budgets after completion."""
        for budget in self.get_applicable_budgets(request):
            budget.deduct(actual_cost)
            self.check_alerts(budget)
```

### 2.2 Rate Limiting Patterns

Traditional request-per-second limiting is insufficient for LLMs due to variable request costs.

#### Rate Limiting Approaches

| Approach | Pros | Cons | Best For |
|----------|------|------|----------|
| **Requests/minute (RPM)** | Simple to implement | Unfair for mixed loads | Basic protection |
| **Tokens/minute (TPM)** | Fair cost allocation | Requires estimation | Primary limiter |
| **Prompt TPM / Output TPM** | Granular control | Complex to track | Heavy input/output workloads |
| **Concurrency limits** | Prevents queue buildup | Can underutilize capacity | Streaming UIs |
| **Cost/minute** | Direct budget control | Requires accurate pricing | Cost-sensitive apps |

```python
class RateLimiter:
    """Multi-dimensional rate limiter for LLM APIs."""

    def __init__(self, config: RateLimitConfig):
        self.rpm_limiter = TokenBucket(config.requests_per_minute)
        self.tpm_limiter = TokenBucket(config.tokens_per_minute)
        self.concurrency_limiter = Semaphore(config.max_concurrent)
        self.cost_limiter = SlidingWindow(config.cost_per_minute)

    async def acquire(self, request: LLMRequest) -> RateLimitResult:
        """Acquire rate limit tokens before request."""

        estimated_tokens = self.estimate_tokens(request)
        estimated_cost = self.estimate_cost(request)

        # Check all limiters
        if not self.rpm_limiter.try_acquire(1):
            return RateLimitResult(
                allowed=False,
                reason="Request rate limit exceeded",
                retry_after=self.rpm_limiter.time_to_next_token()
            )

        if not self.tpm_limiter.try_acquire(estimated_tokens):
            return RateLimitResult(
                allowed=False,
                reason="Token rate limit exceeded",
                retry_after=self.tpm_limiter.time_to_next_token(estimated_tokens)
            )

        if not self.concurrency_limiter.try_acquire():
            return RateLimitResult(
                allowed=False,
                reason="Concurrency limit reached",
                retry_after=None  # Wait for any request to complete
            )

        return RateLimitResult(allowed=True)
```

### 2.3 Cost Estimation and Caps

```python
class CostEstimator:
    """Estimate and track LLM costs."""

    # Pricing per 1M tokens (example rates)
    PRICING = {
        "gpt-4": {"input": 30.0, "output": 60.0},
        "gpt-4-turbo": {"input": 10.0, "output": 30.0},
        "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
        "claude-3-opus": {"input": 15.0, "output": 75.0},
        "claude-3-sonnet": {"input": 3.0, "output": 15.0},
        "claude-3-haiku": {"input": 0.25, "output": 1.25},
    }

    def estimate_request_cost(
        self,
        model: str,
        input_tokens: int,
        estimated_output_tokens: int
    ) -> float:
        """Estimate cost before request."""
        pricing = self.PRICING.get(model, {"input": 10.0, "output": 30.0})

        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (estimated_output_tokens / 1_000_000) * pricing["output"]

        return input_cost + output_cost

    def calculate_actual_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """Calculate actual cost after request."""
        pricing = self.PRICING[model]

        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]

        return input_cost + output_cost
```

### 2.4 Fallback Strategies When Limits Hit

```python
class FallbackStrategy:
    """Strategies for handling rate/budget limit hits."""

    async def handle_limit_exceeded(
        self,
        request: LLMRequest,
        limit_type: str,
        retry_after: Optional[float]
    ) -> FallbackResult:

        strategy = self.get_strategy(request, limit_type)

        if strategy == "queue":
            # Queue for later execution
            job_id = await self.queue_request(request, retry_after)
            return FallbackResult(action="queued", job_id=job_id)

        elif strategy == "fallback_model":
            # Use cheaper/different model
            fallback_model = self.get_fallback_model(request.model)
            return FallbackResult(
                action="redirect",
                new_model=fallback_model
            )

        elif strategy == "fallback_provider":
            # Try different provider
            fallback_provider = self.get_fallback_provider(request.provider)
            return FallbackResult(
                action="redirect",
                new_provider=fallback_provider
            )

        elif strategy == "cached_response":
            # Return cached/similar response
            cached = await self.find_similar_cached(request)
            if cached:
                return FallbackResult(action="cached", response=cached)

        elif strategy == "graceful_degrade":
            # Return degraded response
            return FallbackResult(
                action="degraded",
                message="Service temporarily limited. Please try again later."
            )

        # Default: reject
        return FallbackResult(
            action="rejected",
            retry_after=retry_after
        )
```

---

## 3. Reliability Patterns

### 3.1 Circuit Breakers

Circuit breakers prevent cascading failures by cutting off traffic to unhealthy services.

#### Circuit Breaker States

```
     ┌──────────┐
     │  CLOSED  │◄────────────────┐
     │ (normal) │                 │
     └────┬─────┘                 │
          │ failures > threshold  │
          ▼                       │ success
     ┌──────────┐                 │
     │   OPEN   │                 │
     │ (failing)│                 │
     └────┬─────┘                 │
          │ cooldown expired      │
          ▼                       │
     ┌──────────┐                 │
     │HALF-OPEN │─────────────────┘
     │ (probing)│
     └──────────┘
          │ failure
          ▼
     ┌──────────┐
     │   OPEN   │
     └──────────┘
```

```python
class CircuitBreaker:
    """Circuit breaker for LLM provider calls."""

    def __init__(self, config: CircuitBreakerConfig):
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.config = config

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""

        if self.state == CircuitState.OPEN:
            if self._cooldown_expired():
                self.state = CircuitState.HALF_OPEN
            else:
                raise CircuitOpenError(
                    f"Circuit open, retry after {self._time_until_retry()}s"
                )

        try:
            result = await func(*args, **kwargs)
            self._record_success()
            return result
        except Exception as e:
            self._record_failure(e)
            raise

    def _record_failure(self, error: Exception):
        """Record a failure and potentially trip the circuit."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == CircuitState.HALF_OPEN:
            # Any failure in half-open state trips the circuit
            self.state = CircuitState.OPEN
            self.failure_count = 0

        elif self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
            self.failure_count = 0
            self._emit_circuit_opened_event()

    def _record_success(self):
        """Record a success and potentially close the circuit."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.success_count = 0
                self._emit_circuit_closed_event()
        else:
            self.failure_count = max(0, self.failure_count - 1)
```

### 3.2 Retry with Exponential Backoff

```python
class RetryConfig:
    """Configuration for retry behavior."""

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    jitter_factor: float = 0.1

    # Retryable errors
    retryable_status_codes: Set[int] = {429, 500, 502, 503, 504}
    retryable_exceptions: Set[Type[Exception]] = {TimeoutError, ConnectionError}

    # Non-retryable (fail fast)
    non_retryable_status_codes: Set[int] = {400, 401, 403, 404}


class RetryHandler:
    """Retry handler with exponential backoff and jitter."""

    def __init__(self, config: RetryConfig):
        self.config = config

    async def execute_with_retry(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute function with retry logic."""

        last_exception = None

        for attempt in range(self.config.max_retries + 1):
            try:
                return await func(*args, **kwargs)

            except Exception as e:
                last_exception = e

                if not self._is_retryable(e):
                    raise

                if attempt == self.config.max_retries:
                    raise

                delay = self._calculate_delay(attempt, e)

                self._log_retry(attempt, delay, e)
                await asyncio.sleep(delay)

        raise last_exception

    def _calculate_delay(self, attempt: int, error: Exception) -> float:
        """Calculate delay with exponential backoff and jitter."""

        # Check for Retry-After header
        if hasattr(error, 'retry_after') and error.retry_after:
            return min(error.retry_after, self.config.max_delay)

        # Exponential backoff
        delay = self.config.base_delay * (self.config.exponential_base ** attempt)
        delay = min(delay, self.config.max_delay)

        # Add jitter to prevent thundering herd
        if self.config.jitter:
            jitter = random.uniform(0, self.config.jitter_factor * delay)
            delay += jitter

        return delay

    def _is_retryable(self, error: Exception) -> bool:
        """Determine if error is retryable."""
        if isinstance(error, HTTPError):
            if error.status_code in self.config.non_retryable_status_codes:
                return False
            return error.status_code in self.config.retryable_status_codes

        return type(error) in self.config.retryable_exceptions
```

### 3.3 Fallback Model Chains

```python
class FallbackChain:
    """Fallback chain for model/provider failures."""

    def __init__(self, chain: List[ProviderConfig]):
        self.chain = chain
        self.health_tracker = HealthTracker()

    async def execute(self, request: LLMRequest) -> LLMResponse:
        """Execute request with fallback chain."""

        errors = []

        for provider_config in self.chain:
            # Skip unhealthy providers
            if not self.health_tracker.is_healthy(provider_config.name):
                continue

            try:
                client = self.get_client(provider_config)
                response = await client.complete(request)

                # Track success for health
                self.health_tracker.record_success(provider_config.name)

                return response

            except Exception as e:
                errors.append((provider_config.name, e))
                self.health_tracker.record_failure(provider_config.name)

                # Check if we should continue trying
                if not self._should_continue(e):
                    break

        # All providers failed
        raise FallbackExhaustedError(
            f"All providers failed: {errors}",
            errors=errors
        )

    def _should_continue(self, error: Exception) -> bool:
        """Determine if we should try the next provider."""
        # Don't continue for client errors (bad request)
        if isinstance(error, HTTPError) and 400 <= error.status_code < 500:
            return False
        return True


# Example fallback chain configuration
FALLBACK_CHAIN = [
    ProviderConfig(
        name="openai",
        model="gpt-4-turbo",
        priority=1,
        max_cost=1.0
    ),
    ProviderConfig(
        name="anthropic",
        model="claude-3-sonnet",
        priority=2,
        max_cost=0.5
    ),
    ProviderConfig(
        name="openai",
        model="gpt-3.5-turbo",
        priority=3,
        max_cost=0.1,
        quality_degradation=True
    ),
]
```

### 3.4 Timeout Handling

```python
class TimeoutConfig:
    """Timeout configuration for LLM requests."""

    # Connection timeout
    connect_timeout: float = 10.0

    # Read timeout (for streaming)
    read_timeout: float = 300.0  # 5 minutes for complex tasks

    # Total request timeout
    total_timeout: float = 600.0  # 10 minutes max

    # Streaming-specific
    chunk_timeout: float = 30.0  # Max time between chunks

    # Adaptive timeout based on request
    adaptive: bool = True
    tokens_per_second_estimate: float = 50.0


class TimeoutHandler:
    """Handle timeouts for LLM requests."""

    def __init__(self, config: TimeoutConfig):
        self.config = config

    def get_timeout(self, request: LLMRequest) -> float:
        """Calculate appropriate timeout for request."""

        if not self.config.adaptive:
            return self.config.total_timeout

        # Estimate based on expected output tokens
        estimated_output = request.max_tokens or 4096
        estimated_time = estimated_output / self.config.tokens_per_second_estimate

        # Add buffer for network latency and processing
        timeout = estimated_time * 1.5 + 30

        return min(timeout, self.config.total_timeout)

    async def execute_with_timeout(
        self,
        func: Callable,
        request: LLMRequest,
        **kwargs
    ) -> LLMResponse:
        """Execute with appropriate timeout."""

        timeout = self.get_timeout(request)

        try:
            return await asyncio.wait_for(
                func(request, **kwargs),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            raise LLMTimeoutError(
                f"Request timed out after {timeout}s",
                request_id=request.id,
                estimated_tokens=request.max_tokens
            )
```

---

## 4. Observability for Guardrails

### 4.1 What to Log When Guardrails Trigger

```python
@dataclass
class GuardrailEvent:
    """Structured guardrail event for logging."""

    # Event identification
    event_id: str
    timestamp: datetime
    request_id: str

    # Guardrail details
    guardrail_name: str
    guardrail_version: str
    guardrail_type: Literal["input", "output", "cost", "rate", "compliance"]

    # Trigger details
    action_taken: Literal["allow", "block", "warn", "redact", "modify"]
    trigger_reason: str
    confidence_score: float

    # Context
    user_id: str
    team_id: str
    model: str
    provider: str

    # Content (sanitized)
    input_excerpt: str  # First 100 chars, PII redacted
    output_excerpt: str  # First 100 chars, PII redacted

    # Impact
    tokens_blocked: int
    estimated_cost_blocked: float
    latency_added_ms: float


class GuardrailLogger:
    """Logger for guardrail events."""

    def log_trigger(self, event: GuardrailEvent):
        """Log a guardrail trigger event."""

        # Structured log for analysis
        log_entry = {
            "event_type": "guardrail_trigger",
            "level": "WARNING" if event.action_taken == "block" else "INFO",
            **asdict(event)
        }

        # Emit to logging system
        self.logger.info(json.dumps(log_entry))

        # Emit metrics
        self.metrics.increment(
            "guardrail.triggers",
            tags={
                "guardrail": event.guardrail_name,
                "action": event.action_taken,
                "type": event.guardrail_type
            }
        )

        # Check for alerting conditions
        self._check_alerts(event)
```

### 4.2 Metrics to Track

#### Core Guardrail Metrics

| Metric | Type | Description | Alerting Threshold |
|--------|------|-------------|-------------------|
| `guardrail.triggers.total` | Counter | Total guardrail activations | Rate > 100/min |
| `guardrail.blocks.total` | Counter | Blocked requests | Rate > 50/min |
| `guardrail.block_rate` | Gauge | % of requests blocked | > 10% |
| `guardrail.latency_ms` | Histogram | Time added by guardrails | p99 > 500ms |
| `guardrail.false_positive_rate` | Gauge | Legitimate requests blocked | > 5% |

#### Reliability Metrics

| Metric | Type | Description | Alerting Threshold |
|--------|------|-------------|-------------------|
| `circuit_breaker.state` | Gauge | Circuit state (0=closed, 1=open) | Any open |
| `circuit_breaker.trips` | Counter | Circuit breaker trips | > 3/hour |
| `retry.attempts.total` | Counter | Total retry attempts | Rate > 100/min |
| `retry.exhausted.total` | Counter | Retries that gave up | Any |
| `fallback.used.total` | Counter | Fallback provider used | Rate > 10/min |
| `timeout.total` | Counter | Request timeouts | Rate > 5/min |

#### Cost Metrics

| Metric | Type | Description | Alerting Threshold |
|--------|------|-------------|-------------------|
| `budget.remaining_usd` | Gauge | Remaining budget | < 20% of allocation |
| `budget.utilization_pct` | Gauge | % of budget used | > 80% |
| `cost.per_request_usd` | Histogram | Cost per request | p99 > $1 |
| `rate_limit.hits.total` | Counter | Rate limit hits | Rate > 50/min |

#### Compliance Metrics

| Metric | Type | Description | Alerting Threshold |
|--------|------|-------------|-------------------|
| `pii.detected.total` | Counter | PII detections | Any unexpected |
| `compliance.violations` | Counter | Compliance rule violations | Any |
| `audit.log_failures` | Counter | Failed audit log writes | Any |
| `hitl.pending` | Gauge | Pending human reviews | > 100 |
| `hitl.timeout` | Counter | Human review timeouts | > 5% |

```python
class GuardrailMetrics:
    """Metrics collection for guardrails."""

    def __init__(self, metrics_backend: MetricsBackend):
        self.metrics = metrics_backend

    def record_guardrail_check(
        self,
        guardrail: str,
        action: str,
        latency_ms: float,
        **tags
    ):
        """Record a guardrail check."""

        self.metrics.increment(
            "guardrail.checks.total",
            tags={"guardrail": guardrail, **tags}
        )

        if action == "block":
            self.metrics.increment(
                "guardrail.blocks.total",
                tags={"guardrail": guardrail, **tags}
            )

        self.metrics.histogram(
            "guardrail.latency_ms",
            latency_ms,
            tags={"guardrail": guardrail}
        )

    def record_circuit_breaker_state(self, provider: str, state: CircuitState):
        """Record circuit breaker state change."""
        self.metrics.gauge(
            "circuit_breaker.state",
            1 if state == CircuitState.OPEN else 0,
            tags={"provider": provider}
        )

        if state == CircuitState.OPEN:
            self.metrics.increment(
                "circuit_breaker.trips",
                tags={"provider": provider}
            )

    def record_budget_status(
        self,
        entity: str,
        entity_type: str,
        remaining_usd: float,
        total_usd: float
    ):
        """Record budget status."""
        utilization = 1 - (remaining_usd / total_usd) if total_usd > 0 else 1

        self.metrics.gauge(
            "budget.remaining_usd",
            remaining_usd,
            tags={"entity": entity, "type": entity_type}
        )

        self.metrics.gauge(
            "budget.utilization_pct",
            utilization * 100,
            tags={"entity": entity, "type": entity_type}
        )
```

### 4.3 Alerting Patterns

```python
class AlertConfig:
    """Configuration for guardrail alerts."""

    # Alert channels
    channels: List[str] = ["slack", "pagerduty", "email"]

    # Alert rules
    rules: List[AlertRule] = [
        AlertRule(
            name="high_block_rate",
            metric="guardrail.block_rate",
            condition="value > 10",
            window="5m",
            severity="warning"
        ),
        AlertRule(
            name="circuit_breaker_open",
            metric="circuit_breaker.state",
            condition="value == 1",
            window="1m",
            severity="critical"
        ),
        AlertRule(
            name="budget_exhausted",
            metric="budget.utilization_pct",
            condition="value > 95",
            window="1m",
            severity="critical"
        ),
        AlertRule(
            name="pii_leakage",
            metric="pii.detected.total",
            condition="rate > 0",
            window="1m",
            severity="critical"
        ),
        AlertRule(
            name="compliance_violation",
            metric="compliance.violations",
            condition="rate > 0",
            window="1m",
            severity="critical"
        ),
    ]


class AlertManager:
    """Manage alerts for guardrail events."""

    def check_and_alert(self, event: GuardrailEvent):
        """Check if event should trigger an alert."""

        # Immediate alerts for critical events
        if event.action_taken == "block" and event.guardrail_type == "compliance":
            self.send_alert(
                severity="critical",
                title=f"Compliance Guardrail Triggered: {event.guardrail_name}",
                details=event,
                channels=["pagerduty", "slack"]
            )

        # Aggregate alerts for patterns
        self.alert_aggregator.add_event(event)

        if self.alert_aggregator.should_alert("high_block_rate"):
            self.send_alert(
                severity="warning",
                title="High Guardrail Block Rate Detected",
                details=self.alert_aggregator.get_summary("high_block_rate"),
                channels=["slack"]
            )
```

---

## 5. Enterprise Library Differentiation

### What Makes an Enterprise-Grade LLM Library

| Capability | Basic Library | Enterprise-Grade |
|------------|---------------|------------------|
| **Provider Support** | Single provider | Multi-provider with intelligent routing |
| **Rate Limiting** | Simple RPM | Token-aware, hierarchical budgets |
| **Retry Logic** | Basic exponential | Circuit breakers, fallback chains |
| **Observability** | Basic logging | OpenTelemetry, structured events, alerts |
| **Compliance** | None | HIPAA/GDPR guardrails, audit logging |
| **Cost Control** | None | Hierarchical budgets, cost estimation |
| **Security** | API keys | PII detection, data residency, HITL |
| **Configuration** | Hardcoded | Hierarchical, runtime updateable |

### Key Differentiators

1. **Production Reliability**
   - Circuit breakers with health tracking
   - Intelligent fallback chains
   - Graceful degradation patterns
   - Comprehensive timeout handling

2. **Cost Control**
   - Hierarchical budget enforcement
   - Real-time cost estimation
   - Token-aware rate limiting
   - Usage attribution and chargebacks

3. **Compliance Ready**
   - Built-in HIPAA/GDPR guardrails
   - Comprehensive audit logging
   - Data residency controls
   - Human-in-the-loop patterns

4. **Enterprise Observability**
   - OpenTelemetry integration
   - Structured guardrail events
   - Comprehensive metrics
   - Alerting integration

5. **Operational Excellence**
   - Runtime configuration updates
   - Feature flags for guardrails
   - A/B testing for prompts
   - Canary deployments

---

## References

### Compliance and Governance
- [Security & Compliance Checklist: SOC 2, HIPAA, GDPR for LLM Gateways](https://www.requesty.ai/blog/security-compliance-checklist-soc-2-hipaa-gdpr-for-llm-gateways-1751655071)
- [Audit Logs for LLM Pipelines: Key Practices](https://www.newline.co/@zaoyang/audit-logs-for-llm-pipelines-key-practices--a08f2c2d)
- [HIPAA Compliance AI: Guide to Using LLMs Safely in Healthcare](https://www.techmagic.co/blog/hipaa-compliant-llms)
- [What Are LLM Regulatory Compliance Requirements for Enterprises?](https://datavid.com/blog/what-are-llm-regulatory-compliance-requirements-for-enterprises)

### Human-in-the-Loop
- [Human-in-the-Loop for AI Agents: Best Practices](https://www.permit.io/blog/human-in-the-loop-for-ai-agents-best-practices-frameworks-use-cases-and-demo)
- [Human-in-the-Loop Review Workflows for LLM Applications](https://www.comet.com/site/blog/human-in-the-loop/)
- [Building Generative AI Prompt Chaining with Human in the Loop](https://aws.amazon.com/blogs/machine-learning/building-generative-ai-prompt-chaining-workflows-with-human-in-the-loop/)

### Cost and Rate Controls
- [Rate Limiting in AI Gateway: The Ultimate Guide](https://www.truefoundry.com/blog/rate-limiting-in-llm-gateway)
- [How to Handle Token Limits and Rate Limits in Large-Scale LLM Inference](https://www.typedef.ai/resources/handle-token-limits-rate-limits-large-scale-llm-inference)
- [Budgets, Rate Limits - liteLLM](https://docs.litellm.ai/docs/proxy/users)
- [How to Implement Budget Limits and Alerts in LLM Applications](https://portkey.ai/blog/budget-limits-and-alerts-in-llm-apps/)
- [LLM Gateway Patterns: Rate Limiting and Load Balancing Guide](https://collabnix.com/llm-gateway-patterns-rate-limiting-and-load-balancing-guide/)

### Reliability Patterns
- [Retries, Fallbacks, and Circuit Breakers in LLM Apps](https://portkey.ai/blog/retries-fallbacks-and-circuit-breakers-in-llm-apps/)
- [How to Implement Retry Logic for LLM API Failures in 2025](https://markaicode.com/llm-api-retry-logic-implementation/)
- [Backoff and Retry Strategies for LLM Failures](https://palospublishing.com/backoff-and-retry-strategies-for-llm-failures/)
- [How to Implement Graceful Degradation in LLM Frameworks](https://markaicode.com/implement-graceful-degradation-llm-frameworks/)
- [API Timeout Handling: Best Practices for LLM Applications](https://markaicode.com/api-timeout-handling-llm-applications/)

### Observability
- [Guardrails + OTEL: Monitor LLM Application Performance](https://www.guardrailsai.com/blog/opentelemetry-llm-performance)
- [LLM Guardrails: Best Practices for Deploying LLM Apps Securely](https://www.datadoghq.com/blog/llm-guardrails-best-practices/)
- [AI Guardrails Metrics to Strengthen LLM Monitoring](https://www.fiddler.ai/articles/ai-guardrails-metrics)
- [LLM Observability: Fundamentals, Practices, and Tools](https://neptune.ai/blog/llm-observability)
- [Observability Best Practices for LLM Apps](https://skywork.ai/blog/llm-observability-best-practices-haiku-logging-tracing-guardrails/)

### Data Residency
- [Why LLM Research in Europe Is Moving to Sovereign Cloud Infrastructure](https://www.nexgencloud.com/blog/thought-leadership/why-llm-research-in-europe-is-moving-to-sovereign-cloud-infrastructure)
- [Local LLMs for Data Sovereignty](https://www.e-spincorp.com/local-llms-data-sovereignty/)
- [Overview of AI and LLM configurations in Microsoft for Sovereignty](https://learn.microsoft.com/en-us/industry/sovereignty/architecture/aiwithllm/overview-ai-llm-configuration)
- [Public vs Private LLMs: Secure AI for Enterprises](https://www.matillion.com/blog/public-vs-private-llms-enterprise-ai-security)

### Enterprise LLM Architecture
- [LLM in Enterprise: A Complete Guide](https://www.truefoundry.com/blog/enterprise-in-llm)
- [Deploying Enterprise LLM Applications with Inference, Guardrails, and Observability](https://www.fiddler.ai/blog/deploying-enterprise-llm-applications-with-inference-guardrails-and-observability)
- [Mastering LLM Guardrails: Complete 2025 Guide](https://orq.ai/blog/llm-guardrails)
- [Building Guardrails for Enterprise AI Applications with LLMs](https://www.infoq.com/presentations/guardrails-ai/)
