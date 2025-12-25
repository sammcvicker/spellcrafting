---
description: Pick up the next GitHub issue and work on it, maintaining state via git (project)
allowed-tools: Bash, Read, Write, Edit, Glob, Grep, Task, TodoWrite, AskUserQuestion
---

# Work Session

You are starting a work session. Your goal is to pick up the next available issue, work on it to completion, and leave clear state for the next session.

## Issue Volume Context

This project has ~180 open issues from a comprehensive codebase audit. Issues are categorized by label and should be tackled strategically.

### Priority Order (work on these first)

1. **`broken`** (10 issues) - Features that don't work as documented
2. **`architecture`** (9 issues) - Structural problems blocking other work
3. **`type-safety`** (19 issues) - Type system improvements
4. **`tech-debt`** (36 issues) - Code quality improvements
5. **`testing`** (37 issues) - Test coverage gaps
6. **`documentation`** (32 issues) - Docs that need updating
7. **`enhancement`** (45 issues) - New features/improvements

### Quick Wins vs Deep Work

**Quick wins** (< 30 min each, batch these):
- `dead-code` - Remove unused imports/code
- `cleanup` - Simple code cleanup
- `chore` - Maintenance tasks
- `docs-stale` - Update outdated docs

**Deep work** (1+ hours, focus on one):
- `architecture` - Major refactors
- `refactor` - Significant code changes
- `complexity` - Reducing cyclomatic complexity

## Step 1: Gather Context

First, gather the current project state:

```bash
# Check current state
git status && git branch --show-current

# Recent commits
git log --oneline -5

# Count issues by priority label
echo "=== Issue counts by priority ===" && \
gh issue list --state open --label "broken" --json number | jq length && \
gh issue list --state open --label "architecture" --json number | jq length && \
gh issue list --state open --label "type-safety" --json number | jq length
```

## Step 2: Select an Issue

### If continuing previous work:
```bash
# Check current branch
git branch --show-current

# If on feature branch, continue that issue
```

### If starting fresh, pick by priority:

```bash
# Priority 1: Broken features (fix these first)
gh issue list --state open --label "broken" --limit 5

# Priority 2: Architecture issues (unblock other work)
gh issue list --state open --label "architecture" --limit 5

# Priority 3: Quick wins (batch several together)
gh issue list --state open --label "dead-code" --limit 10
gh issue list --state open --label "cleanup" --limit 10

# Or find related issues to batch:
gh issue list --state open --search "spell.py in:title" --limit 10
gh issue list --state open --search "guard in:title" --limit 10
```

### Selection Strategy:

1. **Broken first**: Fix `broken` label issues before other work
2. **Batch related**: If working on spell.py, grab multiple spell.py issues
3. **Quick win sessions**: Knock out 3-5 `cleanup`/`dead-code` issues together
4. **One deep issue**: For `architecture`/`refactor`, focus on one at a time

When selecting, read the full issue:
```bash
gh issue view {number}
```

## Step 3: Create Feature Branch

```bash
# For single issue
git checkout -b issue-{number}-{short-description}

# For batched quick wins (use first issue number)
git checkout -b cleanup-batch-{first-issue-number}
```

## Step 4: Do the Work

1. **Use TodoWrite** to track sub-tasks
2. **Run tests frequently**:
   ```bash
   uv run pytest tests/ -x -q
   ```
3. **Check types** (when touching type annotations):
   ```bash
   uv run python -c "from magically import spell, guard, Config; print('imports OK')"
   ```

### For Batched Quick Wins:

Work through each issue in sequence:
1. Make the change
2. Commit with issue reference: `git commit -m "Fix #123: Remove unused import"`
3. Move to next issue
4. Final commit closes all: `Closes #123, closes #124, closes #125`

## Step 5: Commit Progress

```bash
# Single issue progress
git add -A && git commit -m "Progress on #123: description"

# Batch commits (reference each issue)
git add -A && git commit -m "Fix #123: Remove unused functools import"
git add -A && git commit -m "Fix #124: Remove unused time import"
```

## Step 6: Complete the Issue(s)

1. **Ensure tests pass**:
   ```bash
   uv run pytest tests/ -q
   ```

2. **Update CHANGELOG.md** (for user-facing changes only)

3. **Final commit**:
   ```bash
   # Single issue
   git add -A && git commit -m "Complete #123: title"

   # Batch
   git add -A && git commit -m "Complete cleanup batch: closes #123, closes #124, closes #125"
   ```

4. **Merge and cleanup**:
   ```bash
   git checkout main
   git merge --no-ff issue-{number}-{description} -m "Merge: Complete #123"
   git branch -d issue-{number}-{description}
   ```

5. **Close issues**:
   ```bash
   # Single issue
   gh issue close {number} --comment "Completed in $(git rev-parse --short HEAD)"

   # Batch (GitHub auto-closes from commit message "closes #X")
   git push origin main
   ```

## Step 7: Report Status

Summarize:
- Issues completed this session
- Any blockers discovered
- Suggested next issues to tackle
- Remaining count: `gh issue list --state open --json number | jq length`

## Issue Batching Guide

### Good batches (related changes):

```bash
# All spell.py issues
gh issue list --state open --search "spell.py" --limit 20

# All guard-related
gh issue list --state open --search "guard" --limit 20

# All type-safety in one module
gh issue list --state open --label "type-safety" --search "config" --limit 10

# All dead-code (usually independent)
gh issue list --state open --label "dead-code"

# All docs-stale (independent, fast)
gh issue list --state open --label "docs-stale"
```

### Don't batch:
- `architecture` issues (too complex)
- `broken` issues (need careful attention)
- Issues in different modules

## Labels Quick Reference

| Label | Count | Action |
|-------|-------|--------|
| `broken` | ~10 | Fix immediately, one at a time |
| `architecture` | ~9 | Deep work, one at a time |
| `type-safety` | ~19 | Can batch by module |
| `tech-debt` | ~36 | Can batch related issues |
| `testing` | ~37 | Batch by test file |
| `documentation` | ~32 | Batch docs-stale together |
| `dead-code` | ~5 | Quick batch, all at once |
| `cleanup` | ~20 | Batch by file |
| `refactor` | ~26 | One at a time |
| `inconsistency` | ~13 | Can batch related |

## Project-Specific Notes

- **Testing**: `uv run pytest tests/ -x -q` (stop on first failure)
- **Smoke tests**: Skip with `uv run pytest tests/ -m "not smoke"`
- **Package structure**: `src/magically/`, `tests/`
- **Core files to know**:
  - `spell.py` - Main decorator (900+ lines, needs refactor)
  - `guard.py` - Guard decorators
  - `config.py` - Model alias configuration
  - `logging.py` - Observability/tracing
  - `on_fail.py` - Failure strategies
  - `validator.py` - LLM-powered validation

## Important Rules

- **Prioritize `broken` label**: These are user-facing bugs
- **Batch wisely**: Group 3-5 related quick wins
- **One deep issue at a time**: Don't mix architecture work
- **Leave clean state**: Always commit before ending
- **Run tests before merge**: `uv run pytest tests/ -q`
- **Skip CHANGELOG for internal changes**: Only document user-facing changes
