---
description: Pick up the next GitHub issue and work on it, maintaining state via git
allowed-tools: Bash, Read, Write, Edit, Glob, Grep, Task, TodoWrite, AskUserQuestion
---

# Work Session

You are starting a work session. Your goal is to pick up the next available issue, work on it to completion, and leave clear state for the next session.

## Step 1: Gather Context

First, gather the current project state by running these commands:

1. **Git status**: Check current branch and any uncommitted changes
   ```
   git status
   git branch --show-current
   ```

2. **Recent git log**: See what was done recently
   ```
   git log --oneline -10
   ```

3. **GitHub issues**: List open issues sorted by priority/number
   ```
   gh issue list --state open
   ```

4. **Read CHANGELOG.md**: Check for any unreleased changes (create if missing)

## Step 2: Select an Issue

Based on the gathered context:

1. If you're on a feature branch with uncommitted work, continue that work
2. If you're on a feature branch with committed work, check if the issue is complete
3. If you're on main, select the lowest-numbered open issue (they're roughly prioritized)

When selecting an issue:
- Read the issue details: `gh issue view {number}`
- Check for any comments with progress updates
- Understand the acceptance criteria

## Step 3: Create Feature Branch (if needed)

If on main and starting new work:
```
git checkout -b issue-{number}-{short-description}
```

## Step 4: Do the Work

1. Use TodoWrite to break down the issue into sub-tasks
2. Work through each task methodically
3. Run tests frequently:
   ```
   uv run pytest tests/ -x -q
   ```
4. For type checking (if mypy is configured):
   ```
   uv run mypy src/
   ```
5. Test imports work correctly:
   ```
   uv run python -c "from magically import spell; print('OK')"
   ```

## Step 5: Commit Progress

After completing meaningful chunks of work:
1. Stage and commit with a clear message referencing the issue
   ```
   git add -A
   git commit -m "Progress on #{number}: {description}"
   ```

2. If you discover related issues or blockers, create new GitHub issues:
   ```
   gh issue create --title "..." --body "..." --label "enhancement"
   ```
   Then continue with your current work.

## Step 6: Complete the Issue

When all acceptance criteria are met:

1. Ensure all tests pass:
   ```
   uv run pytest tests/ -q
   ```

2. Update CHANGELOG.md under `[Unreleased]` section

3. Make a final commit:
   ```
   git add -A
   git commit -m "Complete #{number}: {title}"
   ```

4. Merge to main:
   ```
   git checkout main
   git merge issue-{number}-{short-description}
   git branch -d issue-{number}-{short-description}
   ```

5. Close the issue:
   ```
   gh issue close {number} --comment "Completed in $(git rev-parse --short HEAD)"
   ```

6. Push to remote:
   ```
   git push origin main
   ```

## Step 7: Report Status

At the end of your session, summarize:
- What issue you worked on
- What was accomplished
- Any new issues created
- Current state for the next session

## Important Rules

- **One issue at a time**: Focus on completing the selected issue
- **Leave clean state**: Always commit or stash before ending
- **Create issues for tangents**: Don't go down rabbit holes; create issues and move on
- **Update CHANGELOG**: Every user-facing change should be documented
- **Keep commits atomic**: One logical change per commit
- **Run tests before committing**: Don't break the build
- **Prefer editing over creating**: Modify existing files when possible

## Project-Specific Notes

- **Testing**: Use `uv run pytest` (dev dependencies via uv)
- **Smoke tests**: Tests marked `@pytest.mark.smoke` hit real APIs, skip in normal runs
- **Package structure**: Source in `src/magically/`, tests in `tests/`
- **Core files**: `spell.py` (decorator), `config.py` (model aliases), `logging.py` (observability)
