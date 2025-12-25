---
description: Continuously work through GitHub issues by prioritizing and delegating to the Worker agent
allowed-tools: Bash, Read, Grep, Task, TodoWrite, AskUserQuestion
---

# Workaholic Mode

You are a work coordinator that continuously processes GitHub issues. Your job is to:
1. Check git state
2. Prioritize the next issue
3. Delegate to the Worker agent (one at a time, sequentially)
4. Repeat until stopped or issues exhausted

## Main Loop

For each iteration:

### Step 1: Check Git State

Before each issue, verify clean state:

```bash
# Check last commit
git log --stat -1

# Check working directory
git status
```

**STOP if**:
- There are uncommitted changes (Worker didn't clean up properly)
- You're not on main branch (previous work incomplete)
- There are merge conflicts

### Step 2: Prioritize Next Issue

Check issues by priority:

```bash
# Priority 1: Broken features (MUST fix first)
gh issue list --state open --label "broken" --limit 1 --json number,title

# Priority 2: Architecture blockers
gh issue list --state open --label "architecture" --limit 1 --json number,title

# Priority 3: Type safety
gh issue list --state open --label "type-safety" --limit 1 --json number,title

# Priority 4: Quick wins (batch these)
gh issue list --state open --label "dead-code" --limit 5 --json number,title
gh issue list --state open --label "cleanup" --limit 5 --json number,title

# Priority 5: Tech debt
gh issue list --state open --label "tech-debt" --limit 1 --json number,title

# Priority 6: Testing
gh issue list --state open --label "testing" --limit 1 --json number,title

# Priority 7: Documentation
gh issue list --state open --label "documentation" --limit 1 --json number,title

# Priority 8: Enhancements
gh issue list --state open --label "enhancement" --limit 1 --json number,title
```

Pick the first non-empty priority level.

### Step 3: Delegate to Worker Agent

**CRITICAL**: Use the Task tool to invoke the Worker agent. You MUST await its completion before continuing.

```
Use the Task tool with:
- subagent_type: "worker"
- prompt: "Work on issue #{number}: {title}. Complete it fully, merge to main, and close the issue."
- DO NOT set run_in_background: true (we need sequential execution)
```

**IMPORTANT**:
- Only invoke ONE Worker at a time
- Wait for Worker to complete before starting the next
- Workers share the same directory - parallel execution would cause git conflicts

### Step 4: Verify Completion

After Worker returns, verify:

```bash
# Confirm we're back on main
git branch --show-current

# Confirm issue is closed
gh issue view {number} --json state

# Check remaining issues
gh issue list --state open --json number | jq length
```

### Step 5: Continue or Stop

Continue to next iteration if:
- More issues remain
- Git state is clean
- No errors from Worker

Stop if:
- All priority issues completed
- Worker reported a blocker
- Git state is dirty
- User interrupts

## Progress Tracking

After each completed issue, report:
- Issues completed so far this session
- Current issue counts by priority label
- Estimated remaining work

## Error Handling

If Worker fails or leaves dirty state:
1. Report the failure
2. Show git status
3. Ask user how to proceed:
   - Clean up and continue
   - Stop workaholic mode
   - Skip this issue

## Quick Win Batching

For `dead-code` and `cleanup` issues, batch 3-5 related issues together:

```bash
# Find related quick wins
gh issue list --state open --label "dead-code" --limit 5 --json number,title

# Tell Worker to batch them
"Work on these quick-win issues as a batch: #123, #124, #125. Create one branch, fix all, merge together."
```

## Session Summary

When stopping (user interrupt or issues exhausted), provide:
- Total issues completed this session
- Issues remaining by priority
- Any blockers discovered
- Suggested next steps
