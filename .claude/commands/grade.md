---
name: grade
description: "Grade student PR submission"
---

# Grade Command

Quick access to automated grading workflow.

## Usage

```
/grade
```

## Execution

Use the Skill tool to invoke the grade skill:

```
Skill: grade
```

This will:
1. Prompt for PR number
2. Fetch student code from GitHub
3. Run public tests
4. Run internal grading tests
5. Generate detailed feedback report
6. Post comment on PR (optional)
7. Update PR labels
8. Display grading summary
