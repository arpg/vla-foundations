---
name: new-assignment
description: "Scaffold new assignment structure"
---

# New Assignment Command

Create complete assignment structure with templates.

## Usage

```
/new-assignment
```

## Execution

Use the Skill tool to invoke the new-assignment skill:

```
Skill: new-assignment
```

This will:
1. Prompt for assignment name
2. Gather assignment metadata (type, focus, difficulty)
3. Create directory structure
4. Generate starter code templates (with TODOs)
5. Generate solution templates
6. Generate public test suite
7. Generate internal test suite
8. Generate assignment spec (MDX)
9. Create README files
10. Display scaffolding summary

After completion, you should:
- Complete solution implementations
- Generate gold standard fixtures with `/generate-fixtures`
- Update assignment spec with details
- Test with `/test-rigor`
- Release with `/release`
