# VLA Foundations: Course Repository

Vision-Language-Action Foundations for Robotics - CSCI 7000, Spring 2026

**Live Site**: https://www.vlm-robotics.dev

This repository contains the framework for CSCI 7000: VLA Foundations for Robotics.

---

## Workflow for Students

### Setup
Follow the Scratch-0 assignment to configure your environment.

### Branching
All work must be done on a branch named `[assignment]-[username]`.

**Example**: `scratch-1-heckman`

```bash
git checkout -b scratch-1-johndoe
```

### Implementation
- Code stubs are in `src/assignments/`
- Documentation and reports belong in `content/course/submissions/`

### Submission
Open a Pull Request to the `staging` branch. **Do not target `main`**.

1. Go to https://github.com/arpg/vla-foundations
2. Click "Pull requests" → "New pull request"
3. **Base branch**: `staging` (NOT `main`)
4. **Compare branch**: your branch name
5. Title: `Assignment X: Your Name`
6. Add a description of your work

### Review Process
1. Wait for CI checks to pass (GitHub Actions will validate your submission)
2. Wait for instructor review
3. Address any requested changes
4. **ONLY the instructor can merge pull requests**
5. Once approved, the instructor will merge to `staging`, then to `main`

**You do NOT have permission to merge your own PRs. All merges are done by the instructor.**

---

## Engineering Standards

### Semantic Line Breaks
Use semantic line breaks (one sentence per line) in all MDX files.

**Bad:**
```markdown
This is a very long sentence with multiple ideas. It continues on the same line. This makes PR review difficult.
```

**Good:**
```markdown
This is a sentence on its own line.
Each idea gets its own line.
This makes PR review much easier.
```

### Linear Git History
Maintain a linear git history.
Use `git rebase staging` instead of `git merge staging`.

**Example:**
```bash
# Update your branch with latest staging changes
git fetch origin
git rebase origin/staging

# If conflicts occur, resolve them and continue
git rebase --continue

# Force push your rebased branch
git push --force-with-lease
```

---

## Resources

### Documentation
- [MDX Syntax](https://mdxjs.com/)
- [Next.js App Router](https://nextjs.org/docs/app)
- [Tailwind CSS](https://tailwindcss.com/docs)
- [LaTeX Math Symbols](https://www.overleaf.com/learn/latex/List_of_Greek_letters_and_math_symbols)

### Papers & Datasets
- [RT-1 Paper](https://arxiv.org/abs/2212.06817)
- [RT-2 Paper](https://arxiv.org/abs/2307.15818)
- [Open-X Embodiment](https://robotics-transformer-x.github.io/)
- [DROID Dataset](https://droid-dataset.github.io/)

---

## Contact

- **Instructor**: Christoffer Heckman
- **Email**: christoffer.heckman@colorado.edu
- **Course**: CSCI 7000, Spring 2026
- **GitHub**: https://github.com/arpg/vla-foundations

---

## License

Copyright © 2026 Christoffer Heckman. All rights reserved.

Course materials are for educational use by enrolled students only.
