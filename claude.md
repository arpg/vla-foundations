# VLA Foundations Development Guide for AI SWE Agents (Public Repo)

This is the **public student-facing repository** for VLA Foundations. This is a Next.js (App Router) application used as a living textbook and course platform for Vision-Language-Action (VLA) robotics. It uses **Tailwind CSS** for styling, **MDX** for content (textbook/assignments), and **pnpm** for package management. It is deployed to **GitHub Pages** via GitHub Actions.

Read more about the course workflow in [README.md](README.md).

---

## Repository Purpose

This is a **student repository** containing:
- Assignment starter code with TODOs (`src/assignments/`)
- Public validation tests (`tests/public/`)
- Textbook content and assignment specs (`content/`)
- Course website (Next.js application)

**Note**: Complete solutions and internal grading tests are maintained in a separate private instructor repository.

---

## Initial Setup

### Prerequisites
```bash
# Install Node.js 18+ and pnpm
npm install -g pnpm

# Install Python 3.11+ for assignments
python3 --version
```

### Installation
```bash
# Install dependencies
pnpm install

# Run development server
pnpm dev
# Access at http://localhost:3000

# Build production (static export in out/)
pnpm build

# Lint
pnpm lint
```

---

## Student Workflow

### 1. Setup
Follow the Scratch-0 assignment to configure your environment.

### 2. Branching
All work must be done on a branch named `<assignment>-<username>`.

**Example**: `scratch-1-johndoe`

```bash
git checkout -b scratch-1-johndoe
```

### 3. Implementation
- Code stubs are in `src/assignments/`
- Documentation and reports belong in `content/course/submissions/`

### 4. Testing
Run public tests to verify your implementation:

```bash
# Test specific assignment
pytest tests/public/test_scratch1_basic.py -v

# Run all public tests
pytest tests/public/ -v
```

**Expected output**:
```
tests/public/test_scratch1_basic.py::TestModelStructure::test_model_initialization PASSED
tests/public/test_scratch1_basic.py::TestModelStructure::test_forward_pass_shape PASSED
tests/public/test_scratch1_basic.py::TestModelStructure::test_no_nans PASSED
tests/public/test_scratch1_basic.py::TestModelStructure::test_gradient_flow PASSED
```

### 5. Submission
Open a Pull Request to the `staging` branch. **Do not target `main`**.

1. Go to https://github.com/arpg/vla-foundations
2. Click "Pull requests" → "New pull request"
3. **Base branch**: `staging` (NOT `main`)
4. **Compare branch**: your branch name
5. Title: `Assignment X: Your Name`
6. Add a description of your work

### 6. Review Process
1. Wait for CI checks to pass (GitHub Actions will validate your submission)
2. Wait for instructor review
3. Address any requested changes
4. **ONLY the instructor can merge pull requests**
5. Once approved, the instructor will merge to `staging`, then to `main`

**You do NOT have permission to merge your own PRs. All merges are done by the instructor.**

---

## Commands Useful in Development

### Python Testing
```bash
# Install test dependencies
pip install pytest torch

# Run public tests
pytest tests/public/ -v

# Run specific test
pytest tests/public/test_scratch1_basic.py::TestModelStructure::test_forward_pass_shape -v

# Run with detailed output
pytest tests/public/ -v --tb=short
```

### Assignment Development
```bash
# Navigate to assignment directory
cd src/assignments/scratch-1

# Run your implementation
python3 model.py

# Test your implementation
pytest ../../tests/public/test_scratch1_basic.py -v
```

### Git Operations
```bash
# Create feature branch
git checkout -b scratch-1-username

# Stage changes
git add src/assignments/scratch-1/

# Commit
git commit -m "feat: implement scratch-1 model"

# Push to your branch
git push origin scratch-1-username

# Update with latest staging changes
git fetch origin
git rebase origin/staging
git push --force-with-lease
```

---

## Linting and Formatting

### Semantic Line Breaks
**All MDX files MUST use one sentence per line.** This is mandatory to allow granular, line-by-line feedback in Pull Requests.

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

### LaTeX
Use formal LaTeX for all mathematical expressions:

```markdown
The action distribution is:
$$
p(a_t | s_t, I_t) = \text{softmax}(\text{MLP}(h_t))
$$
```

Do not use code blocks for math.

### Next.js Linting
```bash
pnpm lint
```

---

## Testing Philosophy

### Public Tests (`tests/public/`)
These are the tests **you can see and run** to validate your implementation.

**What they test**:
- Basic model structure (initialization, parameter counts)
- Forward pass correctness (no NaNs, correct output shapes)
- Gradient flow (backpropagation works)
- Basic functionality (model can be trained)

**You should ensure all public tests pass before submitting your PR.**

### Internal Tests (Private)
The instructor also runs **internal grading tests** that you cannot see. These test:
- Implementation correctness against gold standards
- Edge cases and error handling
- Performance and efficiency
- Advanced requirements

**Your grade depends on passing both public and internal tests.**

---

## Interacting with the App

### Local Development
```bash
pnpm dev
# Access at http://localhost:3000
```

Navigate to:
- Textbook: `http://localhost:3000/textbook`
- Assignments: `http://localhost:3000/textbook/assignments/scratch-1`

### Staging Previews
Every Pull Request triggers a deployment to a unique subdirectory:
```
https://vlm-robotics.dev/staging/pulls/[PR_NUMBER]/
```

You can preview your changes (if you submitted MDX content) at this URL after CI completes.

### Production Site
The live course website is at:
```
https://www.vlm-robotics.dev
```

**Note**: Only instructor-approved content is deployed to production.

---

## Patterns & Standards

### Amazon Principle
Course content follows the "audit" format. We write rigorous, durable technical audits, not summaries.

### Git Hygiene
We are a **rebase-only** repository. Use `git rebase staging` instead of `git merge staging`.

**Correct workflow**:
```bash
# Update your branch with latest staging changes
git fetch origin
git rebase origin/staging

# If conflicts occur, resolve them and continue
git rebase --continue

# Force push your rebased branch
git push --force-with-lease
```

**Never use merge commits.** PRs containing "Merge branch 'staging'" will be closed.

### Assignment File Structure
Each assignment follows this structure:

```
src/assignments/scratch-1/
├── __init__.py          # Package initialization
├── model.py             # Model implementation (with TODOs)
├── train.py             # Training script (with TODOs)
├── generate_data.py     # Data generation utilities
└── README.md            # Assignment-specific instructions
```

**Complete the TODO sections** to implement the assignment.

---

## File Map of Interest

### Configuration
- [next.config.ts](next.config.ts) - Next.js configuration with dynamic routing
- [tailwind.config.ts](tailwind.config.ts) - Tailwind CSS configuration
- [tsconfig.json](tsconfig.json) - TypeScript configuration

### Content
- [content/course/](content/course/) - Course content (textbook chapters, assignments)
- [content/course/assignments/](content/course/assignments/) - Assignment specifications

### Components
- [components/audit/AuditLayout.tsx](components/audit/AuditLayout.tsx) - Primary wrapper for textbook chapters
- [components/ui/](components/ui/) - Reusable UI components

### Testing
- [tests/public/](tests/public/) - Public validation tests
- [pytest.ini](pytest.ini) - pytest configuration

### GitHub Actions
- [.github/workflows/vla-audit.yml](.github/workflows/vla-audit.yml) - PR validation and staging deployment
- [.github/workflows/deploy.yml](.github/workflows/deploy.yml) - Production deployment

---

## Assignment Workflow Example

### Scratch-1: Decoder-Only Transformer

1. **Read the assignment spec**:
   - Navigate to: https://www.vlm-robotics.dev/textbook/assignments/scratch-1
   - Or locally: http://localhost:3000/textbook/assignments/scratch-1

2. **Create your branch**:
   ```bash
   git checkout -b scratch-1-johndoe
   ```

3. **Implement the TODOs**:
   ```bash
   cd src/assignments/scratch-1
   # Edit model.py, train.py, etc.
   ```

4. **Test your implementation**:
   ```bash
   pytest tests/public/test_scratch1_basic.py -v
   ```

5. **Commit and push**:
   ```bash
   git add src/assignments/scratch-1/
   git commit -m "feat: implement scratch-1 decoder-only transformer"
   git push origin scratch-1-johndoe
   ```

6. **Open PR**:
   - Go to https://github.com/arpg/vla-foundations
   - Create PR from `scratch-1-johndoe` to `staging`
   - Title: "Scratch-1: John Doe"
   - Description: Brief summary of your implementation

7. **Wait for review**:
   - CI will run public tests
   - Instructor will run internal grading tests
   - Instructor will provide feedback

8. **Address feedback** (if needed):
   ```bash
   # Make changes
   git add .
   git commit -m "fix: address review comments"
   git push origin scratch-1-johndoe
   ```

---

## Common Issues

### Issue: Tests Fail Locally
**Solution**: Ensure you installed dependencies:
```bash
pip install pytest torch numpy
```

### Issue: Import Errors
**Solution**: Make sure you're running tests from the repository root:
```bash
cd /path/to/vla-foundations
pytest tests/public/test_scratch1_basic.py -v
```

### Issue: Merge Conflicts
**Solution**: Rebase your branch:
```bash
git fetch origin
git rebase origin/staging
# Resolve conflicts in your editor
git add .
git rebase --continue
git push --force-with-lease
```

### Issue: PR Checks Failing
**Solution**: Check the GitHub Actions logs:
1. Go to your PR
2. Click "Details" next to the failing check
3. Read the error messages
4. Fix the issues and push again

### Issue: Can't Merge PR
**Expected**: Students cannot merge their own PRs. Wait for instructor approval and merge.

---

## Resources

### Documentation
- [MDX Syntax](https://mdxjs.com/)
- [Next.js App Router](https://nextjs.org/docs/app)
- [Tailwind CSS](https://tailwindcss.com/docs)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [LaTeX Math Symbols](https://www.overleaf.com/learn/latex/List_of_Greek_letters_and_math_symbols)

### Papers & Datasets
- [RT-1 Paper](https://arxiv.org/abs/2212.06817)
- [RT-2 Paper](https://arxiv.org/abs/2307.15818)
- [Open-X Embodiment](https://robotics-transformer-x.github.io/)
- [DROID Dataset](https://droid-dataset.github.io/)

### Course Resources
- **Course Website**: https://www.vlm-robotics.dev
- **Repository**: https://github.com/arpg/vla-foundations
- **Instructor**: Christoffer Heckman (christoffer.heckman@colorado.edu)

---

## Engineering Standards

### Code Quality
- Follow PEP 8 style guide for Python
- Use type hints where appropriate
- Add docstrings to classes and functions
- Keep functions focused and single-purpose

### Testing
- All public tests must pass before submission
- Write clean, readable code
- Handle edge cases appropriately

### Documentation
- Update README.md if adding new features
- Document complex algorithms with comments
- Use semantic line breaks in MDX files

---

## Grading Rubric (Typical)

Assignments are typically graded on:

| Component | Points | Description |
|-----------|--------|-------------|
| Correctness | 50% | Implementation meets requirements, passes tests |
| Code Quality | 20% | Clean, readable, well-documented code |
| Performance | 15% | Efficient implementation, appropriate algorithms |
| Documentation | 15% | Clear explanations, proper formatting |

**Note**: Exact rubric varies by assignment. See individual assignment specs.

---

## Getting Help

1. **Read the assignment spec carefully** - Many questions are answered there
2. **Check the course forum** - Others may have asked similar questions
3. **Attend office hours** - Best place for detailed help
4. **Ask in Discord/Slack** - Quick questions and clarifications
5. **Email instructor** - For private concerns

**Do NOT share solution code publicly.** Help each other understand concepts, but do your own implementation.

---

## License

Copyright © 2026 Christoffer Heckman. All rights reserved.

Course materials are for educational use by enrolled students only.

---

**Good luck with your assignments!** Remember to start early, test often, and ask for help when needed.
