# VLA Foundations

Vision-Language-Action Foundations: A living textbook and course repository.

**Live Site**: https://www.vlm-robotics.dev
**GitHub**: https://github.com/arpg/vla-foundations

## Project Overview

This repository serves as the source of truth for:

1. **The Living Textbook**: An 8-chapter technical reference on VLAs for Robotics
2. **CSCI 7000 Course Portal**: Course logistics, assignments, and student contributions

**Tech Stack**: Next.js 16, TypeScript, Tailwind CSS, MDX, Contentlayer

## Repository Structure

```
vla-foundations/
├── app/                      # Next.js App Router pages
│   ├── page.tsx             # Landing page
│   ├── textbook/[slug]/     # Dynamic chapter pages
│   └── reference/           # Reference implementations
├── components/
│   └── textbook/            # UI components for textbook
│       ├── Sidebar.tsx      # Navigation
│       └── TextbookLayout.tsx
├── content/
│   ├── textbook/            # 8-chapter VLA textbook
│   │   ├── foundations/     # Chapter 0
│   │   ├── architectures/   # Chapter 1
│   │   ├── data/            # Chapter 2
│   │   ├── training/        # Chapter 3
│   │   ├── evaluation/      # Chapter 4
│   │   ├── deployment/      # Chapter 5
│   │   ├── applications/    # Chapter 6
│   │   └── future/          # Chapter 7
│   ├── course/              # Course materials
│   │   ├── Syllabus.mdx
│   │   └── assignments/     # Scratch-0, Paper Audit, Capstone
│   └── contributors/        # Student profiles (PR requirement)
├── lib/
│   └── chapters.ts          # Chapter metadata utilities
├── scripts/
│   └── deploy.sh            # Automated deployment script
├── .github/
│   └── workflows/
│       └── vla-audit.yml    # CI/CD for PR validation
└── README.md
```

## The 8-Chapter Textbook

0. **Foundations** - Core concepts and problem formulation
1. **Architectures** - Model designs and network topologies
2. **Data** - Dataset construction and curation strategies
3. **Training** - Optimization and fine-tuning methods
4. **Evaluation** - Metrics and benchmarking protocols
5. **Deployment** - Production systems and scaling
6. **Applications** - Real-world use cases and case studies
7. **Future Directions** - Open problems and research frontiers

## Development Workflow

### Initial Setup

```bash
# Clone the repository
git clone https://github.com/arpg/vla-foundations.git
cd vla-foundations

# Install dependencies
pnpm install

# Run development server
pnpm dev
```

Navigate to `http://localhost:3000` to see the site.

### Adding Content

#### New Textbook Section

```bash
# Create a new MDX file in the appropriate chapter
touch content/textbook/[chapter-name]/new-section.mdx
```

```mdx
---
title: "Section Title"
chapter: 1
subsection: 5
---

# Your Content Here

Include LaTeX: $f(x) = x^2$

\`\`\`python
# Include code examples
def example():
    pass
\`\`\`
```

#### New Assignment

```bash
touch content/course/assignments/assignment-name.mdx
```

### Building and Deploying

#### Local Build

```bash
# Build the static site
pnpm build

# Preview the production build
pnpm start
```

#### Deploy to Production

```bash
# Automated deployment to vlm-robotics.dev
./scripts/deploy.sh
```

This will:
1. Build the Next.js site locally
2. Sync to the remote server via rsync
3. Deploy to `https://www.vlm-robotics.dev`

**Requirements**: SSH access with the automation key (`~/.ssh/id_ed25519_automation`)

## Course Workflow

### For Students

**IMPORTANT**: All students must follow this workflow for assignment submissions.

1. **Create your own branch**:
   ```bash
   git checkout -b assignment-name-yourname
   ```
   Example: `git checkout -b scratch-1-johndoe`

2. **Make your changes**:
   - Add your code to `src/assignments/`
   - Create your submission report in `content/course/submissions/`
   - Update your contributor profile if needed

3. **Commit your work**:
   ```bash
   git add .
   git commit -m "Complete Assignment X: Your Name"
   git push origin assignment-name-yourname
   ```

4. **Open a Pull Request**:
   - Go to https://github.com/arpg/vla-foundations
   - Click "Pull requests" → "New pull request"
   - **Base branch**: `staging` (NOT `main`)
   - **Compare branch**: your branch name
   - Title: `Assignment X: Your Name`
   - Add a description of your work

5. **Wait for CI checks** to pass (GitHub Actions will validate your submission)

6. **Wait for instructor review**:
   - **ONLY the instructor can merge pull requests**
   - The instructor will review your code and report
   - You may be asked to make changes
   - Once approved, the instructor will merge to `staging`, then to `main`

**You do NOT have permission to merge your own PRs. All merges are done by the instructor.**

### For the Instructor

1. **Review student PRs** on the `staging` branch
2. **Provide feedback** and request changes if needed
3. **Merge to `staging`** when approved
4. **Periodically merge `staging` to `main`**
5. **Deploy to production** using `./scripts/deploy.sh`

## CI/CD Automation

### GitHub Actions Workflow

The `vla-audit.yml` workflow runs on all PRs to `staging` and `main`:

- ✅ Contentlayer build validation
- ✅ MDX syntax checking
- ✅ LaTeX rendering verification
- ✅ Full site build

**Location**: `.github/workflows/vla-audit.yml`

### Branch Protection

- `main`: Requires passing CI checks and instructor approval
- `staging`: Student PR target, requires passing CI

## Branching Strategy

- **`main`**: Production branch (deployed to live site)
- **`staging`**: Student PR target (integration testing)
- **Feature branches**: `feature/assignment-name`, `paper-audit-N-yourname`

## Technologies

### Core
- **Next.js 16**: Static site generation
- **TypeScript**: Type safety
- **Tailwind CSS**: Styling
- **MDX**: Markdown with JSX

### Content Processing
- **Contentlayer**: Content validation
- **gray-matter**: Frontmatter parsing
- **remark-math** + **rehype-katex**: LaTeX rendering
- **remark-gfm**: GitHub-flavored Markdown

### Deployment
- **Apache**: Web server on DigitalOcean
- **rsync**: File synchronization
- **SSH**: Secure remote access

## Development Tools

### Local Commands

```bash
# Development
pnpm dev              # Start dev server (hot reload)
pnpm build            # Build production site
pnpm build-content    # Validate content only
pnpm lint             # Run ESLint

# Deployment
./scripts/deploy.sh   # Deploy to production
```

### Remote Server Access

```bash
# SSH into production server
ssh -i ~/.ssh/id_ed25519_automation crh@ristoffer.ch

# Navigate to project
cd /var/www/vlm-robotics.dev

# Manual deployment (if needed)
npm run build
cp -r out/* public_html/
```

## Contributing

### Content Guidelines

- **Technical accuracy**: Ensure all claims are well-supported
- **Clear writing**: Avoid jargon unless necessary
- **LaTeX formatting**: Use `$inline$` and `$$display$$` syntax
- **Code examples**: Include runnable code snippets
- **References**: Cite papers using numbered references

### Code Guidelines

- **TypeScript**: Use types for all functions
- **ESLint**: Follow the configured linting rules
- **Components**: Keep components small and focused
- **Naming**: Use descriptive variable and function names

### Pull Request Process

1. Create a descriptive branch name
2. Write clear commit messages
3. Ensure all CI checks pass
4. Request review from instructor
5. Address feedback and update PR
6. Merge when approved

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

## Instructor TODO List

**Course Content Completion Tasks**:

- [ ] **Scratch-0 Assignment**: Create full assignment specification in `content/course/assignments/scratch-0.mdx`
- [ ] **Scratch-1 Assignment**: Create full assignment specification in `content/course/assignments/scratch-1.mdx`
- [ ] **Paper Audit Assignment**: Create full assignment specification in `content/course/assignments/paper-audit.mdx`
- [ ] **Capstone Assignment**: Create full assignment specification in `content/course/assignments/capstone.mdx`
- [ ] **Update /course page**: Link assignments once content is ready (remove "Coming soon" placeholders)
- [ ] **Syllabus**: Upload current syllabus to Canvas and verify link works
- [ ] **Textbook Chapters**: Begin writing content for 8 chapters
- [ ] **Example Paper Audit**: Create sample audit for students to reference
- [ ] **Setup Branch Protection**: Configure GitHub branch protection rules (students require PRs, instructor can bypass)

**Deployment**:
- [ ] Test assignment submission workflow end-to-end
- [ ] Verify CI/CD pipeline catches common errors
- [ ] Create assignment grading rubric

**Notes**:
- Syllabus is currently linked to Canvas: https://canvas.colorado.edu/courses/134529/files/82424359/download?download_frd=1
- Students must submit 1 paper audit (not 4)
- Capstone tracks: Research or Engineering (no survey track)

## Contact

- **Instructor**: Christoffer Heckman
- **Email**: christoffer.heckman@colorado.edu
- **Course**: CSCI 7000, Spring 2026
- **GitHub**: https://github.com/arpg/vla-foundations

## License

Copyright © 2026 Christoffer Heckman. All rights reserved.

Course materials are for educational use by enrolled students only.
