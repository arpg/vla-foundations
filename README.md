# VLA Foundations

Vision-Language-Action Foundations course repository for the ARPG organization.

## Repository Structure

```
vla-foundations/
├── content/
│   ├── course/
│   │   └── assignments/    # Course assignments and materials
│   └── contributors/        # Student contributor profiles
├── .github/
│   └── workflows/           # CI/CD automation
└── README.md
```

## Development

This repository uses:
- **pnpm** for package management
- **Contentlayer** for content validation
- **GitHub Actions** for automated content audits

## Branching Strategy

- `main`: Production branch
- `staging`: Student PR target branch

## Contributing

Students must:
1. Create a contributor profile in `content/contributors/`
2. Submit assignments via pull requests to the `staging` branch
3. Ensure all CI checks pass before merge

## Content Audit

All PRs are automatically validated for:
- Contentlayer build success
- LaTeX notation compliance
- Build integrity

## Setup

```bash
pnpm install
pnpm build-content
pnpm build
```
