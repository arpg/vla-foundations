# Paper Audit Submission and Review Guide

## For Students: Submitting a Paper Audit

### Overview
Paper audits are technical deep-dives into VLM and robotics papers. Your audit will be published on the course website after instructor review.

### Submission Process

1. **Create your audit file**
   - Create a new `.mdx` file in `content/textbook/audits/`
   - Filename format: `your_topic_audit.mdx` (use underscores, lowercase)
   - Example: `llava_architecture_audit.mdx`

2. **Required frontmatter**
   ```mdx
   ---
   title: "Your Paper Title - Architecture Analysis"
   author: "Your GitHub Username"
   prNumber: 0  # Will be updated when you create the PR
   ---
   ```

3. **Content structure**
   Your audit should include:
   - **Introduction**: Brief overview of the paper and its significance
   - **Architecture/Methods**: Technical deep-dive into the approach
   - **Evaluation**: Analysis of experiments and results
   - **Critique**: Strengths, weaknesses, and open questions
   - **Conclusion**: Key takeaways and implications

4. **Use LaTeX for math**
   ```mdx
   Inline math: $P(a|s,l)$

   Display math:
   $$
   \text{loss} = -\log P(a^* | s, l)
   $$
   ```

5. **Create a Pull Request to `staging` branch**
   ```bash
   git checkout -b my-paper-audit
   git add content/textbook/audits/your_audit.mdx
   git commit -m "Add [Paper Name] audit"
   git push origin my-paper-audit

   # Create PR targeting staging branch
   gh pr create --base staging --title "Paper Audit: [Paper Name]"
   ```

6. **Update the PR number**
   - After creating the PR, note the PR number (e.g., #25)
   - Update the `prNumber` in your audit's frontmatter
   - Commit and push this change

### Review Process

1. **View your rendered audit**
   - Once merged to staging, your audit appears at:
   - `https://vlm-robotics.dev/textbook/audits/your_topic_audit/`

2. **Respond to instructor comments**
   - Instructor will add comments directly on your PR
   - Make revisions in your branch
   - Push updates to the same PR
   - Comment on the PR when ready for re-review

3. **Publication**
   - Once approved, your audit will be merged to main
   - It becomes part of the permanent textbook

---

## For Instructors: Reviewing Paper Audits

### Review Workflow

1. **Student creates PR to `staging` branch**
   ```bash
   # List open PRs targeting staging
   gh pr list --base staging
   ```

2. **Merge to staging for preview**
   ```bash
   # Review the PR
   gh pr view [PR-NUMBER]

   # Merge to staging
   gh pr merge [PR-NUMBER] --merge
   ```

   This automatically triggers deployment to the live site.

3. **Review the rendered page**
   - Visit: `https://vlm-robotics.dev/textbook/audits/[slug]/`
   - Check formatting, LaTeX rendering, and content quality

4. **Add review comments on GitHub**
   ```bash
   # Start a PR review
   gh pr review [PR-NUMBER]
   ```

   Or use the GitHub web interface:
   - Go to the PR page
   - Click "Files changed"
   - Add inline comments on the MDX source
   - Submit review with "Request changes" or "Comment"

5. **Student revises based on feedback**
   - Student pushes updates to their branch
   - Updates automatically deploy to staging
   - Refresh the live URL to see changes

6. **Approve and publish**
   When satisfied:
   ```bash
   # Approve the PR
   gh pr review [PR-NUMBER] --approve

   # Merge staging to main
   git checkout main
   git merge staging
   git push origin main
   ```

   This publishes the audit to the production site.

### Quality Checklist

Before approving, verify:

- [ ] **Technical accuracy**: Claims are supported and correct
- [ ] **Depth**: Sufficient detail in architecture/methods explanation
- [ ] **Clarity**: Writing is clear and well-organized
- [ ] **Math rendering**: All LaTeX formulas render correctly
- [ ] **Citations**: Key claims reference the paper
- [ ] **Critical analysis**: Includes thoughtful critique, not just summary
- [ ] **Frontmatter**: Title, author, and PR number are correct

### Managing Multiple Audits

View all audits in progress:
```bash
# See all staging PRs
gh pr list --base staging --state open

# See all merged audits awaiting publication
gh pr list --base staging --state merged
```

Bulk publish audits:
```bash
# After reviewing multiple audits on staging
git checkout main
git merge staging
git push origin main
```

### Troubleshooting

**Audit not appearing on site?**
- Check GitHub Actions: `gh run list --workflow="Deploy Staging Branch"`
- View deployment logs: `gh run view [RUN-ID]`
- Verify file is in `content/textbook/audits/` with `.mdx` extension

**LaTeX not rendering?**
- Check that math is wrapped in `$...$` or `$$...$$`
- Verify no special characters need escaping
- Test locally with `npm run dev`

**Styling issues?**
- Audits inherit textbook styling automatically
- Use standard Markdown/MDX - no custom CSS needed
- Report persistent issues on GitHub

---

## Technical Details

### Automatic Deployment

Pushing to the `staging` branch triggers:
1. GitHub Actions workflow runs on the server
2. Pulls latest code from `staging` branch
3. Runs `npm install` and `npm run build`
4. Deploys static files to web server
5. Site updates at `vlm-robotics.dev`

### File Structure

```
content/textbook/audits/
├── README.md                    # This file
├── multimodality_audit.mdx      # Example audit
└── [student_audits].mdx         # Student submissions

app/textbook/audits/
├── page.tsx                     # Audits index page
└── [slug]/page.tsx              # Individual audit page

components/audit/
└── AuditLayout.tsx              # Page layout component
```

### Adding Audit Metadata

Edit `lib/audits.ts` to add your audit to the index:
```typescript
{
  title: 'Your Paper Title',
  slug: 'your_topic_audit',
  author: 'username',
  prNumber: 25,
  description: 'Brief description'
}
```

This makes it appear on the audits index page.
