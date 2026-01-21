# Instructor Guide: Paper Audit Review System

## Quick Start

### Setting Up for Review

1. **Student submits PR to staging:**
   - Student creates PR from their branch → `staging`
   - PR includes their audit file in `content/textbook/audits/`

2. **Merge to staging for review:**
   ```bash
   gh pr merge 23 --merge --repo arpg/vla-foundations --into staging
   ```
   - This triggers automatic deployment to `vlm-robotics.dev/staging/`
   - Wait ~2-3 minutes for deployment to complete

3. **Access review interface:**
   - Navigate to: `vlm-robotics.dev/staging/textbook/audits/{slug}?review=true`
   - For PR 23 (multimodality audit): `vlm-robotics.dev/staging/textbook/audits/multimodality_audit?review=true`

### Reviewing and Commenting

1. **Add comments:**
   - Click the comment button (bottom right)
   - Select the section you're commenting on
   - Write your feedback
   - Click "Add Comment"
   - Comments automatically save to `database/comments/{pr-number}.json`

2. **Mark comments as resolved:**
   - As students address feedback, click "Mark Resolved"
   - Resolved comments turn green

3. **View all comments:**
   - Comment sidebar shows all feedback for the PR
   - Filter by resolved/unresolved status

### Finalizing the Review

1. **When satisfied with revisions:**
   ```bash
   git checkout staging
   git pull origin staging

   # Merge staging → main to publish
   git checkout main
   git merge staging
   git push origin main
   ```

2. **Published audit appears at:**
   - Production: `vlm-robotics.dev/textbook/audits/{slug}`
   - Comments persist but review UI hidden in production

## Workflow Diagram

```
Student Branch → PR to Staging → Instructor Reviews → Revisions → Merge to Main
                     ↓                    ↓              ↓
              Auto-deploy to       Add comments    Push updates
              /staging/            via web UI      to same PR
```

## URL Structure

- **Staging (review mode):** `vlm-robotics.dev/staging/textbook/audits/{slug}?review=true`
- **Staging (public view):** `vlm-robotics.dev/staging/textbook/audits/{slug}`
- **Production:** `vlm-robotics.dev/textbook/audits/{slug}`
- **Audits index:** `vlm-robotics.dev/textbook/audits/`

## Comment Persistence

- Comments stored in: `database/comments/{pr-number}.json`
- Committed to git repository
- Survive force pushes and rebases (tied to PR number)
- Can be exported for grading records

## Tips

1. **Section Organization:**
   - Use consistent section names across audits
   - Common sections: Introduction, Architecture, Evaluation, Conclusion

2. **Effective Feedback:**
   - Be specific about what needs improvement
   - Reference line numbers or specific claims
   - Suggest resources or examples
   - Mark minor comments vs. critical issues

3. **Student Updates:**
   - Students push to same branch
   - Re-merge to staging to see updates
   - Previous comments remain, check if addressed

4. **Multiple Reviewers:**
   - All comments stored centrally
   - Add author names to comments for multi-instructor courses

## Troubleshooting

**Deployment not updating:**
- Check GitHub Actions: `https://github.com/arpg/vla-foundations/actions`
- SSH to server and check logs: `~/vla-staging/`

**Comments not saving:**
- Check API endpoint: `/api/comments/{prNumber}`
- Verify `database/comments/` directory exists and is writable

**Audit not appearing:**
- Verify MDX file exists in `content/textbook/audits/`
- Check frontmatter includes required fields: `title`, `prNumber`
- Ensure file ends in `.mdx`

## Advanced Features

### Bulk Comment Export
```bash
# Export all comments for grading
cat database/comments/23.json | jq '.comments'
```

### Custom Sections
Update `CommentSidebar.tsx` to add new section options:
```tsx
<option value="your-section">Your Section</option>
```

### Authentication (Future)
Currently no auth required. To add:
- Implement simple password check
- Or integrate GitHub OAuth
- See TODO in `app/api/comments/[prNumber]/route.ts`
