# Student Paper Audit Review System

## Overview

This system enables instructors to review student paper audits with inline commenting on rendered pages. Comments persist across student revisions and are stored in the git repository.

## Architecture

### Components

1. **Audit Pages** (`/textbook/audits/[slug]`)
   - Renders student paper audits from MDX files
   - Located in `content/textbook/audits/`
   - Supports both public view and instructor review mode

2. **Review Mode** (`/textbook/audits/[slug]?review=true`)
   - Adds commenting sidebar for instructor feedback
   - Comments tied to PR numbers for version tracking
   - Visual interface for adding/resolving comments

3. **Comments API** (`/api/comments/`)
   - RESTful API for managing comments
   - Comments stored in `database/comments/{pr-number}.json`
   - Persists in git repository for audit trail

4. **Staging Deployment**
   - Staging branch deploys to `vlm-robotics.dev/staging/`
   - Automatic deployment on push to `staging` branch
   - Allows review before merging to production

## Workflow

### For Students:

1. Create paper audit in `content/textbook/audits/{topic}_audit.mdx`
2. Add frontmatter with metadata:
   ```yaml
   ---
   title: 'Your Audit Title'
   author: 'your-github-username'
   prNumber: XX
   ---
   ```
3. Submit PR to `staging` branch
4. Wait for instructor review
5. Address comments and push updates
6. PR merged to `main` when approved

### For Instructors:

1. Student submits PR to `staging` branch
2. Merge PR to `staging` (triggers auto-deploy)
3. Visit `vlm-robotics.dev/staging/textbook/audits/{slug}?review=true`
4. Add comments using the sidebar:
   - Select section (Introduction, Architecture, etc.)
   - Write feedback
   - Comments automatically saved
5. Student sees comments, makes revisions
6. Mark comments as resolved as student addresses them
7. When satisfied, merge staging → main

## File Structure

```
vla-foundations/
├── app/
│   ├── textbook/audits/
│   │   ├── [slug]/page.tsx          # Individual audit page
│   │   └── page.tsx                 # Audits index page
│   └── api/comments/
│       └── [prNumber]/
│           ├── route.ts             # GET/POST comments
│           └── [commentId]/route.ts # PUT/DELETE comment
├── components/audit/
│   ├── AuditLayout.tsx              # Layout with comment sidebar
│   └── CommentSidebar.tsx           # Comment UI component
├── content/textbook/audits/
│   └── *.mdx                        # Student audit submissions
├── database/comments/
│   └── {pr-number}.json             # Comments for each PR
└── lib/
    └── audits.ts                    # Audit data management
```

## Comment Data Structure

Comments are stored in JSON files:

```json
{
  "prNumber": 23,
  "auditSlug": "multimodality_audit",
  "comments": [
    {
      "id": "uuid",
      "sectionId": "introduction",
      "text": "Great start! Consider expanding on the architectural implications.",
      "author": "instructor",
      "timestamp": "2026-01-21T...",
      "resolved": false
    }
  ]
}
```

## Deployment

### Staging
- **Trigger**: Push to `staging` branch
- **URL**: `vlm-robotics.dev/staging/`
- **Purpose**: Review student work before production

### Production
- **Trigger**: Push to `main` branch
- **URL**: `vlm-robotics.dev/`
- **Purpose**: Published course content

## API Endpoints

- `GET /api/comments/{prNumber}` - Fetch all comments for a PR
- `POST /api/comments/{prNumber}` - Add new comment
- `PUT /api/comments/{prNumber}/{commentId}` - Update comment (mark resolved)
- `DELETE /api/comments/{prNumber}/{commentId}` - Delete comment

## Future Enhancements

- [ ] Authentication for instructor-only access
- [ ] Email notifications when comments are added
- [ ] Auto-generate section IDs from headings
- [ ] Comment threading/replies
- [ ] Rich text formatting in comments
- [ ] Export comments to PDF for grading records
