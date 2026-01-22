# Student Paper Audit Review System - COMPLETE ✅

## System Status: FULLY OPERATIONAL

The student paper audit review system is fully deployed at http://vlm-robotics.dev/staging/ with complete textbook styling integration.

## Features

### 1. Consistent Styling with Textbook Pages
- **Left Sidebar**: Chapter navigation (Foundations, Architectures, Data, etc.)
- **Main Content**: Full-width article with MDX rendering and LaTeX support
- **Right Sidebar**:
  - Table of Contents placeholder (normal view)
  - Review Comments interface (review mode with `?review=true`)

### 2. Audit Pages

**Normal View**: http://vlm-robotics.dev/staging/textbook/audits/multimodality_audit/
- Matches textbook page styling exactly
- Left navigation sidebar
- Right TOC sidebar
- Professional academic layout

**Review Mode**: http://vlm-robotics.dev/staging/textbook/audits/multimodality_audit/?review=true
- Same left navigation
- **Right sidebar becomes comment interface**:
  - PR #23 badge
  - Section dropdown (General, Introduction, Architecture, Evaluation, Conclusion)
  - Comment textarea
  - "Add Comment" button
  - Live comment list with resolve/unresolve functionality
  - Floating comment toggle button (bottom right corner)

### 3. Working API Routes
- `GET /api/comments/[prNumber]/` - Fetch all comments
- `POST /api/comments/[prNumber]/` - Add new comment
- `PUT /api/comments/[prNumber]/[commentId]/` - Update comment (mark resolved/unresolved)
- `DELETE /api/comments/[prNumber]/[commentId]/` - Delete comment

**Test**: http://vlm-robotics.dev/staging/api/comments/23/ returns JSON with 2 demo comments

### 4. Demo Content
PR #23 has example audit from gyanigkali with 2 demo comments:
- Comment on Introduction section
- Comment on Architecture section

## Your Review Workflow

### When a Student Submits an Audit:

1. **Student creates PR to `staging`**
   ```bash
   gh pr list --base staging
   ```

2. **Merge to staging**
   ```bash
   gh pr merge [PR-NUMBER] --merge
   ```

3. **Deploy to server**
   ```bash
   ssh crh@ristoffer.ch
   cd ~/vla-staging
   git pull origin staging
   npm run build
   kill -9 $(ps aux | grep 'next-server' | grep -v grep | awk '{print $2}')
   nohup npm start -- -p 3002 > ~/vla-staging-server.log 2>&1 &
   ```

4. **Review the audit**
   - Visit: `http://vlm-robotics.dev/staging/textbook/audits/[slug]/?review=true`
   - Read the audit in the main content area
   - Use the right sidebar to:
     - Select section from dropdown
     - Type your comment
     - Click "Add Comment"
   - Comments save automatically to `database/comments/[pr-number].json`

5. **Student revises**
   - Student sees comments at the same review URL
   - Student updates their PR
   - You pull changes and repeat review

6. **Approve and publish**
   ```bash
   git checkout main
   git merge staging
   git push origin main
   ```

## Technical Architecture

### Page Layout Structure
```
┌─────────────┬──────────────────────┬─────────────────┐
│   Sidebar   │   Main Content       │  Right Sidebar  │
│             │                      │                 │
│ Foundations │ # Audit Title        │ Normal:         │
│ Architectures│                     │  - TOC          │
│ Data        │ MDX content with     │                 │
│ Training    │ LaTeX math           │ Review Mode:    │
│ Evaluation  │                      │  - Comments     │
│ ...         │                      │  - Add Comment  │
│             │                      │  - PR Badge     │
└─────────────┴──────────────────────┴─────────────────┘
```

### Component Hierarchy
- `app/textbook/audits/[slug]/page.tsx` - Server component, fetches data
  - `components/audit/AuditLayout.tsx` - Client component, handles layout
    - `components/textbook/Sidebar.tsx` - Left navigation
    - Main `<article>` - MDX content
    - `components/audit/CommentSidebar.tsx` - Right sidebar (review mode)

### Data Flow
```
User visits ?review=true
    ↓
Server: page.tsx passes isReviewMode={true}
    ↓
Client: AuditLayout renders CommentSidebar
    ↓
Client: CommentSidebar fetches /api/comments/23/
    ↓
Server: API route reads database/comments/23.json
    ↓
Client: Renders comments in sidebar
    ↓
User adds comment → POST /api/comments/23/
    ↓
Server: Appends to JSON file, returns new comment
    ↓
Client: Updates UI with new comment
```

## File Structure

### Server (~/vla-staging/)
```
app/
├── textbook/audits/
│   ├── page.tsx                 # Audits index
│   └── [slug]/page.tsx          # Individual audit pages
├── api/comments/
│   ├── [prNumber]/route.ts      # GET/POST comments
│   └── [prNumber]/[commentId]/route.ts  # PUT/DELETE comment

components/
├── audit/
│   ├── AuditLayout.tsx          # Page layout with sidebar switching
│   └── CommentSidebar.tsx       # Comment UI
└── textbook/
    └── Sidebar.tsx              # Left navigation (shared with textbook)

content/textbook/audits/
└── multimodality_audit.mdx      # Example audit

database/comments/
└── 23.json                      # Comments for PR 23

lib/
└── audits.ts                    # Audit metadata utilities
```

### Apache Configuration
```apache
ProxyPass /staging/ http://localhost:3002/
ProxyPassReverse /staging/ http://localhost:3002/
```

### Next.js Configuration
```typescript
// next.config.ts
{
  // Dynamic mode (no static export)
  trailingSlash: true,
  images: { unoptimized: true }
}
```

## Server Management

**Check Status:**
```bash
ssh crh@ristoffer.ch "ps aux | grep next-server | grep -v grep"
```

**View Logs:**
```bash
ssh crh@ristoffer.ch "tail -f ~/vla-staging-server.log"
```

**Restart Server:**
```bash
ssh crh@ristoffer.ch "cd ~/vla-staging && kill -9 \$(ps aux | grep 'next-server' | grep -v grep | awk '{print \$2}') && nohup npm start -- -p 3002 > ~/vla-staging-server.log 2>&1 &"
```

**Test API:**
```bash
curl http://vlm-robotics.dev/staging/api/comments/23/
```

## System Verification

All components tested and operational:
- ✅ Left sidebar navigation (matches textbook)
- ✅ Main content with MDX + LaTeX rendering
- ✅ Right sidebar switches between TOC and Comments
- ✅ Review mode activates with `?review=true`
- ✅ Comment form functional
- ✅ API routes returning JSON
- ✅ Comment storage working
- ✅ Demo comments visible
- ✅ Floating toggle button (bottom right)
- ✅ Server healthy and running

## What Changed

### From Previous Version:
1. **Added left sidebar navigation** - Now matches textbook pages exactly
2. **Improved styling consistency** - Uses same prose classes as textbook
3. **Right sidebar dual-purpose** - TOC in normal view, comments in review mode
4. **Better visual integration** - Audits feel like part of the textbook

### Final Fixes Applied:
1. Updated `AuditLayout.tsx` to include `Sidebar` component
2. Fixed TypeScript interface to use `chapter` instead of `order`
3. Passed `chapters` prop from server to client component
4. Updated `CommentSidebar.tsx` styling to match TOC sidebar

## Ready for Use! 🎉

The system is production-ready. Visit the review URL to test:
http://vlm-robotics.dev/staging/textbook/audits/multimodality_audit/?review=true

Students can now submit paper audits, and you can review them with inline comments in a professional academic interface that matches your textbook styling.
