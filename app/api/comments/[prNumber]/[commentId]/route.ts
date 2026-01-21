import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

interface Comment {
  id: string;
  sectionId: string;
  text: string;
  author: string;
  timestamp: string;
  resolved: boolean;
}

interface CommentData {
  prNumber: number;
  auditSlug?: string;
  comments: Comment[];
}

const commentsDir = path.join(process.cwd(), 'database', 'comments');

function getCommentFilePath(prNumber: string): string {
  return path.join(commentsDir, `${prNumber}.json`);
}

function readComments(prNumber: string): CommentData {
  const filePath = getCommentFilePath(prNumber);

  if (!fs.existsSync(filePath)) {
    return {
      prNumber: parseInt(prNumber),
      comments: [],
    };
  }

  try {
    const data = fs.readFileSync(filePath, 'utf8');
    return JSON.parse(data);
  } catch (error) {
    console.error('Error reading comments:', error);
    return {
      prNumber: parseInt(prNumber),
      comments: [],
    };
  }
}

function writeComments(prNumber: string, data: CommentData): void {
  const filePath = getCommentFilePath(prNumber);
  fs.writeFileSync(filePath, JSON.stringify(data, null, 2), 'utf8');
}

// PUT /api/comments/[prNumber]/[commentId]
export async function PUT(
  request: NextRequest,
  { params }: { params: Promise<{ prNumber: string; commentId: string }> }
) {
  try {
    const { prNumber, commentId } = await params;
    const body = await request.json();
    const { text, resolved } = body;

    const data = readComments(prNumber);
    const commentIndex = data.comments.findIndex(c => c.id === commentId);

    if (commentIndex === -1) {
      return NextResponse.json(
        { error: 'Comment not found' },
        { status: 404 }
      );
    }

    // Update comment
    if (text !== undefined) {
      data.comments[commentIndex].text = text;
    }
    if (resolved !== undefined) {
      data.comments[commentIndex].resolved = resolved;
    }

    writeComments(prNumber, data);

    return NextResponse.json(data.comments[commentIndex]);
  } catch (error) {
    console.error('Error updating comment:', error);
    return NextResponse.json(
      { error: 'Failed to update comment' },
      { status: 500 }
    );
  }
}

// DELETE /api/comments/[prNumber]/[commentId]
export async function DELETE(
  request: NextRequest,
  { params }: { params: Promise<{ prNumber: string; commentId: string }> }
) {
  try {
    const { prNumber, commentId } = await params;
    const data = readComments(prNumber);

    data.comments = data.comments.filter(c => c.id !== commentId);
    writeComments(prNumber, data);

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('Error deleting comment:', error);
    return NextResponse.json(
      { error: 'Failed to delete comment' },
      { status: 500 }
    );
  }
}
