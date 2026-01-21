import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { randomUUID } from 'crypto';

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

// Ensure comments directory exists
if (!fs.existsSync(commentsDir)) {
  fs.mkdirSync(commentsDir, { recursive: true });
}

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

// GET /api/comments/[prNumber]
export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ prNumber: string }> }
) {
  const { prNumber } = await params;
  const data = readComments(prNumber);
  return NextResponse.json(data);
}

// POST /api/comments/[prNumber]
export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ prNumber: string }> }
) {
  try {
    const { prNumber } = await params;
    const body = await request.json();
    const { sectionId, text } = body;

    if (!sectionId || !text) {
      return NextResponse.json(
        { error: 'sectionId and text are required' },
        { status: 400 }
      );
    }

    const data = readComments(prNumber);

    const newComment: Comment = {
      id: randomUUID(),
      sectionId,
      text,
      author: 'instructor', // TODO: Add authentication
      timestamp: new Date().toISOString(),
      resolved: false,
    };

    data.comments.push(newComment);
    writeComments(prNumber, data);

    return NextResponse.json(newComment, { status: 201 });
  } catch (error) {
    console.error('Error creating comment:', error);
    return NextResponse.json(
      { error: 'Failed to create comment' },
      { status: 500 }
    );
  }
}
