import express from 'express';
import cors from 'cors';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { randomUUID } from 'crypto';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3001;

// Middleware
app.use(cors());
app.use(express.json());

// Comments directory
const commentsDir = path.join(__dirname, '..', 'database', 'comments');

// Ensure comments directory exists
if (!fs.existsSync(commentsDir)) {
  fs.mkdirSync(commentsDir, { recursive: true });
}

function getCommentFilePath(prNumber) {
  return path.join(commentsDir, `${prNumber}.json`);
}

function readComments(prNumber) {
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

function writeComments(prNumber, data) {
  const filePath = getCommentFilePath(prNumber);
  fs.writeFileSync(filePath, JSON.stringify(data, null, 2), 'utf8');
}

// GET /api/comments/:prNumber
app.get('/api/comments/:prNumber', (req, res) => {
  const { prNumber } = req.params;
  const data = readComments(prNumber);
  res.json(data);
});

// POST /api/comments/:prNumber
app.post('/api/comments/:prNumber', (req, res) => {
  try {
    const { prNumber } = req.params;
    const { sectionId, text } = req.body;

    if (!sectionId || !text) {
      return res.status(400).json({ error: 'sectionId and text are required' });
    }

    const data = readComments(prNumber);

    const newComment = {
      id: randomUUID(),
      sectionId,
      text,
      author: 'instructor',
      timestamp: new Date().toISOString(),
      resolved: false,
    };

    data.comments.push(newComment);
    writeComments(prNumber, data);

    res.status(201).json(newComment);
  } catch (error) {
    console.error('Error creating comment:', error);
    res.status(500).json({ error: 'Failed to create comment' });
  }
});

// PUT /api/comments/:prNumber/:commentId
app.put('/api/comments/:prNumber/:commentId', (req, res) => {
  try {
    const { prNumber, commentId } = req.params;
    const { text, resolved } = req.body;

    const data = readComments(prNumber);
    const commentIndex = data.comments.findIndex(c => c.id === commentId);

    if (commentIndex === -1) {
      return res.status(404).json({ error: 'Comment not found' });
    }

    if (text !== undefined) {
      data.comments[commentIndex].text = text;
    }
    if (resolved !== undefined) {
      data.comments[commentIndex].resolved = resolved;
    }

    writeComments(prNumber, data);

    res.json(data.comments[commentIndex]);
  } catch (error) {
    console.error('Error updating comment:', error);
    res.status(500).json({ error: 'Failed to update comment' });
  }
});

// DELETE /api/comments/:prNumber/:commentId
app.delete('/api/comments/:prNumber/:commentId', (req, res) => {
  try {
    const { prNumber, commentId } = req.params;
    const data = readComments(prNumber);

    data.comments = data.comments.filter(c => c.id !== commentId);
    writeComments(prNumber, data);

    res.json({ success: true });
  } catch (error) {
    console.error('Error deleting comment:', error);
    res.status(500).json({ error: 'Failed to delete comment' });
  }
});

// Health check
app.get('/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

app.listen(PORT, () => {
  console.log(`VLA Comments API server running on port ${PORT}`);
  console.log(`Comments directory: ${commentsDir}`);
});
