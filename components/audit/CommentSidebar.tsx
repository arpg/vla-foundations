'use client';

import { useState, useEffect } from 'react';

interface Comment {
  id: string;
  sectionId: string;
  text: string;
  author: string;
  timestamp: string;
  resolved: boolean;
}

interface CommentSidebarProps {
  prNumber: number;
  slug: string;
  visible: boolean;
  onToggle: () => void;
}

export function CommentSidebar({ prNumber, slug, visible, onToggle }: CommentSidebarProps) {
  const [comments, setComments] = useState<Comment[]>([]);
  const [loading, setLoading] = useState(true);
  const [newComment, setNewComment] = useState('');
  const [selectedSection, setSelectedSection] = useState('general');

  useEffect(() => {
    fetchComments();
  }, [prNumber]);

  const fetchComments = async () => {
    try {
      const response = await fetch(`/api/comments/${prNumber}`);
      if (response.ok) {
        const data = await response.json();
        setComments(data.comments || []);
      }
    } catch (error) {
      console.error('Failed to fetch comments:', error);
    } finally {
      setLoading(false);
    }
  };

  const addComment = async () => {
    if (!newComment.trim()) return;

    try {
      const response = await fetch(`/api/comments/${prNumber}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          sectionId: selectedSection,
          text: newComment,
        }),
      });

      if (response.ok) {
        const data = await response.json();
        setComments([...comments, data]);
        setNewComment('');
      }
    } catch (error) {
      console.error('Failed to add comment:', error);
    }
  };

  const toggleResolved = async (commentId: string) => {
    try {
      const comment = comments.find(c => c.id === commentId);
      if (!comment) return;

      const response = await fetch(`/api/comments/${prNumber}/${commentId}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          resolved: !comment.resolved,
        }),
      });

      if (response.ok) {
        setComments(comments.map(c =>
          c.id === commentId ? { ...c, resolved: !c.resolved } : c
        ));
      }
    } catch (error) {
      console.error('Failed to update comment:', error);
    }
  };

  if (!visible) return null;

  return (
    <aside className="w-96 bg-white rounded-lg shadow-sm p-6 sticky top-8 h-fit max-h-[calc(100vh-4rem)] overflow-y-auto">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-bold text-gray-900">Review Comments</h2>
        <span className="text-sm text-gray-500">PR #{prNumber}</span>
      </div>

      {/* Add new comment */}
      <div className="mb-6 pb-6 border-b border-gray-200">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Section
        </label>
        <select
          value={selectedSection}
          onChange={(e) => setSelectedSection(e.target.value)}
          className="w-full px-3 py-2 border border-gray-300 rounded-md mb-3 text-sm"
        >
          <option value="general">General</option>
          <option value="introduction">Introduction</option>
          <option value="architecture">Architecture</option>
          <option value="evaluation">Evaluation</option>
          <option value="conclusion">Conclusion</option>
        </select>

        <label className="block text-sm font-medium text-gray-700 mb-2">
          Comment
        </label>
        <textarea
          value={newComment}
          onChange={(e) => setNewComment(e.target.value)}
          placeholder="Add your review comment..."
          className="w-full px-3 py-2 border border-gray-300 rounded-md mb-3 text-sm"
          rows={4}
        />
        <button
          onClick={addComment}
          disabled={!newComment.trim()}
          className="w-full bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed text-sm font-medium"
        >
          Add Comment
        </button>
      </div>

      {/* Comments list */}
      <div className="space-y-4">
        {loading ? (
          <p className="text-gray-500 text-sm">Loading comments...</p>
        ) : comments.length === 0 ? (
          <p className="text-gray-500 text-sm">No comments yet. Add the first one!</p>
        ) : (
          comments.map((comment) => (
            <div
              key={comment.id}
              className={`p-4 rounded-lg border ${
                comment.resolved
                  ? 'bg-green-50 border-green-200'
                  : 'bg-yellow-50 border-yellow-200'
              }`}
            >
              <div className="flex items-start justify-between mb-2">
                <span className="text-xs font-medium text-gray-600 uppercase">
                  {comment.sectionId}
                </span>
                <button
                  onClick={() => toggleResolved(comment.id)}
                  className="text-xs text-blue-600 hover:text-blue-800"
                >
                  {comment.resolved ? '✓ Resolved' : 'Mark Resolved'}
                </button>
              </div>
              <p className="text-sm text-gray-800 mb-2">{comment.text}</p>
              <div className="text-xs text-gray-500">
                {comment.author} • {new Date(comment.timestamp).toLocaleDateString()}
              </div>
            </div>
          ))
        )}
      </div>
    </aside>
  );
}
