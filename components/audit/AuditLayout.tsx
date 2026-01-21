'use client';

import { ReactNode, useState } from 'react';
import { CommentSidebar } from './CommentSidebar';

interface AuditLayoutProps {
  children: ReactNode;
  prNumber?: number;
  slug: string;
  isReviewMode?: boolean;
}

export function AuditLayout({ children, prNumber, slug, isReviewMode = false }: AuditLayoutProps) {
  const [showComments, setShowComments] = useState(isReviewMode);

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="flex gap-8">
          {/* Main content */}
          <article
            id="audit-content"
            className={`flex-1 bg-white rounded-lg shadow-sm p-8 prose prose-lg max-w-none ${
              showComments ? 'lg:max-w-3xl' : ''
            }`}
          >
            {children}
          </article>

          {/* Comment sidebar - only show in review mode */}
          {isReviewMode && prNumber && (
            <CommentSidebar
              prNumber={prNumber}
              slug={slug}
              visible={showComments}
              onToggle={() => setShowComments(!showComments)}
            />
          )}
        </div>

        {/* Floating toggle button for comments */}
        {isReviewMode && prNumber && (
          <button
            onClick={() => setShowComments(!showComments)}
            className="fixed bottom-8 right-8 bg-blue-600 hover:bg-blue-700 text-white rounded-full p-4 shadow-lg transition-colors"
            aria-label="Toggle comments"
          >
            <svg
              className="w-6 h-6"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M7 8h10M7 12h4m1 8l-4-4H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-3l-4 4z"
              />
            </svg>
            {showComments && (
              <span className="absolute -top-1 -right-1 bg-red-500 text-white text-xs rounded-full w-5 h-5 flex items-center justify-center">
                !
              </span>
            )}
          </button>
        )}
      </div>
    </div>
  );
}
