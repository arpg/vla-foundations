'use client';

import { ReactNode } from 'react';
import { Sidebar } from '../textbook/Sidebar';

interface Chapter {
  title: string;
  slug: string;
  description: string;
  chapter: number;
}

interface AuditLayoutProps {
  children: ReactNode;
  chapters: Chapter[];
  isReviewMode?: boolean;
  prNumber?: string;
}

export function AuditLayout({ children, chapters, isReviewMode = false, prNumber }: AuditLayoutProps) {
  return (
    <div className="flex min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      <Sidebar chapters={chapters} />

      <main className="flex-1 flex">
        <article className="flex-1 max-w-5xl mx-auto px-8 sm:px-12 lg:px-16 py-12 bg-white shadow-sm">
          {/* Review Mode Banner */}
          {isReviewMode && (
            <div className="mb-8 p-6 bg-gradient-to-r from-amber-50 to-yellow-50 border-2 border-amber-300 rounded-xl shadow-sm">
              <div className="flex items-start gap-4">
                <div className="flex-shrink-0">
                  <svg className="w-6 h-6 text-amber-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                  </svg>
                </div>
                <div className="flex-1">
                  <h3 className="text-lg font-bold text-amber-900 mb-1">
                    👁️ REVIEW MODE
                  </h3>
                  <p className="text-sm text-amber-800 mb-3">
                    You are viewing a preview of this audit. This content is under review and not yet published.
                  </p>
                  {prNumber && (
                    <p className="text-xs text-amber-700 font-mono bg-amber-100 px-3 py-1.5 rounded inline-block">
                      Preview from PR #{prNumber}
                    </p>
                  )}
                </div>
              </div>
            </div>
          )}

          <div className="prose prose-lg prose-slate max-w-none">
            {children}
          </div>
        </article>

        <aside className="hidden xl:block w-72 border-l border-slate-200 bg-gradient-to-b from-slate-50 to-white p-8 overflow-y-auto h-screen sticky top-0">
          <div className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-4 pb-2 border-b border-slate-200">
            On This Page
          </div>
          <div className="text-sm text-slate-600">
            <p className="text-xs italic text-slate-400">Table of contents</p>
          </div>
        </aside>
      </main>
    </div>
  );
}
