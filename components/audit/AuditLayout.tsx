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
}

export function AuditLayout({ children, chapters }: AuditLayoutProps) {
  return (
    <div className="flex min-h-screen">
      <Sidebar chapters={chapters} />

      <main className="flex-1 flex">
        <article className="flex-1 max-w-4xl mx-auto px-8 py-12">
          <div className="prose prose-lg prose-slate max-w-none">
            {children}
          </div>
        </article>

        <aside className="hidden xl:block w-64 border-l border-gray-200 bg-gray-50 p-6 overflow-y-auto h-screen sticky top-0">
          <div className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-3">
            On This Page
          </div>
          <div className="text-sm text-gray-600">
            <p className="text-xs italic">Table of contents</p>
          </div>
        </aside>
      </main>
    </div>
  );
}
