import { ReactNode } from 'react';
import { Sidebar } from './Sidebar';
import { getAllChapters } from '@/lib/chapters';

interface TextbookLayoutProps {
  children: ReactNode;
}

export function TextbookLayout({ children }: TextbookLayoutProps) {
  const chapters = getAllChapters();
  
  return (
    <div className="flex min-h-screen">
      <Sidebar chapters={chapters} />
      
      <main className="flex-1 flex">
        <article className="flex-1 max-w-4xl mx-auto px-8 py-12">
          <div className="prose prose-lg prose-slate max-w-none">
            {children}
          </div>
        </article>
        
        {/* Table of Contents - Right sidebar */}
        <aside className="hidden xl:block w-64 border-l border-slate-300 bg-gradient-to-b from-slate-50 to-slate-100 p-6 overflow-y-auto h-screen sticky top-0">
          <div className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3 flex items-center gap-2">
            <span className="w-1 h-4 bg-teal-500 rounded-full"></span>
            On This Page
          </div>
          <div className="text-sm text-slate-600">
            {/* TOC will be populated dynamically */}
            <p className="text-xs italic text-slate-500">Table of contents</p>
          </div>
        </aside>
      </main>
    </div>
  );
}
