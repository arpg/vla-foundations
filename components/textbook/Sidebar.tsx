'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { ChapterMetadata } from '@/lib/chapters';

interface SidebarProps {
  chapters: ChapterMetadata[];
}

export function Sidebar({ chapters }: SidebarProps) {
  const pathname = usePathname();
  
  return (
    <aside className="w-64 border-r border-slate-300 bg-gradient-to-b from-slate-50 to-slate-100 p-6 overflow-y-auto h-screen sticky top-0">
      <div className="mb-8">
        <Link href="/" className="block group">
          <h1 className="text-2xl font-bold bg-gradient-to-r from-slate-800 to-slate-600 bg-clip-text text-transparent group-hover:from-emerald-600 group-hover:to-teal-600 transition-all">VLA Stack</h1>
          <p className="text-sm text-slate-600 mt-1">Vision-Language-Action</p>
        </Link>
      </div>

      <nav className="space-y-1">
        <Link
          href="/textbook"
          className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3 flex items-center gap-2 hover:text-emerald-600 transition-colors group"
        >
          <span className="w-1 h-4 bg-emerald-500 rounded-full group-hover:bg-emerald-600"></span>
          Living Textbook
        </Link>

        {chapters.map((chapter) => {
          const href = `/textbook/${chapter.slug}`;
          const isActive = pathname === href;

          return (
            <Link
              key={chapter.slug}
              href={href}
              className={`block px-3 py-2 rounded-lg text-sm transition-all duration-200 ${
                isActive
                  ? 'bg-gradient-to-r from-emerald-500 to-teal-500 text-white font-semibold shadow-md shadow-emerald-200'
                  : 'text-slate-700 hover:bg-slate-200 hover:text-slate-900 hover:translate-x-0.5'
              }`}
            >
              <div className="flex items-baseline gap-2">
                <span className={`text-xs font-mono ${isActive ? 'text-emerald-100' : 'text-slate-500'}`}>{chapter.chapter}.</span>
                <span className="flex-1">{chapter.title}</span>
              </div>
            </Link>
          );
        })}
      </nav>

      <div className="mt-8 pt-8 border-t border-slate-300">
        <nav className="space-y-1">
          <Link
            href="/reference"
            className="block px-3 py-2 rounded-lg text-sm text-slate-700 hover:bg-slate-200 hover:text-slate-900 transition-all hover:translate-x-0.5"
          >
            Reference Implementations
          </Link>
        </nav>
      </div>
    </aside>
  );
}
