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
    <aside className="w-64 border-r border-gray-200 bg-gray-50 p-6 overflow-y-auto h-screen sticky top-0">
      <div className="mb-8">
        <Link href="/" className="block">
          <h1 className="text-2xl font-bold text-gray-900">VLA Stack</h1>
          <p className="text-sm text-gray-600 mt-1">Vision-Language-Action</p>
        </Link>
      </div>
      
      <nav className="space-y-1">
        <div className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-3">
          Living Textbook
        </div>
        
        {chapters.map((chapter) => {
          const href = `/textbook/${chapter.slug}`;
          const isActive = pathname === href;
          
          return (
            <Link
              key={chapter.slug}
              href={href}
              className={`block px-3 py-2 rounded-md text-sm transition-colors ${
                isActive
                  ? 'bg-blue-100 text-blue-900 font-medium'
                  : 'text-gray-700 hover:bg-gray-200'
              }`}
            >
              <div className="flex items-baseline gap-2">
                <span className="text-xs text-gray-500">{chapter.chapter}.</span>
                <span className="flex-1">{chapter.title}</span>
              </div>
            </Link>
          );
        })}
      </nav>
      
      <div className="mt-8 pt-8 border-t border-gray-200">
        <nav className="space-y-1">
          <Link
            href="/reference"
            className="block px-3 py-2 rounded-md text-sm text-gray-700 hover:bg-gray-200"
          >
            Reference Implementations
          </Link>
        </nav>
      </div>
    </aside>
  );
}
