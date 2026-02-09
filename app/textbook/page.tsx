import Link from 'next/link';
import { getAllChapters } from '@/lib/chapters';

export const metadata = {
  title: 'VLA Foundations Textbook',
  description: 'A living textbook on Vision-Language-Action models for robotics',
};

export default function TextbookIndexPage() {
  const chapters = getAllChapters();

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-slate-50">
      <div className="max-w-5xl mx-auto px-8 py-16">
        {/* Header */}
        <div className="text-center mb-16">
          <h1 className="text-5xl font-bold bg-gradient-to-r from-slate-800 to-slate-600 bg-clip-text text-transparent mb-4">
            VLA Foundations
          </h1>
          <p className="text-xl text-slate-600 mb-2">
            A Living Textbook on Vision-Language-Action Models
          </p>
          <p className="text-lg text-slate-500">
            Vision-Language-Action for Robotics
          </p>
        </div>

        {/* Work in Progress Notice */}
        <div className="mb-12 p-8 bg-gradient-to-r from-amber-50 to-yellow-50 border-2 border-amber-200 rounded-xl shadow-sm">
          <div className="flex items-start gap-4">
            <div className="flex-shrink-0">
              <svg className="w-8 h-8 text-amber-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
              </svg>
            </div>
            <div className="flex-1">
              <h2 className="text-2xl font-bold text-amber-900 mb-2">
                🚧 Work in Progress
              </h2>
              <p className="text-base text-amber-800 mb-3 leading-relaxed">
                This textbook is actively under development as a <strong>living document</strong>.
                Chapters are being written, refined, and enhanced continuously.
              </p>
              <p className="text-base text-amber-800 leading-relaxed">
                Feel free to explore the available chapters below to get a sense of the format,
                depth, and pedagogical approach. New content is added regularly.
              </p>
            </div>
          </div>
        </div>

        {/* About Section */}
        <div className="mb-12 prose prose-lg prose-slate max-w-none">
          <h2 className="text-3xl font-bold text-slate-800 mb-4">About This Textbook</h2>
          <p className="text-slate-700 leading-relaxed">
            This textbook provides a comprehensive, rigorous treatment of Vision-Language-Action (VLA)
            models for robotic manipulation. Rather than surface-level summaries, we conduct deep
            <strong> technical audits</strong> of seminal papers, focusing on:
          </p>
          <ul className="text-slate-700 space-y-2">
            <li><strong>The Lineage of Failure</strong> - Why previous approaches failed</li>
            <li><strong>Intuitive Derivations</strong> - Geometric and mathematical foundations</li>
            <li><strong>Implementation Gotchas</strong> - Practitioner insights and debugging notes</li>
          </ul>
          <p className="text-slate-700 leading-relaxed">
            Each chapter is structured around the <strong>Interface</strong>: how models project
            pixels to tokens, transform tokens to trajectories, and learn through carefully designed
            loss functions.
          </p>
        </div>

        {/* Chapter Grid */}
        <div className="mb-12">
          <h2 className="text-3xl font-bold text-slate-800 mb-6 flex items-center gap-3">
            <span className="w-2 h-8 bg-gradient-to-b from-emerald-500 to-teal-500 rounded-full"></span>
            Available Chapters
          </h2>
          <div className="grid md:grid-cols-2 gap-6">
            {chapters.map((chapter) => (
              <Link
                key={chapter.slug}
                href={`/textbook/${chapter.slug}`}
                className="group block p-6 bg-white border-2 border-slate-200 rounded-xl hover:border-emerald-400 hover:shadow-xl transition-all duration-300 hover:-translate-y-1"
              >
                <div className="flex items-start gap-4">
                  <div className="flex-shrink-0 w-12 h-12 bg-gradient-to-br from-emerald-500 to-teal-500 rounded-lg flex items-center justify-center text-white font-bold text-lg shadow-md">
                    {chapter.chapter}
                  </div>
                  <div className="flex-1">
                    <h3 className="text-xl font-bold text-slate-800 mb-2 group-hover:text-emerald-600 transition-colors">
                      {chapter.title}
                    </h3>
                    <p className="text-sm text-slate-600 leading-relaxed">
                      {chapter.description}
                    </p>
                  </div>
                </div>
              </Link>
            ))}
          </div>
        </div>

        {/* Additional Resources */}
        <div className="p-8 bg-gradient-to-r from-slate-100 to-slate-50 border-2 border-slate-300 rounded-xl">
          <h2 className="text-2xl font-bold text-slate-800 mb-4">Additional Resources</h2>
          <div className="space-y-3">
            <Link
              href="/textbook/audits"
              className="block p-4 bg-white border border-slate-200 rounded-lg hover:border-teal-400 hover:shadow-md transition-all"
            >
              <h3 className="text-lg font-semibold text-slate-800 mb-1">
                📚 Student Paper Audits
              </h3>
              <p className="text-sm text-slate-600">
                Deep technical reviews of VLA research papers by course students
              </p>
            </Link>
            <Link
              href="/reference"
              className="block p-4 bg-white border border-slate-200 rounded-lg hover:border-teal-400 hover:shadow-md transition-all"
            >
              <h3 className="text-lg font-semibold text-slate-800 mb-1">
                🔧 Reference Implementations
              </h3>
              <p className="text-sm text-slate-600">
                Production-quality code examples and implementation guides
              </p>
            </Link>
          </div>
        </div>

        {/* Footer */}
        <div className="mt-16 pt-8 border-t border-slate-200 text-center">
          <p className="text-sm text-slate-500">
            Part of the <Link href="/" className="text-emerald-600 hover:text-emerald-700 font-medium">VLA Stack</Link> course
          </p>
        </div>
      </div>
    </div>
  );
}
