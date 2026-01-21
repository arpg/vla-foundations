import { getAllAudits } from '@/lib/audits';
import Link from 'next/link';

export default function AuditsIndexPage() {
  const audits = getAllAudits();

  return (
    <div className="min-h-screen bg-gray-50 py-12">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="bg-white rounded-lg shadow-sm p-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            Student Paper Audits
          </h1>
          <p className="text-lg text-gray-600 mb-8">
            Technical deep-dives into VLM and robotics papers by course students.
          </p>

          {audits.length === 0 ? (
            <div className="text-center py-12">
              <p className="text-gray-500 text-lg">
                No audits available yet. Check back soon!
              </p>
            </div>
          ) : (
            <div className="space-y-6">
              {audits.map((audit) => (
                <Link
                  key={audit.slug}
                  href={`/textbook/audits/${audit.slug}`}
                  className="block p-6 border border-gray-200 rounded-lg hover:border-blue-500 hover:shadow-md transition-all"
                >
                  <div className="flex items-start justify-between">
                    <div>
                      <h2 className="text-xl font-semibold text-gray-900 mb-2">
                        {audit.title}
                      </h2>
                      {audit.author && (
                        <p className="text-sm text-gray-600">
                          By {audit.author}
                        </p>
                      )}
                    </div>
                    {audit.prNumber && (
                      <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                        PR #{audit.prNumber}
                      </span>
                    )}
                  </div>
                  <div className="mt-4 flex items-center gap-4 text-sm text-gray-500">
                    <span>View Audit →</span>
                    {audit.prNumber && (
                      <Link
                        href={`/textbook/audits/${audit.slug}?review=true`}
                        className="text-blue-600 hover:text-blue-800"
                        onClick={(e) => e.stopPropagation()}
                      >
                        Review Mode →
                      </Link>
                    )}
                  </div>
                </Link>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
