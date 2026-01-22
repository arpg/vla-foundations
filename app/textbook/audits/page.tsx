import Link from "next/link";
import fs from "fs";
import path from "path";
import matter from "gray-matter";

interface AuditMetadata {
  title: string;
  author: string;
  paper: string;
  topic: string;
  slug: string;
  isStaging?: boolean;
}

function getAllAudits(includeStaging: boolean = false): AuditMetadata[] {
  const auditsDir = path.join(process.cwd(), "content", "textbook", "audits");
  const audits: AuditMetadata[] = [];

  // Get production audits
  if (fs.existsSync(auditsDir)) {
    const files = fs.readdirSync(auditsDir).filter(file =>
      file.endsWith(".mdx") && file !== "README.mdx"
    );

    for (const file of files) {
      const filePath = path.join(auditsDir, file);
      const fileContents = fs.readFileSync(filePath, "utf8");
      const { data } = matter(fileContents);

      audits.push({
        title: data.title || file.replace(".mdx", ""),
        author: data.author || "Unknown",
        paper: data.paper || "Unknown Paper",
        topic: data.topic || "General",
        slug: file.replace(".mdx", ""),
        isStaging: false,
      });
    }
  }

  // Get staging audits (only in preview/development)
  if (includeStaging && process.env.VERCEL_ENV !== "production") {
    const stagingDir = path.join(auditsDir, "staging");
    if (fs.existsSync(stagingDir)) {
      const stagingFiles = fs.readdirSync(stagingDir).filter(file =>
        file.endsWith(".mdx") && file !== "README.mdx"
      );

      for (const file of stagingFiles) {
        const filePath = path.join(stagingDir, file);
        const fileContents = fs.readFileSync(filePath, "utf8");
        const { data } = matter(fileContents);

        audits.push({
          title: data.title || file.replace(".mdx", ""),
          author: data.author || "Unknown",
          paper: data.paper || "Unknown Paper",
          topic: data.topic || "General",
          slug: `staging/${file.replace(".mdx", "")}`,
          isStaging: true,
        });
      }
    }
  }

  return audits.sort((a, b) => a.author.localeCompare(b.author));
}

export default function AuditsPage() {
  const audits = getAllAudits(true);
  const productionAudits = audits.filter(a => !a.isStaging);
  const stagingAudits = audits.filter(a => a.isStaging);
  const isPreview = process.env.VERCEL_ENV !== "production";

  return (
    <div className="min-h-screen bg-white">
      <div className="max-w-6xl mx-auto px-6 py-16">
        <Link href="/course" className="text-sm text-blue-600 hover:text-blue-800 mb-8 inline-block">
          ← Back to Course
        </Link>

        <h1 className="text-4xl font-bold text-gray-900 mb-4">
          Student Paper Audits
        </h1>
        <p className="text-xl text-gray-600 mb-12">
          Technical deep-dives into VLA research papers by CSCI 7000 students.
        </p>

        {productionAudits.length === 0 && stagingAudits.length === 0 && (
          <div className="text-center py-12 text-gray-500">
            No audits have been published yet. Check back soon!
          </div>
        )}

        {/* Production Audits */}
        {productionAudits.length > 0 && (
          <section className="mb-12">
            <h2 className="text-2xl font-bold text-gray-900 mb-6">Published Audits</h2>
            <div className="grid gap-6 md:grid-cols-2">
              {productionAudits.map((audit) => (
                <Link
                  key={audit.slug}
                  href={`/textbook/audits/${audit.slug}`}
                  className="block border border-gray-200 rounded-lg p-6 hover:shadow-lg transition-shadow"
                >
                  <div className="flex items-start justify-between mb-3">
                    <span className="text-sm font-medium text-blue-600 bg-blue-50 px-3 py-1 rounded-full">
                      {audit.topic}
                    </span>
                  </div>
                  <h3 className="text-xl font-semibold text-gray-900 mb-2">
                    {audit.title}
                  </h3>
                  <p className="text-gray-600 mb-3">{audit.paper}</p>
                  <div className="text-sm text-gray-500">
                    By <span className="font-medium">{audit.author}</span>
                  </div>
                </Link>
              ))}
            </div>
          </section>
        )}

        {/* Staging Audits (Preview only) */}
        {isPreview && stagingAudits.length > 0 && (
          <section>
            <div className="bg-yellow-50 border-l-4 border-yellow-400 p-4 mb-6">
              <p className="text-sm text-yellow-800">
                <strong>Preview Mode:</strong> The following audits are in review and only visible in preview deployments.
              </p>
            </div>
            <h2 className="text-2xl font-bold text-gray-900 mb-6">Audits Under Review</h2>
            <div className="grid gap-6 md:grid-cols-2">
              {stagingAudits.map((audit) => (
                <Link
                  key={audit.slug}
                  href={`/textbook/audits/${audit.slug}`}
                  className="block border border-yellow-200 bg-yellow-50 rounded-lg p-6 hover:shadow-lg transition-shadow"
                >
                  <div className="flex items-start justify-between mb-3">
                    <span className="text-sm font-medium text-yellow-700 bg-yellow-100 px-3 py-1 rounded-full">
                      {audit.topic} • DRAFT
                    </span>
                  </div>
                  <h3 className="text-xl font-semibold text-gray-900 mb-2">
                    {audit.title}
                  </h3>
                  <p className="text-gray-600 mb-3">{audit.paper}</p>
                  <div className="text-sm text-gray-500">
                    By <span className="font-medium">{audit.author}</span>
                  </div>
                </Link>
              ))}
            </div>
          </section>
        )}
      </div>
    </div>
  );
}
