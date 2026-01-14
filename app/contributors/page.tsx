import Link from "next/link";
import fs from "fs";
import path from "path";
import matter from "gray-matter";

interface Contributor {
  name: string;
  role: string;
  semester: string;
  github: string;
  slug: string;
}

function getAllContributors(): Contributor[] {
  const contributorsDir = path.join(process.cwd(), "content", "contributors");

  if (!fs.existsSync(contributorsDir)) {
    return [];
  }

  const files = fs.readdirSync(contributorsDir)
    .filter(file => file.endsWith(".mdx") && !file.startsWith("_"));

  const contributors = files.map(file => {
    const filePath = path.join(contributorsDir, file);
    const fileContents = fs.readFileSync(filePath, "utf8");
    const { data } = matter(fileContents);

    return {
      name: data.name || "Unknown",
      role: data.role || "Student",
      semester: data.semester || "",
      github: data.github || "",
      slug: file.replace(".mdx", ""),
    };
  });

  // Sort: instructors first, then by name
  return contributors.sort((a, b) => {
    if (a.role === "Instructor" && b.role !== "Instructor") return -1;
    if (a.role !== "Instructor" && b.role === "Instructor") return 1;
    return a.name.localeCompare(b.name);
  });
}

export default function ContributorsPage() {
  const contributors = getAllContributors();

  return (
    <div className="min-h-screen bg-white">
      <div className="max-w-6xl mx-auto px-6 py-16">
        <Link href="/" className="text-sm text-blue-600 hover:text-blue-800 mb-8 inline-block">
          ← Back to Home
        </Link>

        <h1 className="text-4xl font-bold text-gray-900 mb-4">
          Contributors
        </h1>
        <p className="text-xl text-gray-600 mb-12">
          Students and instructors contributing to the VLA Foundations textbook.
        </p>

        {contributors.length === 0 ? (
          <div className="text-center py-12 border border-gray-200 rounded-lg">
            <p className="text-gray-600">No contributors yet. Be the first to submit!</p>
          </div>
        ) : (
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {contributors.map((contributor) => (
              <Link
                key={contributor.slug}
                href={`/contributors/${contributor.slug}`}
                className="group"
              >
                <div className="border border-gray-200 rounded-lg p-6 hover:shadow-md transition-shadow h-full">
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-lg font-semibold text-gray-900 group-hover:text-blue-600 transition-colors">
                      {contributor.name}
                    </h3>
                    {contributor.role === "Instructor" && (
                      <span className="text-xs font-medium text-blue-600 bg-blue-50 px-2 py-1 rounded">
                        Instructor
                      </span>
                    )}
                  </div>

                  <div className="space-y-2">
                    {contributor.semester && (
                      <p className="text-sm text-gray-600">
                        {contributor.semester}
                      </p>
                    )}
                    {contributor.github && (
                      <p className="text-sm text-gray-500">
                        @{contributor.github}
                      </p>
                    )}
                  </div>
                </div>
              </Link>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
