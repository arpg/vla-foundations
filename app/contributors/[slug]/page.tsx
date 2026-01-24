import Link from "next/link";
import { notFound } from "next/navigation";
import fs from "fs";
import path from "path";
import { MDXRemote } from "next-mdx-remote/rsc";
import remarkGfm from "remark-gfm";

interface PageProps {
  params: Promise<{ slug: string }>;
}

export async function generateStaticParams() {
  const contributorsDir = path.join(process.cwd(), "content", "contributors");

  if (!fs.existsSync(contributorsDir)) {
    return [];
  }

  const files = fs.readdirSync(contributorsDir)
    .filter(file => file.endsWith(".mdx") && !file.startsWith("_"));

  return files.map(file => ({
    slug: file.replace(".mdx", ""),
  }));
}

export async function generateMetadata({ params }: PageProps) {
  const { slug } = await params;
  const filePath = path.join(process.cwd(), "content", "contributors", `${slug}.mdx`);

  if (!fs.existsSync(filePath)) {
    return { title: "Contributor Not Found" };
  }

  const matter = await import("gray-matter");
  const fileContents = fs.readFileSync(filePath, "utf8");
  const { data } = matter.default(fileContents);

  return {
    title: `${data.name || slug} - VLA Foundations`,
    description: `Contributor profile for ${data.name || slug}`,
  };
}

export default async function ContributorProfilePage({ params }: PageProps) {
  const { slug } = await params;
  const filePath = path.join(process.cwd(), "content", "contributors", `${slug}.mdx`);

  if (!fs.existsSync(filePath)) {
    notFound();
  }

  const matter = await import("gray-matter");
  const fileContents = fs.readFileSync(filePath, "utf8");
  const { data, content } = matter.default(fileContents);

  return (
    <div className="min-h-screen bg-white">
      <div className="max-w-4xl mx-auto px-6 py-16">
        <Link href="/contributors" className="text-sm text-blue-600 hover:text-blue-800 mb-8 inline-block">
          ← Back to Contributors
        </Link>

        {/* Profile Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            {data.name || slug}
          </h1>
          <div className="flex items-center gap-4 text-gray-600">
            {data.role && (
              <span className="text-sm font-medium text-blue-600 bg-blue-50 px-3 py-1 rounded-full">
                {data.role}
              </span>
            )}
            {data.semester && (
              <span className="text-sm">{data.semester}</span>
            )}
          </div>
        </div>

        {/* GitHub Link */}
        {data.github && (
          <div className="mb-8">
            <a
              href={`https://github.com/${data.github}`}
              target="_blank"
              rel="noopener noreferrer"
              className="text-blue-600 hover:text-blue-800"
            >
              @{data.github} ↗
            </a>
          </div>
        )}

        {/* MDX Content */}
        <div className="prose prose-lg max-w-none">
          <MDXRemote
            source={content}
            options={{
              mdxOptions: {
                remarkPlugins: [remarkGfm],
              },
            }}
          />
        </div>
      </div>
    </div>
  );
}
