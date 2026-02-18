import Link from "next/link";
import { notFound } from "next/navigation";
import fs from "fs";
import path from "path";
import { MDXRemote } from "next-mdx-remote/rsc";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";

interface PageProps {
  params: Promise<{ handle: string }>;
}

export async function generateStaticParams() {
  const capstoneDir = path.join(process.cwd(), "content", "course", "assignments", "capstone");

  if (!fs.existsSync(capstoneDir)) {
    return [];
  }

  const handles = fs.readdirSync(capstoneDir, { withFileTypes: true })
    .filter(entry => entry.isDirectory())
    .filter(entry => fs.existsSync(path.join(capstoneDir, entry.name, "Report.mdx")))
    .map(entry => ({ handle: entry.name }));

  return handles;
}

export async function generateMetadata({ params }: PageProps) {
  const { handle } = await params;
  const filePath = path.join(process.cwd(), "content", "course", "assignments", "capstone", handle, "Report.mdx");

  if (!fs.existsSync(filePath)) {
    return { title: "Project Not Found" };
  }

  const matter = await import("gray-matter");
  const fileContents = fs.readFileSync(filePath, "utf8");
  const { data } = matter.default(fileContents);

  return {
    title: `${data.title || handle} - VLA Capstone`,
    description: `Capstone project report for @${handle}`,
  };
}

export default async function CapstoneReportPage({ params }: PageProps) {
  const { handle } = await params;
  const filePath = path.join(process.cwd(), "content", "course", "assignments", "capstone", handle, "Report.mdx");

  if (!fs.existsSync(filePath)) {
    notFound();
  }

  const matter = await import("gray-matter");
  const fileContents = fs.readFileSync(filePath, "utf8");
  const { data, content } = matter.default(fileContents);

  const groupLabel = data.group ? `Group ${data.group}` : null;
  const labLabel = data.lab || null;

  return (
    <div className="min-h-screen bg-white">
      <div className="max-w-4xl mx-auto px-6 py-16">
        <Link href="/course/assignments/capstone" className="text-sm text-blue-600 hover:text-blue-800 mb-8 inline-block">
          ← Back to Capstone
        </Link>

        {/* Report Header */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-4">
            {groupLabel && (
              <span className="text-sm font-medium text-blue-600 bg-blue-50 px-3 py-1 rounded-full border border-blue-200">
                {groupLabel}
              </span>
            )}
            {labLabel && (
              <span className="text-sm font-medium text-slate-600 bg-slate-100 px-3 py-1 rounded-full">
                {labLabel}
              </span>
            )}
            <span className="text-sm font-medium text-amber-600 bg-amber-50 px-3 py-1 rounded-full border border-amber-200">
              Finals Week
            </span>
          </div>

          <div className="flex items-center gap-3">
            <img
              src={`https://github.com/${handle}.png?size=64`}
              alt={handle}
              width={48}
              height={48}
              className="rounded-full"
            />
            <div>
              <h1 className="text-3xl font-bold text-gray-900">
                {data.title || `@${handle} — Capstone Report`}
              </h1>
              <a
                href={`https://github.com/${handle}`}
                className="text-sm text-slate-500 hover:text-slate-700"
                target="_blank"
                rel="noopener noreferrer"
              >
                @{handle}
              </a>
            </div>
          </div>
        </div>

        {/* MDX Content */}
        <div className="prose prose-lg max-w-none">
          <MDXRemote
            source={content}
            options={{
              mdxOptions: {
                remarkPlugins: [remarkMath, remarkGfm],
                rehypePlugins: [rehypeKatex],
              },
            }}
          />
        </div>

        <div className="mt-12 pt-8 border-t border-gray-200">
          <Link href="/course/assignments/capstone" className="text-blue-600 hover:text-blue-800 font-medium">
            ← Back to Capstone
          </Link>
        </div>
      </div>
    </div>
  );
}
