import Link from "next/link";
import { notFound } from "next/navigation";
import fs from "fs";
import path from "path";
import { MDXRemote } from "next-mdx-remote/rsc";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";

interface PageProps {
  params: Promise<{ slug: string }>;
}

export async function generateStaticParams() {
  const assignmentsDir = path.join(process.cwd(), "content", "course", "assignments");

  if (!fs.existsSync(assignmentsDir)) {
    return [];
  }

  const files = fs.readdirSync(assignmentsDir)
    .filter(file => file.endsWith(".mdx"));

  return files.map(file => ({
    slug: file.replace(".mdx", ""),
  }));
}

export async function generateMetadata({ params }: PageProps) {
  const { slug } = await params;
  const filePath = path.join(process.cwd(), "content", "course", "assignments", `${slug}.mdx`);

  if (!fs.existsSync(filePath)) {
    return { title: "Assignment Not Found" };
  }

  const matter = await import("gray-matter");
  const fileContents = fs.readFileSync(filePath, "utf8");
  const { data } = matter.default(fileContents);

  return {
    title: `${data.title || slug} - VLA Foundations`,
    description: data.description || `Assignment: ${data.title || slug}`,
  };
}

export default async function AssignmentPage({ params }: PageProps) {
  const { slug } = await params;
  const filePath = path.join(process.cwd(), "content", "course", "assignments", `${slug}.mdx`);

  if (!fs.existsSync(filePath)) {
    notFound();
  }

  const matter = await import("gray-matter");
  const fileContents = fs.readFileSync(filePath, "utf8");
  const { data, content } = matter.default(fileContents);

  return (
    <div className="min-h-screen bg-white">
      <div className="max-w-4xl mx-auto px-6 py-16">
        <Link href="/course" className="text-sm text-blue-600 hover:text-blue-800 mb-8 inline-block">
          ← Back to Course
        </Link>

        {/* Assignment Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            {data.title || slug}
          </h1>

          <div className="flex items-center gap-4 text-sm text-gray-600">
            {data.due && (
              <span className="flex items-center gap-2">
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                </svg>
                <span>Due: {data.due}</span>
              </span>
            )}
            {data.points && (
              <span className="px-3 py-1 bg-blue-50 text-blue-600 rounded-full font-medium">
                {data.points} points
              </span>
            )}
          </div>
        </div>

        {/* MDX Content with KaTeX support */}
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

        {/* Back to course link at bottom */}
        <div className="mt-12 pt-8 border-t border-gray-200">
          <Link href="/course" className="text-blue-600 hover:text-blue-800 font-medium">
            ← Back to Course
          </Link>
        </div>
      </div>
    </div>
  );
}
