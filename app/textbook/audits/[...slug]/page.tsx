import Link from "next/link";
import { notFound } from "next/navigation";
import fs from "fs";
import path from "path";
import { MDXRemote } from "next-mdx-remote/rsc";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import matter from "gray-matter";
import { AuditLayout } from "@/components/audit/AuditLayout";
import { getAllChapters } from "@/lib/chapters";

interface PageProps {
  params: Promise<{ slug: string[] }>;
}

export async function generateStaticParams() {
  const auditsDir = path.join(process.cwd(), "content", "textbook", "audits");
  const params: { slug: string[] }[] = [];

  // Generate params for production audits
  if (fs.existsSync(auditsDir)) {
    const files = fs.readdirSync(auditsDir)
      .filter(file => file.endsWith(".mdx") && file !== "README.mdx");

    params.push(...files.map(file => ({
      slug: [file.replace(".mdx", "")],
    })));
  }

  // Generate params for staging audits (only in preview/dev)
  if (process.env.VERCEL_ENV !== "production" && process.env.STAGING_PR_NUMBER !== undefined) {
    const stagingDir = path.join(auditsDir, "staging");
    if (fs.existsSync(stagingDir)) {
      const stagingFiles = fs.readdirSync(stagingDir)
        .filter(file => file.endsWith(".mdx") && file !== "README.mdx");

      params.push(...stagingFiles.map(file => ({
        slug: ["staging", file.replace(".mdx", "")],
      })));
    }
  }

  return params;
}

export async function generateMetadata({ params }: PageProps) {
  const { slug } = await params;
  const auditsDir = path.join(process.cwd(), "content", "textbook", "audits");

  // Check if it's a staging audit
  const isStaging = slug[0] === "staging";
  const fileName = isStaging ? slug[1] : slug[0];
  const filePath = isStaging
    ? path.join(auditsDir, "staging", `${fileName}.mdx`)
    : path.join(auditsDir, `${fileName}.mdx`);

  if (!fs.existsSync(filePath)) {
    return { title: "Audit Not Found" };
  }

  const fileContents = fs.readFileSync(filePath, "utf8");
  const { data } = matter(fileContents);

  return {
    title: `${data.title || "Paper Audit"} - VLA Foundations`,
    description: data.paper || "Paper audit by CSCI 7000 student",
  };
}

export default async function AuditPage({ params }: PageProps) {
  const { slug } = await params;
  const auditsDir = path.join(process.cwd(), "content", "textbook", "audits");

  // Check if it's a staging audit
  const isStaging = slug[0] === "staging";
  const fileName = isStaging ? slug[1] : slug[0];
  const filePath = isStaging
    ? path.join(auditsDir, "staging", `${fileName}.mdx`)
    : path.join(auditsDir, `${fileName}.mdx`);

  if (!fs.existsSync(filePath)) {
    notFound();
  }

  const fileContents = fs.readFileSync(filePath, "utf8");
  const { data, content } = matter(fileContents);

  // Get chapters for sidebar
  const chapters = getAllChapters();

  return (
    <AuditLayout chapters={chapters}>
      <Link href="/textbook/audits" className="text-sm text-blue-600 hover:text-blue-800 mb-8 inline-block">
        ← Back to Audits
      </Link>

      {/* Staging Banner */}
      {isStaging && (
        <div className="bg-yellow-50 border-l-4 border-yellow-400 text-yellow-800 px-6 py-4 mb-8 rounded">
          <p className="font-bold text-lg">⚠️ DRAFT AUDIT - UNDER REVIEW</p>
          <p className="mt-2 text-sm">
            This is a preview of a student audit currently under review. Content may change before final publication.
          </p>
        </div>
      )}

      {/* Audit Header */}
      <div className="mb-8">
        <div className="flex items-center gap-3 mb-4">
          {data.topic && (
            <span className="text-sm font-medium text-blue-600 bg-blue-50 px-3 py-1 rounded-full">
              {data.topic}
            </span>
          )}
          {isStaging && (
            <span className="text-sm font-medium text-yellow-700 bg-yellow-100 px-3 py-1 rounded-full">
              DRAFT
            </span>
          )}
        </div>

        <h1 className="text-4xl font-bold text-gray-900 mb-4">
          {data.title || "Paper Audit"}
        </h1>

        {data.paper && (
          <p className="text-xl text-gray-600 mb-4">{data.paper}</p>
        )}

        {data.author && (
          <p className="text-gray-700">
            By <span className="font-medium">{data.author}</span>
          </p>
        )}
      </div>

      {/* MDX Content with KaTeX support */}
      <MDXRemote
        source={content}
        options={{
          mdxOptions: {
            remarkPlugins: [remarkMath, remarkGfm],
            rehypePlugins: [rehypeKatex],
          },
        }}
      />

      {/* Back to audits link at bottom */}
      <div className="mt-12 pt-8 border-t border-gray-200">
        <Link href="/textbook/audits" className="text-blue-600 hover:text-blue-800 font-medium">
          ← Back to All Audits
        </Link>
      </div>
    </AuditLayout>
  );
}
