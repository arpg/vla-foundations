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

  // Determine if we're in review mode based on environment
  const isReviewMode = isStaging && process.env.STAGING_PR_NUMBER !== undefined;
  const prNumber = process.env.STAGING_PR_NUMBER;

  return (
    <AuditLayout
      chapters={chapters}
      isReviewMode={isReviewMode}
      prNumber={prNumber}
    >
      <Link href="/textbook/audits" className="text-sm text-blue-600 hover:text-blue-800 mb-8 inline-block">
        ← Back to Audits
      </Link>

      {/* Staging Banner (shown when in staging but not in review mode) */}
      {isStaging && !isReviewMode && (
        <div className="bg-gradient-to-r from-yellow-50 to-orange-50 border-l-4 border-yellow-500 text-yellow-900 px-6 py-5 mb-8 rounded-lg shadow-sm">
          <p className="font-bold text-lg flex items-center gap-2">
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
            DRAFT AUDIT - UNDER REVIEW
          </p>
          <p className="mt-2 text-sm leading-relaxed">
            This is a preview of a student audit currently under review. Content may change before final publication.
          </p>
        </div>
      )}

      {/* Audit Header */}
      <div className="mb-12 pb-8 border-b-2 border-slate-200">
        <div className="flex items-center gap-3 mb-6">
          {data.topic && (
            <span className="text-sm font-semibold text-blue-700 bg-blue-50 px-4 py-1.5 rounded-full border border-blue-200">
              {data.topic}
            </span>
          )}
          {isStaging && (
            <span className="text-sm font-semibold text-yellow-700 bg-yellow-100 px-4 py-1.5 rounded-full border border-yellow-300">
              DRAFT
            </span>
          )}
        </div>

        <h1 className="text-5xl font-extrabold text-slate-900 mb-6 leading-tight tracking-tight">
          {data.title || "Paper Audit"}
        </h1>

        {data.paper && (
          <p className="text-xl text-slate-600 mb-5 font-light leading-relaxed">{data.paper}</p>
        )}

        {data.author && (
          <p className="text-base text-slate-700 flex items-center gap-2">
            <svg className="w-5 h-5 text-slate-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
            </svg>
            <span>By <span className="font-semibold">{data.author}</span></span>
          </p>
        )}
      </div>

      {/* MDX Content with KaTeX support */}
      <MDXRemote
        source={content}
        options={{
          mdxOptions: {
            remarkPlugins: [remarkMath, remarkGfm],
            rehypePlugins: [
              [rehypeKatex, {
                strict: false, // Don't fail on unknown LaTeX commands
                trust: true, // Allow some advanced LaTeX features
                throwOnError: false, // Gracefully handle errors
              }]
            ],
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
