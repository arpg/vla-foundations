import { AuditLayout } from '@/components/audit/AuditLayout';
import { getAllAudits, getAuditBySlug, getAuditContent } from '@/lib/audits';
import { notFound } from 'next/navigation';
import { MDXRemote } from 'next-mdx-remote/rsc';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import remarkGfm from 'remark-gfm';

interface PageProps {
  params: Promise<{ slug: string }>;
  searchParams: Promise<{ review?: string }>;
}

export async function generateStaticParams() {
  const audits = getAllAudits();
  return audits.map((audit) => ({
    slug: audit.slug,
  }));
}

export async function generateMetadata({ params }: PageProps) {
  const { slug } = await params;
  const audit = getAuditBySlug(slug);

  if (!audit) {
    return {
      title: 'Audit Not Found',
    };
  }

  return {
    title: `${audit.title} - Paper Audit`,
    description: `Technical paper audit by ${audit.author || 'Student'}`,
  };
}

export default async function AuditPage({ params, searchParams }: PageProps) {
  const { slug } = await params;
  const { review } = await searchParams;
  const audit = getAuditBySlug(slug);

  if (!audit) {
    notFound();
  }

  const content = getAuditContent(slug);

  if (!content) {
    notFound();
  }

  // Check if we're in review mode (from query param)
  const isReviewMode = review === 'true';

  return (
    <AuditLayout
      prNumber={audit.prNumber}
      slug={slug}
      isReviewMode={isReviewMode}
    >
      <MDXRemote
        source={content}
        options={{
          parseFrontmatter: true,
          mdxOptions: {
            remarkPlugins: [remarkMath, remarkGfm],
            rehypePlugins: [rehypeKatex],
          },
        }}
      />
    </AuditLayout>
  );
}
