import { AuditLayout } from '@/components/audit/AuditLayout';
import { getAllAudits, getAuditBySlug, getAuditContent } from '@/lib/audits';
import { getAllChapters } from '@/lib/chapters';
import { notFound } from 'next/navigation';
import { MDXRemote } from 'next-mdx-remote/rsc';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import remarkGfm from 'remark-gfm';

interface PageProps {
  params: Promise<{ slug: string }>;
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

export default async function AuditPage({ params }: PageProps) {
  const { slug } = await params;
  const audit = getAuditBySlug(slug);
  const chapters = getAllChapters();

  if (!audit) {
    notFound();
  }

  const content = getAuditContent(slug);

  if (!content) {
    notFound();
  }

  return (
    <AuditLayout chapters={chapters}>
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
