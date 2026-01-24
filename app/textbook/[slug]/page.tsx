import { TextbookLayout } from '@/components/textbook/TextbookLayout';
import { getAllChapters, getChapterBySlug } from '@/lib/chapters';
import { notFound } from 'next/navigation';
import fs from 'fs';
import path from 'path';
import { MDXRemote } from 'next-mdx-remote/rsc';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import remarkGfm from 'remark-gfm';

interface PageProps {
  params: Promise<{ slug: string }>;
}

export async function generateStaticParams() {
  const chapters = getAllChapters();
  return chapters.map((chapter) => ({
    slug: chapter.slug,
  }));
}

export async function generateMetadata({ params }: PageProps) {
  const { slug } = await params;
  const chapter = getChapterBySlug(slug);
  
  if (!chapter) {
    return {
      title: 'Chapter Not Found',
    };
  }
  
  return {
    title: `${chapter.title} - VLA Stack`,
    description: chapter.description,
  };
}

export default async function ChapterPage({ params }: PageProps) {
  const { slug } = await params;
  const chapter = getChapterBySlug(slug);
  
  if (!chapter) {
    notFound();
  }
  
  // Read the MDX file
  const contentPath = path.join(process.cwd(), 'content', 'textbook', slug, 'index.mdx');
  
  if (!fs.existsSync(contentPath)) {
    notFound();
  }
  
  const source = fs.readFileSync(contentPath, 'utf8');
  
  return (
    <TextbookLayout>
      <MDXRemote 
        source={source}
        options={{
          parseFrontmatter: true,
          mdxOptions: {
            remarkPlugins: [remarkMath, remarkGfm],
            rehypePlugins: [rehypeKatex],
          },
        }}
      />
    </TextbookLayout>
  );
}
