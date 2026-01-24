import fs from 'fs';
import path from 'path';
import matter from 'gray-matter';

export interface AuditMetadata {
  title: string;
  author?: string;
  prNumber?: number;
  slug: string;
  filePath: string;
}

const auditsDirectory = path.join(process.cwd(), 'content', 'textbook', 'audits');

export function getAllAudits(): AuditMetadata[] {
  const audits: AuditMetadata[] = [];

  // Check if audits directory exists
  if (!fs.existsSync(auditsDirectory)) {
    return [];
  }

  // Get all .mdx files in audits directory
  const files = fs.readdirSync(auditsDirectory, { withFileTypes: true })
    .filter(dirent => dirent.isFile() && dirent.name.endsWith('.mdx'))
    .map(dirent => dirent.name);

  for (const file of files) {
    const mdxPath = path.join(auditsDirectory, file);
    const fileContents = fs.readFileSync(mdxPath, 'utf8');
    const { data } = matter(fileContents);
    const slug = file.replace(/\.mdx$/, '');

    audits.push({
      title: data.title || slug,
      author: data.author,
      prNumber: data.prNumber,
      slug,
      filePath: mdxPath,
    });
  }

  return audits.sort((a, b) => a.title.localeCompare(b.title));
}

export function getAuditBySlug(slug: string): AuditMetadata | null {
  const mdxPath = path.join(auditsDirectory, `${slug}.mdx`);

  if (!fs.existsSync(mdxPath)) {
    return null;
  }

  const fileContents = fs.readFileSync(mdxPath, 'utf8');
  const { data } = matter(fileContents);

  return {
    title: data.title || slug,
    author: data.author,
    prNumber: data.prNumber,
    slug,
    filePath: mdxPath,
  };
}

export function getAuditContent(slug: string): string | null {
  const mdxPath = path.join(auditsDirectory, `${slug}.mdx`);

  if (!fs.existsSync(mdxPath)) {
    return null;
  }

  return fs.readFileSync(mdxPath, 'utf8');
}
