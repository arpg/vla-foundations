import fs from 'fs';
import path from 'path';
import matter from 'gray-matter';

export interface ChapterMetadata {
  title: string;
  chapter: number;
  description: string;
  slug: string;
}

const contentDirectory = path.join(process.cwd(), 'content', 'textbook');

export function getAllChapters(): ChapterMetadata[] {
  const chapters: ChapterMetadata[] = [];

  // Get all directories in content/textbook/
  const dirs = fs.readdirSync(contentDirectory, { withFileTypes: true })
    .filter(dirent => dirent.isDirectory())
    .map(dirent => dirent.name);

  for (const dir of dirs) {
    const mdxPath = path.join(contentDirectory, dir, 'index.mdx');

    if (fs.existsSync(mdxPath)) {
      const fileContents = fs.readFileSync(mdxPath, 'utf8');
      const { data } = matter(fileContents);

      chapters.push({
        title: data.title || dir,
        chapter: data.chapter || 0,
        description: data.description || '',
        slug: dir,
      });
    }
  }

  // Sort by chapter number
  return chapters.sort((a, b) => a.chapter - b.chapter);
}

export function getChapterBySlug(slug: string): ChapterMetadata | null {
  const mdxPath = path.join(contentDirectory, slug, 'index.mdx');

  if (!fs.existsSync(mdxPath)) {
    return null;
  }

  const fileContents = fs.readFileSync(mdxPath, 'utf8');
  const { data } = matter(fileContents);

  return {
    title: data.title || slug,
    chapter: data.chapter || 0,
    description: data.description || '',
    slug,
  };
}
