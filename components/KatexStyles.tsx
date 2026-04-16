'use client';

import { useEffect } from 'react';

export function KatexStyles() {
  useEffect(() => {
    // Ensure KaTeX CSS is loaded
    const link = document.createElement('link');
    link.rel = 'stylesheet';
    link.href = 'https://cdn.jsdelivr.net/npm/katex@0.16.27/dist/katex.min.css';
    link.integrity = 'sha384-yp+jpRNKIa0xGrYaVtwImDXkFq7ZOCV5kJZVDg/uAFfYPmtFcKr0sxhVJy1HqnWD';
    link.crossOrigin = 'anonymous';

    // Check if already loaded
    const existing = document.querySelector('link[href*="katex"]');
    if (!existing) {
      document.head.appendChild(link);
    }
  }, []);

  return null;
}
