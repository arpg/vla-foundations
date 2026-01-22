import type { NextConfig } from "next";

// Check if building for PR preview on GitHub Pages
const isPRPreview = process.env.STAGING_PR_NUMBER !== undefined;
const prNumber = process.env.STAGING_PR_NUMBER;
const basePath = isPRPreview ? `/staging/pulls/${prNumber}` : '';

const nextConfig: NextConfig = {
  output: 'export', // Static export for Apache deployment
  trailingSlash: true, // Generate /course/index.html instead of /course.html
  basePath: basePath, // Set basePath for PR previews
  assetPrefix: basePath, // Set assetPrefix for PR previews
  images: {
    unoptimized: true, // Required for static export
  },
  turbopack: {}, // Silence Turbopack warning

  // Development proxy for API server
  async rewrites() {
    return {
      beforeFiles: [
        {
          source: '/api/:path*',
          destination: 'http://localhost:3001/api/:path*',
        },
      ],
    };
  },
};

export default nextConfig;
