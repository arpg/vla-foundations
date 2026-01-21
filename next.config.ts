import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: 'export', // Static export for Apache deployment
  trailingSlash: true, // Generate /course/index.html instead of /course.html
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
