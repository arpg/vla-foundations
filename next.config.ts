import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: 'export', // Static export for Apache deployment
  trailingSlash: true, // Generate /course/index.html instead of /course.html
  images: {
    unoptimized: true, // Required for static export
  },
  turbopack: {}, // Silence Turbopack warning
};

export default nextConfig;
