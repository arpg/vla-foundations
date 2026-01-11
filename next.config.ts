import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: 'export', // Static export for Nginx deployment
  images: {
    unoptimized: true, // Required for static export
  },
  turbopack: {}, // Silence Turbopack warning
};

export default nextConfig;
