/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  output: "export",
  trailingSlash: true,
  poweredByHeader: false,
  images: { unoptimized: true },
};

export default nextConfig;
