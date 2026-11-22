/** @type {import('next').NextConfig} */

const courseApiOrigin = (process.env.NEXT_PUBLIC_API_URL || "")
  .trim()
  .replace(/\/+$/, "");

const nextConfig = {
  eslint: {
    ignoreDuringBuilds: true,
  },
  async headers() {
    const connectSources = [
      "'self'",
      courseApiOrigin,
      "https://api.themoviedb.org",
      "http://localhost:*",
    ]
      .filter(Boolean)
      .join(" ");

    return [
      {
        source: '/(.*)',
        headers: [
          {
            key: 'X-Content-Type-Options',
            value: 'nosniff',
          },
          {
            key: 'X-Frame-Options',
            value: 'DENY',
          },
          {
            key: 'X-XSS-Protection',
            value: '1; mode=block',
          },
          {
            key: 'Referrer-Policy',
            value: 'strict-origin-when-cross-origin',
          },
          {
            key: 'Permissions-Policy',
            value: 'camera=(), microphone=(), geolocation=()',
          },
          {
            key: 'Content-Security-Policy',
            value: [
              "default-src 'self'",
              "script-src 'self' 'unsafe-eval' 'unsafe-inline'",
              "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
              "font-src 'self' https://fonts.gstatic.com",
              "img-src 'self' data: blob: https://image.tmdb.org https://api.themoviedb.org https://*.ltrbxd.com https://a.ltrbxd.com https://*.letterboxd.com",
              `connect-src ${connectSources}`,
              "frame-ancestors 'none'",
              "base-uri 'self'",
              "form-action 'self'",
            ].join('; '),
          },
        ],
      },
    ];
  },
};

export default nextConfig;
