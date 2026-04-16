import type { Config } from "tailwindcss";

export default {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./content/**/*.{md,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "var(--background)",
        foreground: "var(--foreground)",
      },
      typography: {
        DEFAULT: {
          css: {
            maxWidth: 'none',
            color: '#1e293b',
            lineHeight: '1.75',
            fontSize: '1.125rem',
            a: {
              color: '#2563eb',
              textDecoration: 'underline',
              textDecorationColor: '#93c5fd',
              '&:hover': {
                color: '#1d4ed8',
                textDecorationColor: '#2563eb',
              },
            },
            'h1, h2, h3, h4': {
              fontWeight: '700',
              letterSpacing: '-0.025em',
            },
            p: {
              marginTop: '1.25rem',
              marginBottom: '1.25rem',
            },
            // Ensure math displays properly
            '.katex-display': {
              margin: '1.5rem 0',
              padding: '1.5rem 0',
            },
            code: {
              backgroundColor: '#f1f5f9',
              color: '#1e293b',
              padding: '0.25rem 0.375rem',
              borderRadius: '0.25rem',
              fontWeight: '500',
            },
            'code::before': {
              content: '""',
            },
            'code::after': {
              content: '""',
            },
          },
        },
        lg: {
          css: {
            fontSize: '1.125rem',
            lineHeight: '1.875',
            p: {
              marginTop: '1.5rem',
              marginBottom: '1.5rem',
            },
            h1: {
              fontSize: '2.5rem',
              marginTop: '0',
              marginBottom: '1.5rem',
            },
            h2: {
              fontSize: '2rem',
              marginTop: '3rem',
              marginBottom: '1.25rem',
            },
            h3: {
              fontSize: '1.5rem',
              marginTop: '2.5rem',
              marginBottom: '1rem',
            },
          },
        },
      },
    },
  },
  plugins: [
    require('@tailwindcss/typography'),
  ],
} satisfies Config;
