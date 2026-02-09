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
              color: '#059669',
              textDecoration: 'underline',
              textDecorationColor: '#6ee7b7',
              textUnderlineOffset: '3px',
              fontWeight: '500',
              '&:hover': {
                color: '#047857',
                textDecorationColor: '#10b981',
              },
            },
            h1: {
              color: '#0f172a',
              fontWeight: '800',
              letterSpacing: '-0.025em',
            },
            h2: {
              color: '#1e293b',
              fontWeight: '700',
              letterSpacing: '-0.025em',
            },
            h3: {
              color: '#334155',
              fontWeight: '600',
              letterSpacing: '-0.025em',
            },
            h4: {
              color: '#475569',
              fontWeight: '600',
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
              color: '#059669',
              backgroundColor: '#f0fdfa',
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
            pre: {
              backgroundColor: '#0f172a',
              color: '#e2e8f0',
            },
            blockquote: {
              borderLeftColor: '#10b981',
              color: '#475569',
            },
            strong: {
              color: '#0f172a',
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
