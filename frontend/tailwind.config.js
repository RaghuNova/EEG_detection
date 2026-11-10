/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        background: {
          DEFAULT: '#080E1B',
          secondary: '#0D1628',
          panel: '#111D32',
        },
        border: {
          DEFAULT: '#23324B',
          subtle: '#1B283D',
        },
        cyan: {
          accent: '#36C5E8',
        },
        blue: {
          accent: '#4D8DFF',
        },
        purple: {
          accent: '#A78BFA',
        },
        emerald: {
          accent: '#34D399',
        },
        amber: {
          accent: '#FBBF24',
        },
        rose: {
          accent: '#FB7185',
        },
        text: {
          primary: '#F1F5F9',
          secondary: '#94A3B8',
          muted: '#64748B',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['JetBrains Mono', 'ui-monospace', 'monospace'],
      },
    },
  },
  plugins: [],
}
