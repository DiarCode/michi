/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        'michi-lime': '#B1E743',
        'michi-lime-light': '#D4F080',
        'michi-lime-dark': '#8BC02A',
        'michi-dark': '#22201F',
        'michi-page': '#F0F4E8',
        'michi-warm': '#FBFCF7',
        'michi-muted': '#9C9C95',
        'michi-body': '#555550',
        'michi-border': '#E8E8E0',
        'michi-amber': '#F5A623',
        'michi-red': '#E54D4D',
        'michi-teal': '#2ABFBF',
        'michi-purple': '#8B5CF6',
      },
      fontFamily: {
        geologica: ['Geologica', 'system-ui', '-apple-system', 'sans-serif'],
      },
      borderRadius: {
        '2xl': '1rem',
        '3xl': '1.5rem',
      },
      boxShadow: {
        'card': '0 1px 3px rgba(0,0,0,0.04)',
        'card-hover': '0 4px 12px rgba(0,0,0,0.06)',
        'tooltip': '0 4px 14px rgba(0,0,0,0.15)',
      },
      fontSize: {
        'kpi': ['2.5rem', { lineHeight: '1.1', letterSpacing: '-0.02em' }],
      },
    },
  },
  plugins: [],
};