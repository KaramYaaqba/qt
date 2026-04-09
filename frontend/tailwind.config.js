/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        arabic: ['Scheherazade New', 'Amiri', 'serif'],
      },
      colors: {
        correct: '#22c55e',
        error: '#ef4444',
        partial: '#f97316',
        primary: '#3b82f6',
      },
    },
  },
  plugins: [],
}
