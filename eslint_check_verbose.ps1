# Change to vite-js directory
Set-Location "vite-js"

# Run ESLint to get actual errors with format=compact
npx eslint "src/**/*.{js,jsx,ts,tsx}" --format=compact
