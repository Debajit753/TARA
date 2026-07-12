#!/bin/sh
# Rebuild each page's stylesheet (every page owns its own CSS file).
# Shared parts live in the _*.css partials; run this after any style change.
cd "$(dirname "$0")"
npx -y tailwindcss@3.4.17 -c tailwind.config.js -i dash-workspace.in.css -o dash-workspace.css --minify
npx -y tailwindcss@3.4.17 -c tailwind.config.js -i about.in.css -o about.css --minify
echo ""
echo "frontend size (budget: 25 MB):"
du -sh .
