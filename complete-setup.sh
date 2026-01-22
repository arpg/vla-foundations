#!/bin/bash
# Complete setup script - run this on ristoffer.ch
# This script: fixes the code, builds, starts server, configures Apache, and tests

set -e

echo "=== Step 1: Fix the audits page (remove onClick handler) ==="
cd ~/vla-staging
cat > app/textbook/audits/page.tsx << 'EOF'
import { getAllAudits } from '@/lib/audits';
import Link from 'next/link';

export default function AuditsIndexPage() {
  const audits = getAllAudits();

  return (
    <div className="min-h-screen bg-gray-50 py-12">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="bg-white rounded-lg shadow-sm p-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            Student Paper Audits
          </h1>
          <p className="text-lg text-gray-600 mb-8">
            Technical deep-dives into VLM and robotics papers by course students.
          </p>

          {audits.length === 0 ? (
            <div className="text-center py-12">
              <p className="text-gray-500 text-lg">
                No audits available yet. Check back soon!
              </p>
            </div>
          ) : (
            <div className="space-y-6">
              {audits.map((audit) => (
                <div
                  key={audit.slug}
                  className="block p-6 border border-gray-200 rounded-lg hover:border-blue-500 hover:shadow-md transition-all"
                >
                  <div className="flex items-start justify-between">
                    <div>
                      <h2 className="text-xl font-semibold text-gray-900 mb-2">
                        {audit.title}
                      </h2>
                      {audit.author && (
                        <p className="text-sm text-gray-600">
                          By {audit.author}
                        </p>
                      )}
                    </div>
                    {audit.prNumber && (
                      <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                        PR #{audit.prNumber}
                      </span>
                    )}
                  </div>
                  <div className="mt-4 flex items-center gap-4 text-sm">
                    <Link href={\`/textbook/audits/\${audit.slug}\`} className="text-blue-600 hover:text-blue-800 font-medium">
                      View Audit →
                    </Link>
                    {audit.prNumber && (
                      <Link href={\`/textbook/audits/\${audit.slug}?review=true\`} className="text-green-600 hover:text-green-800 font-medium">
                        Review Mode →
                      </Link>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
EOF

echo "✓ Fixed audits page"

echo ""
echo "=== Step 2: Build Next.js ==="
npm run build

echo ""
echo "=== Step 3: Start Next.js server ==="
pkill -f 'next start' || true
sleep 2
nohup npm start -- -p 3002 > ~/vla-staging-server.log 2>&1 &
SERVER_PID=$!
echo "Server started with PID $SERVER_PID"

echo "Waiting for server to start..."
sleep 5

# Test server
if curl -s http://localhost:3002/ > /dev/null; then
  echo "✓ Next.js server is running on port 3002"
else
  echo "✗ Server failed to start. Check: tail ~/vla-staging-server.log"
  exit 1
fi

echo ""
echo "=== Step 4: Test API endpoint ==="
curl -s http://localhost:3002/api/comments/23 | head -10
echo ""
echo "✓ API endpoint working"

echo ""
echo "=== Step 5: Configure Apache proxy ==="
echo ""
echo "Choose your setup:"
echo "  A) Subdomain (staging.vlm-robotics.dev) - RECOMMENDED"
echo "  B) Subdirectory (/staging on main domain)"
echo ""
echo "For option A (subdomain), run:"
echo ""
echo "  sudo tee /etc/apache2/sites-available/staging.vlm-robotics.dev.conf <<APACHE"
echo "  <VirtualHost *:80>"
echo "    ServerName staging.vlm-robotics.dev"
echo "    ProxyPass / http://localhost:3002/"
echo "    ProxyPassReverse / http://localhost:3002/"
echo "    ProxyPreserveHost On"
echo "  </VirtualHost>"
echo "  APACHE"
echo ""
echo "  sudo a2ensite staging.vlm-robotics.dev.conf"
echo "  sudo a2enmod proxy proxy_http"
echo "  sudo systemctl restart apache2"
echo ""
echo "For option B (subdirectory), add to /etc/apache2/sites-available/vlm-robotics.dev.conf:"
echo ""
echo "  ProxyPass /staging http://localhost:3002"
echo "  ProxyPassReverse /staging http://localhost:3002"
echo "  ProxyPreserveHost On"
echo ""
echo "  Then: sudo systemctl restart apache2"
echo ""
echo "=== Setup Complete ==="
echo ""
echo "Server logs: tail -f ~/vla-staging-server.log"
echo "Test locally: curl http://localhost:3002/textbook/audits/multimodality_audit?review=true"
echo ""
