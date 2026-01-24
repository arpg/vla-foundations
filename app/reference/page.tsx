import Link from "next/link";

export default function ReferencePage() {
  return (
    <div className="min-h-screen bg-gray-50">
      <header className="border-b border-gray-200 bg-white">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <Link href="/" className="text-blue-600 hover:text-blue-700 text-sm">
            ← Back to Home
          </Link>
        </div>
      </header>
      
      <main className="max-w-4xl mx-auto px-6 py-24">
        <h1 className="text-4xl font-bold text-gray-900 mb-6">
          Reference Implementations
        </h1>
        <p className="text-xl text-gray-600 mb-12">
          Practical implementations and validation frameworks for production VLA systems.
        </p>
        
        <div className="bg-white rounded-lg border border-gray-200 p-8">
          <p className="text-gray-600">
            This section will contain reference implementations demonstrating rigorous validation 
            and production-grade patterns for Vision-Language-Action systems.
          </p>
        </div>
      </main>
    </div>
  );
}
