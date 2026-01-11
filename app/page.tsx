import Link from "next/link";

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-50 to-white">
      {/* Header */}
      <header className="border-b border-gray-200 bg-white/80 backdrop-blur">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div>
            <h1 className="text-xl font-bold text-gray-900">VLA Stack</h1>
            <p className="text-xs text-gray-600">Vision-Language-Action for Robotics</p>
          </div>
          <nav className="flex gap-6">
            <Link href="/textbook/introduction" className="text-sm text-gray-700 hover:text-blue-600">
              Textbook
            </Link>
            <Link href="/reference" className="text-sm text-gray-700 hover:text-blue-600">
              Reference
            </Link>
          </nav>
        </div>
      </header>

      {/* Hero Section */}
      <section className="max-w-7xl mx-auto px-6 py-24">
        <div className="text-center mb-16">
          <h2 className="text-5xl font-bold text-gray-900 mb-6">
            The Vision-Language-Action Stack
          </h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto mb-8">
            A living reference of foundational architectures, rigorous validation strategies, 
            and deploying robot foundation models.
          </p>
          <div className="flex gap-4 justify-center">
            <Link 
              href="/textbook/introduction"
              className="px-6 py-3 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 transition-colors"
            >
              Read the Living Textbook
            </Link>
            <Link 
              href="/reference"
              className="px-6 py-3 border border-gray-300 text-gray-700 rounded-lg font-medium hover:bg-gray-50 transition-colors"
            >
              View Reference Implementations
            </Link>
          </div>
        </div>

        {/* VLA Control Loop Visualization */}
        <div className="max-w-4xl mx-auto bg-white rounded-2xl shadow-lg p-12 border border-gray-200">
          <h3 className="text-2xl font-bold text-gray-900 mb-8 text-center">
            The VLA Control Loop
          </h3>
          
          <div className="flex items-center justify-between gap-8">
            {/* Perception */}
            <div className="flex-1 text-center">
              <div className="w-24 h-24 mx-auto bg-gradient-to-br from-blue-500 to-blue-600 rounded-xl flex items-center justify-center mb-4 shadow-md">
                <svg className="w-12 h-12 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                </svg>
              </div>
              <div className="font-mono text-sm text-gray-600 mb-2">s ∈ S</div>
              <h4 className="font-semibold text-gray-900">Perception</h4>
              <p className="text-sm text-gray-600 mt-2">Scene Encoding</p>
            </div>

            {/* Arrow */}
            <div className="text-gray-400">
              <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
              </svg>
            </div>

            {/* VLM Backbone */}
            <div className="flex-1 text-center">
              <div className="w-24 h-24 mx-auto bg-gradient-to-br from-purple-500 to-purple-600 rounded-xl flex items-center justify-center mb-4 shadow-md">
                <svg className="w-12 h-12 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                </svg>
              </div>
              <div className="font-mono text-sm text-gray-600 mb-2">P(a|s,l)</div>
              <h4 className="font-semibold text-gray-900">VLM Backbone</h4>
              <p className="text-sm text-gray-600 mt-2">Reasoning</p>
            </div>

            {/* Arrow */}
            <div className="text-gray-400">
              <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
              </svg>
            </div>

            {/* Action */}
            <div className="flex-1 text-center">
              <div className="w-24 h-24 mx-auto bg-gradient-to-br from-green-500 to-green-600 rounded-xl flex items-center justify-center mb-4 shadow-md">
                <svg className="w-12 h-12 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
              </div>
              <div className="font-mono text-sm text-gray-600 mb-2">a ∈ A</div>
              <h4 className="font-semibold text-gray-900">Action</h4>
              <p className="text-sm text-gray-600 mt-2">Control Policies</p>
            </div>
          </div>

          {/* Formula */}
          <div className="mt-12 pt-8 border-t border-gray-200 text-center">
            <div className="font-mono text-lg text-gray-900">
              a* = arg max<sub>a</sub> P(a | s, l)
            </div>
          </div>
        </div>
      </section>

      {/* Textbook Structure */}
      <section className="max-w-7xl mx-auto px-6 py-24 bg-white">
        <h3 className="text-3xl font-bold text-gray-900 mb-12 text-center">
          Foundational Pillars
        </h3>
        
        <div className="grid md:grid-cols-3 gap-8">
          <Link href="/textbook/representation" className="group">
            <div className="border border-gray-200 rounded-xl p-8 hover:shadow-lg transition-shadow">
              <div className="text-blue-600 font-bold text-sm mb-2">CHAPTER 1</div>
              <h4 className="text-xl font-bold text-gray-900 mb-3 group-hover:text-blue-600 transition-colors">
                Representation
              </h4>
              <p className="text-gray-600">
                Latent spaces for robotics, multi-modal alignment, and scene tokenization.
              </p>
            </div>
          </Link>

          <Link href="/textbook/reasoning" className="group">
            <div className="border border-gray-200 rounded-xl p-8 hover:shadow-lg transition-shadow">
              <div className="text-purple-600 font-bold text-sm mb-2">CHAPTER 2</div>
              <h4 className="text-xl font-bold text-gray-900 mb-3 group-hover:text-purple-600 transition-colors">
                Reasoning
              </h4>
              <p className="text-gray-600">
                Foundation models as world models, planning vs. execution, chain-of-thought.
              </p>
            </div>
          </Link>

          <Link href="/textbook/scaling" className="group">
            <div className="border border-gray-200 rounded-xl p-8 hover:shadow-lg transition-shadow">
              <div className="text-green-600 font-bold text-sm mb-2">CHAPTER 3</div>
              <h4 className="text-xl font-bold text-gray-900 mb-3 group-hover:text-green-600 transition-colors">
                Scaling
              </h4>
              <p className="text-gray-600">
                Data pipelines, semantic supervision, policy distillation, and safety-critical validation.
              </p>
            </div>
          </Link>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-gray-200 bg-gray-50 mt-24">
        <div className="max-w-7xl mx-auto px-6 py-12">
          <div className="text-center text-sm text-gray-600">
            <p>A reference for VLA systems.</p>
            <p className="mt-2">© 2025 Chris Heckman</p>
          </div>
        </div>
      </footer>
    </div>
  );
}
