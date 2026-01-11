import Link from "next/link";

export default function CoursePage() {
  return (
    <div className="min-h-screen bg-white">
      <div className="max-w-4xl mx-auto px-6 py-16">
        <Link href="/" className="text-sm text-blue-600 hover:text-blue-800 mb-8 inline-block">
          ← Back to Home
        </Link>

        <h1 className="text-4xl font-bold text-gray-900 mb-4">
          CSCI 7000: VLA Foundations for Robotics
        </h1>
        <p className="text-xl text-gray-600 mb-12">
          Spring 2026 • Instructor: Christoffer Heckman
        </p>

        <div className="space-y-12">
          {/* Syllabus */}
          <section>
            <h2 className="text-2xl font-bold text-gray-900 mb-4">Course Materials</h2>
            <div className="border border-gray-200 rounded-lg p-6 hover:shadow-md transition-shadow">
              <h3 className="text-lg font-semibold text-gray-900 mb-2">Syllabus</h3>
              <p className="text-gray-600 mb-4">
                Complete course information, grading policy, and schedule.
              </p>
              <Link href="/course/syllabus" className="text-blue-600 hover:text-blue-800 font-medium">
                View Syllabus →
              </Link>
            </div>
          </section>

          {/* Assignments */}
          <section>
            <h2 className="text-2xl font-bold text-gray-900 mb-4">Assignments</h2>
            <div className="grid gap-4">
              <div className="border border-gray-200 rounded-lg p-6 hover:shadow-md transition-shadow">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-lg font-semibold text-gray-900">Scratch-0: Environment Setup</h3>
                  <span className="text-xs font-medium text-green-600 bg-green-50 px-3 py-1 rounded-full">
                    50 points
                  </span>
                </div>
                <p className="text-gray-600 mb-4">
                  Set up your development environment and submit your first PR.
                </p>
                <Link href="/course/assignments/scratch-0" className="text-blue-600 hover:text-blue-800 font-medium">
                  View Assignment →
                </Link>
              </div>

              <div className="border border-gray-200 rounded-lg p-6 hover:shadow-md transition-shadow">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-lg font-semibold text-gray-900">Scratch-1: Transformer Backbone</h3>
                  <span className="text-xs font-medium text-blue-600 bg-blue-50 px-3 py-1 rounded-full">
                    100 points
                  </span>
                </div>
                <p className="text-gray-600 mb-4">
                  Implement a decoder-only Transformer from scratch for robotic trajectories.
                </p>
                <Link href="/course/assignments/scratch-1" className="text-blue-600 hover:text-blue-800 font-medium">
                  View Assignment →
                </Link>
              </div>

              <div className="border border-gray-200 rounded-lg p-6 hover:shadow-md transition-shadow">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-lg font-semibold text-gray-900">Paper Audit</h3>
                  <span className="text-xs font-medium text-purple-600 bg-purple-50 px-3 py-1 rounded-full">
                    200 points
                  </span>
                </div>
                <p className="text-gray-600 mb-4">
                  Critical analysis of VLA research papers (4 audits total).
                </p>
                <Link href="/course/assignments/paper-audit" className="text-blue-600 hover:text-blue-800 font-medium">
                  View Assignment →
                </Link>
              </div>

              <div className="border border-gray-200 rounded-lg p-6 hover:shadow-md transition-shadow">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-lg font-semibold text-gray-900">Capstone Project</h3>
                  <span className="text-xs font-medium text-orange-600 bg-orange-50 px-3 py-1 rounded-full">
                    300 points
                  </span>
                </div>
                <p className="text-gray-600 mb-4">
                  Textbook contribution with implementation or comprehensive survey.
                </p>
                <Link href="/course/assignments/capstone" className="text-blue-600 hover:text-blue-800 font-medium">
                  View Assignment →
                </Link>
              </div>
            </div>
          </section>

          {/* Repository */}
          <section>
            <h2 className="text-2xl font-bold text-gray-900 mb-4">Course Repository</h2>
            <div className="border border-gray-200 rounded-lg p-6 hover:shadow-md transition-shadow">
              <h3 className="text-lg font-semibold text-gray-900 mb-2">GitHub Repository</h3>
              <p className="text-gray-600 mb-4">
                All assignments must be submitted via pull request to the course repository.
              </p>
              <a
                href="https://github.com/arpg/vla-foundations"
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-600 hover:text-blue-800 font-medium"
              >
                View on GitHub →
              </a>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}
