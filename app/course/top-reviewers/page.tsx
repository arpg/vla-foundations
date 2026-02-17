import Link from "next/link";
import fs from "fs";
import path from "path";
import ReviewerCharts from "./ReviewerCharts";

interface Reviewer {
  login: string;
  avatar_url: string;
  name: string;
  total_comments: number;
  inline_comments: number;
  discussion_comments: number;
  prs_reviewed: number[];
  quality_score: number;
  quality_tier: string;
  is_instructor?: boolean;
  sample_comments: string[];
  comment_categories: {
    technical_depth: number;
    constructive_suggestion: number;
    clarification_request: number;
    praise: number;
  };
}

interface AuditPR {
  number: number;
  title: string;
  authors: string[];
  total_comments: number;
}

interface ReviewerData {
  generated_at: string;
  reviewers: Reviewer[];
  audit_prs: AuditPR[];
}

function getReviewerData(): ReviewerData {
  const filePath = path.join(process.cwd(), "data", "reviewer-stats.json");
  const raw = fs.readFileSync(filePath, "utf8");
  return JSON.parse(raw);
}

const TIER_STYLES: Record<string, { bg: string; text: string; border: string }> = {
  Instructor: { bg: "bg-rose-50", text: "text-rose-700", border: "border-rose-200" },
  Exemplary: { bg: "bg-emerald-50", text: "text-emerald-700", border: "border-emerald-200" },
  Strong: { bg: "bg-blue-50", text: "text-blue-700", border: "border-blue-200" },
  Solid: { bg: "bg-violet-50", text: "text-violet-700", border: "border-violet-200" },
  Developing: { bg: "bg-gray-50", text: "text-gray-600", border: "border-gray-200" },
};

const PODIUM_STYLES = [
  { ring: "ring-amber-400", label: "1st", gradient: "from-amber-50 to-amber-100/50" },
  { ring: "ring-gray-300", label: "2nd", gradient: "from-gray-50 to-gray-100/50" },
  { ring: "ring-amber-600/60", label: "3rd", gradient: "from-orange-50 to-orange-100/50" },
];

function TierBadge({ tier }: { tier: string }) {
  const style = TIER_STYLES[tier] || TIER_STYLES.Developing;
  return (
    <span className={`text-xs font-medium px-2 py-0.5 rounded ${style.bg} ${style.text} ${style.border} border`}>
      {tier}
    </span>
  );
}

export default function TopReviewersPage() {
  const data = getReviewerData();
  const { reviewers, audit_prs } = data;

  // Separate instructor from students
  const instructor = reviewers.find((r) => r.is_instructor);
  const students = reviewers.filter((r) => !r.is_instructor);

  // Sort students by composite: quality_score * 0.6 + normalized_comment_volume * 0.4
  const maxComments = Math.max(...students.map((r) => r.total_comments), 1);
  const ranked = [...students].sort((a, b) => {
    const scoreA = a.quality_score * 0.6 + (a.total_comments / maxComments) * 10 * 0.4;
    const scoreB = b.quality_score * 0.6 + (b.total_comments / maxComments) * 10 * 0.4;
    return scoreB - scoreA;
  });

  const top3 = ranked.slice(0, 3);
  const totalComments = reviewers.reduce((sum, r) => sum + r.total_comments, 0);
  const activePrs = audit_prs.filter((pr) => pr.total_comments > 0).length;

  return (
    <div className="min-h-screen bg-white">
      <div className="max-w-6xl mx-auto px-6 py-16">
        <Link
          href="/course"
          className="text-sm text-blue-600 hover:text-blue-800 mb-8 inline-block"
        >
          &larr; Back to Course
        </Link>

        {/* Hero */}
        <div className="mb-16">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            Top Reviewers
          </h1>
          <p className="text-xl text-gray-600 mb-8">
            Celebrating the peer reviewers who make our textbook audits better
            through thoughtful, constructive feedback.
          </p>

          {/* Summary stats */}
          <div className="grid grid-cols-3 gap-4">
            <div className="border border-gray-200 rounded-lg p-4 text-center">
              <p className="text-3xl font-bold text-gray-900">{totalComments}</p>
              <p className="text-sm text-gray-500">Total Review Comments</p>
            </div>
            <div className="border border-gray-200 rounded-lg p-4 text-center">
              <p className="text-3xl font-bold text-gray-900">{reviewers.length}</p>
              <p className="text-sm text-gray-500">Active Reviewers</p>
            </div>
            <div className="border border-gray-200 rounded-lg p-4 text-center">
              <p className="text-3xl font-bold text-gray-900">{activePrs}</p>
              <p className="text-sm text-gray-500">Audit PRs Reviewed</p>
            </div>
          </div>
        </div>

        {/* Instructor — The Benchmark */}
        {instructor && (
          <section className="mb-16">
            <h2 className="text-2xl font-bold text-gray-900 mb-4">
              The Benchmark
            </h2>
            <div className="border border-rose-200 rounded-lg p-6 bg-gradient-to-r from-rose-50 to-white">
              <div className="flex items-center gap-5">
                <img
                  src={instructor.avatar_url}
                  alt={instructor.login}
                  className="w-16 h-16 rounded-full ring-2 ring-rose-300"
                />
                <div className="flex-1">
                  <div className="flex items-center gap-3 mb-1">
                    <h3 className="text-lg font-semibold text-gray-900">
                      @{instructor.login}
                    </h3>
                    <TierBadge tier="Instructor" />
                    <span className="text-sm font-mono text-rose-600">10.0/10</span>
                  </div>
                  <p className="text-sm text-gray-500 mb-2">
                    Student quality scores are measured relative to the instructor&apos;s review contributions.
                  </p>
                  <div className="flex gap-4 text-sm text-gray-600">
                    <span>
                      <span className="font-semibold text-gray-900">
                        {instructor.total_comments}
                      </span>{" "}
                      comments
                    </span>
                    <span>
                      <span className="font-semibold text-gray-900">
                        {instructor.prs_reviewed.length}
                      </span>{" "}
                      PRs reviewed
                    </span>
                    <span>
                      <span className="font-semibold text-gray-900">
                        {instructor.inline_comments}
                      </span>{" "}
                      inline
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </section>
        )}

        {/* Top 3 Podium */}
        <section className="mb-16">
          <h2 className="text-2xl font-bold text-gray-900 mb-6">
            Podium
          </h2>
          <div className="grid md:grid-cols-3 gap-6">
            {top3.map((reviewer, index) => {
              const podium = PODIUM_STYLES[index];
              return (
                <div
                  key={reviewer.login}
                  className={`relative border border-gray-200 rounded-lg p-6 bg-gradient-to-b ${podium.gradient}`}
                >
                  <div className="absolute top-3 right-3 text-sm font-bold text-gray-400">
                    {podium.label}
                  </div>
                  <div className="flex flex-col items-center text-center">
                    <img
                      src={reviewer.avatar_url}
                      alt={reviewer.login}
                      className={`w-20 h-20 rounded-full ring-4 ${podium.ring} mb-4`}
                    />
                    <h3 className="text-lg font-semibold text-gray-900">
                      @{reviewer.login}
                    </h3>
                    <div className="mt-2 mb-3">
                      <TierBadge tier={reviewer.quality_tier} />
                    </div>
                    <div className="flex gap-4 text-sm text-gray-600">
                      <span>
                        <span className="font-semibold text-gray-900">
                          {reviewer.total_comments}
                        </span>{" "}
                        comments
                      </span>
                      <span>
                        <span className="font-semibold text-gray-900">
                          {reviewer.prs_reviewed.length}
                        </span>{" "}
                        PRs
                      </span>
                    </div>
                    <p className="text-xs text-gray-500 mt-1">
                      Quality: {reviewer.quality_score}/10
                    </p>
                  </div>
                </div>
              );
            })}
          </div>
        </section>

        {/* Charts (client component) — students only */}
        <ReviewerCharts reviewers={students} auditPrs={audit_prs} />

        {/* Full Reviewer Grid */}
        <section className="mt-16">
          <h2 className="text-2xl font-bold text-gray-900 mb-6">
            All Reviewers
          </h2>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {ranked.map((reviewer) => (
              <div
                key={reviewer.login}
                className="border border-gray-200 rounded-lg p-5 hover:shadow-md transition-shadow"
              >
                <div className="flex items-start gap-3 mb-3">
                  <img
                    src={reviewer.avatar_url}
                    alt={reviewer.login}
                    className="w-10 h-10 rounded-full"
                  />
                  <div className="flex-1 min-w-0">
                    <h3 className="text-sm font-semibold text-gray-900 truncate">
                      @{reviewer.login}
                    </h3>
                    <div className="flex items-center gap-2 mt-1">
                      <TierBadge tier={reviewer.quality_tier} />
                      <span className="text-xs text-gray-500">
                        {reviewer.quality_score}/10
                      </span>
                    </div>
                  </div>
                </div>

                <div className="flex gap-4 text-xs text-gray-600 mb-3">
                  <span>{reviewer.total_comments} comments</span>
                  <span>{reviewer.inline_comments} inline</span>
                  <span>{reviewer.prs_reviewed.length} PRs</span>
                </div>

                {reviewer.sample_comments.length > 0 && (
                  <div className="border-t border-gray-100 pt-3">
                    <p className="text-xs text-gray-500 italic line-clamp-3">
                      &ldquo;{reviewer.sample_comments[0]}&rdquo;
                    </p>
                  </div>
                )}
              </div>
            ))}
          </div>
        </section>

        {/* How quality scores work */}
        <section className="mt-16 pt-8 border-t border-gray-200">
          <h2 className="text-2xl font-bold text-gray-900 mb-4">
            How Quality Scores Work
          </h2>
          <div className="prose prose-sm prose-slate max-w-none">
            <p className="text-gray-600 mb-4">
              Each review comment earns weighted points based on its content.
              Points are <strong>summed</strong> across all of a reviewer&apos;s comments to produce
              a total contribution score, then normalized against the <strong>instructor&apos;s
              contributions as a benchmark</strong> (= 10.0) using square-root scaling.
              This rewards both quality and engagement &mdash; writing more high-quality
              comments always helps your score, and you are never penalized for being prolific.
            </p>
            <div className="grid md:grid-cols-2 gap-4 mb-6">
              <div className="border border-gray-200 rounded-lg p-4">
                <h3 className="text-sm font-semibold text-gray-900 mb-2">Scoring Rules</h3>
                <ul className="text-sm text-gray-600 space-y-1">
                  <li><span className="font-medium text-emerald-700">+3</span> Technical depth &mdash; references specific code, formulas, or implementation details</li>
                  <li><span className="font-medium text-blue-700">+2</span> Constructive suggestion &mdash; proposes alternatives (&ldquo;consider&rdquo;, &ldquo;might want to&rdquo;, etc.)</li>
                  <li><span className="font-medium text-amber-700">+2</span> Clarification request &mdash; asks a substantive question (&gt;30 chars with &ldquo;?&rdquo;)</li>
                  <li><span className="font-medium text-gray-700">+1</span> Substantive &mdash; comment exceeds 100 characters</li>
                  <li><span className="font-medium text-gray-400">+0</span> Brief/low-effort &mdash; under 30 characters or generic (&ldquo;LGTM&rdquo;, &ldquo;nice&rdquo;)</li>
                </ul>
              </div>
              <div className="border border-gray-200 rounded-lg p-4">
                <h3 className="text-sm font-semibold text-gray-900 mb-2">Quality Tiers</h3>
                <ul className="text-sm text-gray-600 space-y-1">
                  <li><span className="font-medium text-rose-600">Instructor</span> &mdash; 10.0 (the benchmark)</li>
                  <li><span className="font-medium text-emerald-700">Exemplary</span> &mdash; 5.0+/10 (50%+ of instructor)</li>
                  <li><span className="font-medium text-blue-700">Strong</span> &mdash; 3.5&ndash;4.9</li>
                  <li><span className="font-medium text-violet-700">Solid</span> &mdash; 2.0&ndash;3.4</li>
                  <li><span className="font-medium text-gray-600">Developing</span> &mdash; below 2.0</li>
                </ul>
                <p className="text-xs text-gray-500 mt-3">
                  Categories stack &mdash; a technical question earns both &ldquo;technical depth&rdquo;
                  and &ldquo;clarification&rdquo; points. Scores are normalized with &radic; scaling
                  against the class maximum, so doubling your contribution raises your score
                  by ~41%, not 100%.
                </p>
              </div>
            </div>
          </div>

          {/* crh-bot disclaimer */}
          <div className="mt-8 flex items-start gap-3 border border-gray-200 rounded-lg p-4 bg-gray-50">
            <img
              src="https://avatars.githubusercontent.com/u/260777175?v=4"
              alt="crh-bot"
              className="w-8 h-8 rounded-full flex-shrink-0 mt-0.5"
            />
            <p className="text-xs text-gray-500">
              Quality scores and tier assignments are automatically generated by{" "}
              <span className="font-medium text-gray-700">crh-bot</span> using
              keyword heuristics and do not reflect the instructor&apos;s explicit views
              on any individual reviewer&apos;s work. They are meant to celebrate engagement,
              not to grade. Data collected from PRs{" "}
              {audit_prs.map((pr) => `#${pr.number}`).join(", ")} on{" "}
              {new Date(data.generated_at).toLocaleDateString("en-US", {
                year: "numeric",
                month: "long",
                day: "numeric",
              })}.
            </p>
          </div>
        </section>
      </div>
    </div>
  );
}
