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

  // Sort by a composite: quality_score * 0.6 + normalized_comment_volume * 0.4
  const maxComments = Math.max(...reviewers.map((r) => r.total_comments));
  const ranked = [...reviewers].sort((a, b) => {
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

        {/* Charts (client component) */}
        <ReviewerCharts reviewers={reviewers} auditPrs={audit_prs} />

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

        {/* Footer note */}
        <div className="mt-16 pt-8 border-t border-gray-200">
          <p className="text-xs text-gray-400 text-center">
            Data collected from{" "}
            {audit_prs.map((pr) => `#${pr.number}`).join(", ")} on{" "}
            {new Date(data.generated_at).toLocaleDateString("en-US", {
              year: "numeric",
              month: "long",
              day: "numeric",
            })}
            . Quality scores are heuristic-based (comment length, technical references, constructiveness).
            Instructor and bot comments are excluded.
          </p>
        </div>
      </div>
    </div>
  );
}
