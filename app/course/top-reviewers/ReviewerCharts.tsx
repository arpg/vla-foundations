"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  Legend,
} from "recharts";

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

interface ReviewerChartsProps {
  reviewers: Reviewer[];
  auditPrs: AuditPR[];
}

const TIER_COLORS: Record<string, string> = {
  Exemplary: "#059669",
  Strong: "#2563eb",
  Solid: "#7c3aed",
  Developing: "#6b7280",
};

const CATEGORY_COLORS: Record<string, string> = {
  technical_depth: "#059669",
  constructive_suggestion: "#2563eb",
  clarification_request: "#f59e0b",
  praise: "#ec4899",
};

const CATEGORY_LABELS: Record<string, string> = {
  technical_depth: "Technical Depth",
  constructive_suggestion: "Constructive Suggestion",
  clarification_request: "Clarification",
  praise: "Praise",
};

export default function ReviewerCharts({
  reviewers,
  auditPrs,
}: ReviewerChartsProps) {
  // Sort by total comments for volume chart
  const volumeData = [...reviewers]
    .sort((a, b) => b.total_comments - a.total_comments)
    .map((r) => ({
      login: r.login,
      comments: r.total_comments,
      tier: r.quality_tier,
    }));

  // Quality score data
  const qualityData = [...reviewers]
    .sort((a, b) => b.quality_score - a.quality_score)
    .map((r) => ({
      login: r.login,
      score: r.quality_score,
      tier: r.quality_tier,
    }));

  // PR activity data - each reviewer's comments per PR
  const prActivityData = auditPrs
    .filter((pr) => pr.total_comments > 0)
    .map((pr) => {
      const entry: Record<string, string | number> = {
        pr: `#${pr.number}`,
        title: pr.title,
      };
      for (const r of reviewers) {
        // We don't have per-PR-per-reviewer counts in the data,
        // so show 1 if they reviewed the PR, 0 otherwise
        entry[r.login] = r.prs_reviewed.includes(pr.number) ? 1 : 0;
      }
      return entry;
    });

  // Category breakdown data
  const categoryData = [...reviewers]
    .sort((a, b) => b.total_comments - a.total_comments)
    .filter((r) => r.total_comments >= 3)
    .map((r) => ({
      login: r.login,
      ...r.comment_categories,
    }));

  // Custom tooltip for volume chart
  const VolumeTooltip = ({ active, payload, label }: {
    active?: boolean;
    payload?: Array<{ value: number; payload: { tier: string } }>;
    label?: string;
  }) => {
    if (!active || !payload?.length) return null;
    return (
      <div className="bg-white border border-gray-200 rounded-lg p-3 shadow-lg">
        <p className="font-semibold text-gray-900">@{label}</p>
        <p className="text-sm text-gray-600">
          {payload[0].value} comments
        </p>
        <p className="text-sm" style={{ color: TIER_COLORS[payload[0].payload.tier] }}>
          {payload[0].payload.tier}
        </p>
      </div>
    );
  };

  const QualityTooltip = ({ active, payload, label }: {
    active?: boolean;
    payload?: Array<{ value: number; payload: { tier: string } }>;
    label?: string;
  }) => {
    if (!active || !payload?.length) return null;
    return (
      <div className="bg-white border border-gray-200 rounded-lg p-3 shadow-lg">
        <p className="font-semibold text-gray-900">@{label}</p>
        <p className="text-sm text-gray-600">
          Quality: {payload[0].value}/10
        </p>
        <p className="text-sm" style={{ color: TIER_COLORS[payload[0].payload.tier] }}>
          {payload[0].payload.tier}
        </p>
      </div>
    );
  };

  return (
    <div className="space-y-16">
      {/* Comment Volume */}
      <section>
        <h2 className="text-2xl font-bold text-gray-900 mb-2">
          Comment Volume
        </h2>
        <p className="text-gray-600 mb-6">
          Total review comments per reviewer across all audit PRs.
        </p>
        <div className="bg-white border border-gray-200 rounded-lg p-6">
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={volumeData} margin={{ bottom: 60 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis
                dataKey="login"
                angle={-45}
                textAnchor="end"
                height={80}
                tick={{ fontSize: 12, fill: "#6b7280" }}
              />
              <YAxis
                tick={{ fontSize: 12, fill: "#6b7280" }}
                label={{
                  value: "Comments",
                  angle: -90,
                  position: "insideLeft",
                  style: { fill: "#6b7280", fontSize: 12 },
                }}
              />
              <Tooltip content={<VolumeTooltip />} />
              <Bar dataKey="comments" radius={[4, 4, 0, 0]}>
                {volumeData.map((entry, index) => (
                  <Cell
                    key={index}
                    fill={TIER_COLORS[entry.tier] || "#6b7280"}
                    fillOpacity={0.8}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </section>

      {/* Quality Scores */}
      <section>
        <h2 className="text-2xl font-bold text-gray-900 mb-2">
          Quality Scores
        </h2>
        <p className="text-gray-600 mb-6">
          Quality score relative to instructor benchmark (10.0).
          Tier bands: Exemplary (5+), Strong (3.5-5), Solid (2-3.5), Developing (&lt;2).
        </p>
        <div className="bg-white border border-gray-200 rounded-lg p-6">
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={qualityData} margin={{ bottom: 60 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis
                dataKey="login"
                angle={-45}
                textAnchor="end"
                height={80}
                tick={{ fontSize: 12, fill: "#6b7280" }}
              />
              <YAxis
                domain={[0, 10]}
                ticks={[0, 2, 4, 6, 8, 10]}
                tick={{ fontSize: 12, fill: "#6b7280" }}
                label={{
                  value: "Quality Score",
                  angle: -90,
                  position: "insideLeft",
                  style: { fill: "#6b7280", fontSize: 12 },
                }}
              />
              <Tooltip content={<QualityTooltip />} />
              <Bar dataKey="score" radius={[4, 4, 0, 0]}>
                {qualityData.map((entry, index) => (
                  <Cell
                    key={index}
                    fill={TIER_COLORS[entry.tier] || "#6b7280"}
                    fillOpacity={0.8}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </section>

      {/* Review Breadth */}
      <section>
        <h2 className="text-2xl font-bold text-gray-900 mb-2">
          Review Activity by PR
        </h2>
        <p className="text-gray-600 mb-6">
          Which audit PRs each reviewer engaged with. Wider participation = better peer review culture.
        </p>
        <div className="bg-white border border-gray-200 rounded-lg p-6">
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={prActivityData} margin={{ bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis
                dataKey="pr"
                tick={{ fontSize: 12, fill: "#6b7280" }}
              />
              <YAxis
                tick={{ fontSize: 12, fill: "#6b7280" }}
                label={{
                  value: "Reviewers",
                  angle: -90,
                  position: "insideLeft",
                  style: { fill: "#6b7280", fontSize: 12 },
                }}
              />
              <Tooltip />
              <Legend
                wrapperStyle={{ fontSize: 10 }}
                iconSize={8}
              />
              {reviewers
                .filter((r) => r.total_comments >= 3)
                .slice(0, 8)
                .map((r, i) => {
                  const colors = [
                    "#059669", "#2563eb", "#7c3aed", "#f59e0b",
                    "#ec4899", "#14b8a6", "#f97316", "#6366f1",
                  ];
                  return (
                    <Bar
                      key={r.login}
                      dataKey={r.login}
                      stackId="a"
                      fill={colors[i % colors.length]}
                      fillOpacity={0.7}
                    />
                  );
                })}
            </BarChart>
          </ResponsiveContainer>
        </div>
      </section>

      {/* Comment Categories */}
      <section>
        <h2 className="text-2xl font-bold text-gray-900 mb-2">
          Comment Category Breakdown
        </h2>
        <p className="text-gray-600 mb-6">
          Types of review comments by reviewer. Technical depth and constructive suggestions
          score highest in quality metrics.
        </p>
        <div className="bg-white border border-gray-200 rounded-lg p-6">
          <ResponsiveContainer width="100%" height={Math.max(300, categoryData.length * 40 + 60)}>
            <BarChart
              data={categoryData}
              layout="vertical"
              margin={{ left: 100 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis type="number" tick={{ fontSize: 12, fill: "#6b7280" }} />
              <YAxis
                type="category"
                dataKey="login"
                tick={{ fontSize: 12, fill: "#6b7280" }}
                width={90}
              />
              <Tooltip />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              {Object.entries(CATEGORY_COLORS).map(([key, color]) => (
                <Bar
                  key={key}
                  dataKey={key}
                  name={CATEGORY_LABELS[key]}
                  stackId="a"
                  fill={color}
                  fillOpacity={0.8}
                />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>
      </section>
    </div>
  );
}
