import { useState } from "react";
import { LoremIpsum } from "lorem-ipsum";
import type {
  RepoPullRequest,
  RepoMerge,
  RepoModification,
  ChangeAuthor,
  ChangeStats,
  KernelType,
} from "../types";
import { FaCodePullRequest, FaCodeMerge } from "react-icons/fa6";
import ChangeStatBar from "./ChangeStatBar";

const AUTHORS: ChangeAuthor[] = [
  {
    name: "Surya Jasper",
    profileUrl: "https://avatars.githubusercontent.com/u/45545431?v=4",
  },
  {
    name: "James Smith",
    profileUrl: "https://avatars.githubusercontent.com/u/45545432?v=4",
  },
  {
    name: "Jane Doe",
    profileUrl: "https://avatars.githubusercontent.com/u/45545433?v=4",
  },
  {
    name: "Lip Gallagher",
    profileUrl: "https://avatars.githubusercontent.com/u/45545434?v=4",
  },
  {
    name: "Jules Vinyard",
    profileUrl: "https://avatars.githubusercontent.com/u/45545435?v=4",
  },
];

const lorem = new LoremIpsum();

const KERNEL_TYPES = ["gemm", "attention", "convolution"] as const;

function randomStats(): ChangeStats {
  const stats: Record<KernelType, number> = {} as any;
  for (const k of KERNEL_TYPES) {
    stats[k as KernelType] = parseFloat((Math.random() * 150 - 50).toFixed(2)); // change of speed as percentage (-50% to +100%)
  }
  return stats;
}

function randomAuthor(): ChangeAuthor {
  return AUTHORS[Math.floor(Math.random() * AUTHORS.length)];
}

let idCounter = 0;
function nextId() {
  return (++idCounter).toString();
}

export function generateFakeRepoHistory(count: number): RepoModification[] {
  const history: RepoModification[] = [];
  const pullRequests: RepoPullRequest[] = [];
  let baseTime = Date.now() - 1000 * 60 * 60 * 24 * 30; // start 30 days ago

  for (let i = 0; i < count; i++) {
    const isPR = Math.random() < 0.7 || pullRequests.length === 0; // mostly PRs, at least 1 before merge

    const timestamp = new Date(baseTime + i * 1000 * 60 * 60 * 12); // 12hr increments

    if (isPR) {
      const pr: RepoPullRequest = {
        _id: nextId(),
        type: "pr",
        timestamp,
        author: randomAuthor(),
        title: lorem.generateSentences(1),
        description: lorem.generateParagraphs(1),
        status: "open",
        changeStats: randomStats(),
        commits: [
          {
            _id: nextId(),
            title: lorem.generateSentences(1),
            author: randomAuthor(),
            timestamp,
            description: lorem.generateParagraphs(1),
          },
          {
            _id: nextId(),
            title: lorem.generateSentences(1),
            author: randomAuthor(),
            timestamp,
            description: lorem.generateParagraphs(1),
          },
        ],
      };
      pullRequests.push(pr);
      history.push(pr);
    } else {
      // Select a random unmerged PR
      const openPRs = pullRequests.filter((pr) => pr.status === "open");
      if (openPRs.length === 0) continue;

      const targetPR = openPRs[Math.floor(Math.random() * openPRs.length)];
      targetPR.status = "closed";

      const merge: RepoMerge = {
        _id: nextId(),
        type: "merge",
        timestamp,
        author: randomAuthor(),
        prId: targetPR._id,
        changeStats: randomStats(),
      };
      history.push(merge);
    }
  }

  history.sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime());
  console.log(history);
  return history;
}

const MAX_BAR_WIDTH = 100; // in pixels

export default function History() {
  const [expandedIds, setExpandedIds] = useState<Set<string>>(new Set());
  const [modifications, setModifications] = useState<RepoModification[]>(
    generateFakeRepoHistory(20)
  );

  const toggleExpand = (id: string) => {
    const copy = new Set(expandedIds);
    copy.has(id) ? copy.delete(id) : copy.add(id);
    setExpandedIds(copy);
  };

  const getPullRequestTitleById = (prId: string) => {
    for (const modification of modifications) {
      if (modification._id === prId && modification.type === "pr") {
        const pr = modification as RepoPullRequest;
        return pr.title;
      }
    }
    return "No PR found";
  };

  return (
    <div className="px-24 py-6 space-y-8">
      <div className="flex flex-col gap-4">
        {modifications.reverse().map((mod) => {
          const isPR = mod.type === "pr";
          const isExpanded = expandedIds.has(mod._id);
          const base = isPR ? (mod as RepoPullRequest) : (mod as RepoMerge);
          const isMerge = base.type === "merge";

          const title = isMerge
            ? `Merge: ${getPullRequestTitleById(base.prId)}`
            : base.title;

          const prLink = isMerge ? "https://www.google.com" : undefined;

          return (
            <div
              key={base._id}
              className={`p-4 rounded-md shadow-md ${
                isMerge ? "bg-green-50" : "bg-gray-50"
              } cursor-pointer`}
              onClick={() => isPR && toggleExpand(base._id)}
            >
              <div className="flex justify-between items-start gap-4">
                {/* Icon */}
                <div className="mt-1">
                  {isMerge ? (
                    <FaCodeMerge className="text-green-700 text-xl" />
                  ) : (
                    <FaCodePullRequest className="text-gray-700 text-xl" />
                  )}
                </div>

                {/* Title + Author */}
                <div className="flex flex-col">
                  {isMerge ? (
                    <a
                      href={prLink}
                      className="text-lg font-semibold text-green-800 underline"
                    >
                      {title}
                    </a>
                  ) : (
                    <div className="text-lg font-semibold">{title}</div>
                  )}
                  <div className="flex items-center gap-2 mt-4">
                    <img
                      src={base.author.profileUrl}
                      alt={base.author.name}
                      className="w-6 h-6 rounded-full"
                    />
                    <span className="text-sm text-gray-700">
                      {base.author.name}
                    </span>
                  </div>
                </div>

                {/* Change Stats */}
                <div className="flex flex-col gap-2 ml-auto">
                  {Object.entries(base.changeStats).map(
                    ([kernelType, change]) => (
                      <ChangeStatBar
                        key={kernelType}
                        kernelType={kernelType}
                        change={change}
                      />
                    )
                  )}
                </div>
              </div>

              {/* Expandable PR Description */}
              {isPR && isExpanded && (base as RepoPullRequest).description && (
                <div className="mt-4 text-sm text-gray-800 whitespace-pre-line">
                  {(base as RepoPullRequest).description}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
