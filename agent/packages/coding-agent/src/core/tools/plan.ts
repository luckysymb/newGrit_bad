import type { AgentTool } from "@mariozechner/pi-agent-core";
import { Text } from "@mariozechner/pi-tui";
import { type Static, Type } from "@sinclair/typebox";
import { spawnSync } from "node:child_process";
import { basename } from "node:path";
import type { ToolDefinition } from "../extensions/types.js";
import { resolveToCwd, resolveWorkspacePath } from "./path-utils.js";
import { wrapToolDefinition } from "./tool-definition-wrapper.js";

const planItemSchema = Type.Object(
	{
		path: Type.String({ description: "File path to edit in implementation mode" }),
		plan: Type.String({ description: "Detailed edit plan for this file" }),
		acceptance_criteria: Type.Array(Type.String(), {
			description:
				"Task acceptance criteria covered by this file plan. Each value must exactly match one item in top-level task_acceptance_criteria.",
			minItems: 1,
		}),
		is_new_file: Type.Boolean({
			description: "Whether this plan item creates a new file. Use false for existing files that must be edited.",
		}),
	},
	{ additionalProperties: false },
);

const planSchema = Type.Object(
	{
		task_acceptance_criteria: Type.Array(Type.String(), {
			description: "Exact task acceptance criteria list for this task.",
			minItems: 1,
		}),
		plans: Type.Array(planItemSchema, {
			description: "Implementation plan items, each with target file, and detailed edit instructions and is_new_file flag.",
			minItems: 1,
		}),
	},
	{ additionalProperties: false },
);

export type PlanToolInput = Static<typeof planSchema>;

type NormalizedPlanItem = {
	path: string;
	plan: string;
	acceptance_criteria: string[];
	is_new_file: boolean;
};

type PlanValidationResult = {
	path: string;
	is_new_file: boolean;
	validation_result: "passed" | "failed";
	message: string;
	suggested_paths?: string[];
};

const MIN_PLAN_CHARS = 260;
const MIN_PLAN_LINES = 4;
const REQUIRED_PLAN_SECTIONS = ["Scope", "Edits", "Acceptance", "Verification"] as const;

export interface PlanToolDetails {
	planCount: number;
	paths: string[];
	allPassed: boolean;
	criteriaAllCovered: boolean;
	uncoveredCriteria: string[];
	validationResults: PlanValidationResult[];
}

function normalizePlanArgs(input: unknown): PlanToolInput {
	// Accept both:
	// 1) { plans: [{path, plan}, ...] }
	// 2) [{path, plan}, ...]
	if (Array.isArray(input)) {
		return { plans: input as any[] } as PlanToolInput;
	}
	if (!input || typeof input !== "object") return input as PlanToolInput;
	const obj = input as Record<string, unknown>;
	if (Array.isArray(obj.plans)) return input as PlanToolInput;
	return input as PlanToolInput;
}

function validatePlanDetailText(planText: string): { ok: boolean; reason?: string } {
	const text = planText.trim();
	if (text.length < MIN_PLAN_CHARS) {
		return {
			ok: false,
			reason: `failed: plan too short (${text.length} chars). Minimum is ${MIN_PLAN_CHARS} chars`,
		};
	}
	const nonEmptyLines = text
		.split(/\r?\n/)
		.map((line) => line.trim())
		.filter((line) => line.length > 0);
	if (nonEmptyLines.length < MIN_PLAN_LINES) {
		return {
			ok: false,
			reason: `failed: plan too shallow (${nonEmptyLines.length} non-empty lines). Minimum is ${MIN_PLAN_LINES}`,
		};
	}
	const missingSections = REQUIRED_PLAN_SECTIONS.filter((section) => {
		const re = new RegExp(`\\b${section}\\s*:`, "i");
		return !re.test(text);
	});
	if (missingSections.length > 0) {
		return {
			ok: false,
			reason: `failed: missing required sections (${missingSections.join(", ")}). Required format: Scope:, Edits:, Acceptance:, Verification:`,
		};
	}
	return { ok: true };
}

function addMissingRequiredSections(planText: string, acceptanceCriteria: string[]): string {
	let text = planText.trim();
	const hasSection = (name: string): boolean => new RegExp(`\\b${name}\\s*:`, "i").test(text);
	if (!hasSection("Scope")) {
		text += "\n\nScope: Implement the required behavior for this file.";
	}
	if (!hasSection("Edits")) {
		text += "\nEdits: Update the concrete symbols/blocks required by the task.";
	}
	if (!hasSection("Acceptance")) {
		const crit = acceptanceCriteria.length > 0 ? acceptanceCriteria.join(" | ") : "Task acceptance criteria assigned to this file plan.";
		text += `\nAcceptance: ${crit}`;
	}
	if (!hasSection("Verification")) {
		text += "\nVerification: Re-read this file and confirm all planned edits are present and consistent.";
	}
	return text;
}

export function createPlanToolDefinition(): ToolDefinition<typeof planSchema, PlanToolDetails> {
	return {
		name: "plan",
		label: "plan",
		description:
			"Submit final PLAN->IMPLEMENT handoff JSON. Required interface: { \"task_acceptance_criteria\": [\"...\"], \"plans\": [{ \"path\": \"relative/file/path\", \"plan\": \"Scope: ...\\nEdits: ...\\nAcceptance: ...\\nVerification: ...\", \"acceptance_criteria\": [\"criterion covered by this plan item\"], \"is_new_file\": false }] }. `plans` must be non-empty; each item must include all keys.",
		promptSnippet:
			"Call plan with exact JSON interface: { task_acceptance_criteria: [...], plans: [{ path, plan, acceptance_criteria, is_new_file }] }",
		promptGuidelines: [
			"Call plan only after broad and thorough exploration in plan mode",
			"Include every target file needed to satisfy all acceptance criteria",
			"Use exact interface keys: task_acceptance_criteria, path, plan, acceptance_criteria, is_new_file (no aliases)",
			"Each plan item must declare which acceptance criteria it covers via acceptance_criteria",
			"Coverage validation is count-based: the number of task_acceptance_criteria must match criteria covered across plan items",
			"When creating new files, use literal task/symbol names and nearest sibling naming patterns; avoid invented prefixes/suffixes unless explicitly requested",
			"Each plan item must be implementation-ready and include sections: Scope:, Edits:, Acceptance:, Verification:",
			"Set is_new_file=false for existing files and true only for files that must be newly created (basename must not already exist anywhere in the repo)",
		],
		parameters: planSchema,
		prepareArguments: normalizePlanArgs,
		async execute(_toolCallId, input: PlanToolInput) {
			const cwd = process.cwd();
			const normalizedTaskCriteria: string[] = [...new Set(
				((input.task_acceptance_criteria as string[] | undefined) ?? [])
					.map((c: string) => c.trim())
					.filter((c: string) => c.length > 0),
			)];
			if (normalizedTaskCriteria.length === 0) {
				throw new Error("plan requires non-empty task_acceptance_criteria");
			}
			const normalized: NormalizedPlanItem[] = input.plans
				.map((p: { path: string; plan: string; acceptance_criteria: string[]; is_new_file: boolean }) => ({
					path: p.path.trim(),
					plan: p.plan.trim(),
					acceptance_criteria: Array.isArray(p.acceptance_criteria)
						? [...new Set(p.acceptance_criteria.map((c) => (typeof c === "string" ? c.trim() : "")).filter((c) => c.length > 0))]
						: [],
					is_new_file: Boolean(p.is_new_file),
				}))
				.filter((p: NormalizedPlanItem) => p.path.length > 0 && p.plan.length > 0);
			if (normalized.length === 0) {
				throw new Error("plan requires at least one valid { path, plan, is_new_file } item");
			}
			ensureShellValidationToolsAvailable();
			const validationResults: PlanValidationResult[] = normalized.map((item) => {
				const normalizedPlanText = addMissingRequiredSections(item.plan, item.acceptance_criteria);
				if (item.acceptance_criteria.length === 0) {
					return {
						path: item.path,
						is_new_file: item.is_new_file,
						validation_result: "failed",
						message: "failed: acceptance_criteria must be non-empty for each plan item",
					};
				}
				const detailValidation = validatePlanDetailText(normalizedPlanText);
				if (!detailValidation.ok) {
					return {
						path: item.path,
						is_new_file: item.is_new_file,
						validation_result: "failed",
						message: detailValidation.reason ?? "failed: insufficient plan detail",
					};
				}
				if (item.is_new_file) {
					const fileName = basename(item.path.replace(/\/+$/, ""));
					const conflicts = findPathsByBasenameViaFind(cwd, fileName, 32);
					if (conflicts.length > 0) {
						return {
							path: item.path,
							is_new_file: true,
							validation_result: "failed",
							message: "failed: is_new_file but a file with this basename already exists",
							suggested_paths: conflicts,
						};
					}
					return {
						path: item.path,
						is_new_file: true,
						validation_result: "passed",
						message: "passed",
					};
				}
				const resolved = resolveWorkspacePath(item.path, cwd, { kind: "file", basenameFallback: false });
				const abs = resolveToCwd(resolved, cwd);
				if (fileExistsViaTest(abs)) {
					return {
						path: item.path,
						is_new_file: false,
						validation_result: "passed",
						message: "passed",
					};
				}
				const fileName = basename(item.path.replace(/\/+$/, ""));
				const suggestions = findPathsByBasenameViaFind(cwd, fileName, 20);
				return {
					path: item.path,
					is_new_file: false,
					validation_result: "failed",
					message: suggestions.length > 0 ? "failed: path not found; suggestions available" : "failed: path not found",
					suggested_paths: suggestions,
				};
			});
			const coveredCriteria = new Set<string>();
			for (const item of normalized) {
				for (const c of item.acceptance_criteria) coveredCriteria.add(c);
			}
			const uncoveredCriteriaCount = Math.max(0, normalizedTaskCriteria.length - coveredCriteria.size);
			const criteriaAllCovered = coveredCriteria.size === normalizedTaskCriteria.length;
			const uncoveredCriteria: string[] = uncoveredCriteriaCount > 0
				? [`count_mismatch: expected ${normalizedTaskCriteria.length}, covered ${coveredCriteria.size}`]
				: [];
			const allPassed = validationResults.every((r) => r.validation_result === "passed") && criteriaAllCovered;
			const reportLines = validationResults.map((r, idx) => {
				if (r.validation_result === "passed") {
					return `#${idx + 1} ${r.path} => passed`;
				}
				const suggestions = r.suggested_paths && r.suggested_paths.length > 0
					? `; suggested_paths=${r.suggested_paths.join(", ")}`
					: "";
				return `#${idx + 1} ${r.path} => failed${suggestions}`;
			});
			return {
				content: [
					{
						type: "text",
						text: allPassed
							? `Plan validation passed for all ${normalized.length} file(s).\nValidation_result:\n${reportLines.join("\n")}`
							: `Plan validation failed. Fix failed paths/criteria coverage and call plan again.\nEach plan item must be detailed and include sections: Scope:, Edits:, Acceptance:, Verification:.\n${!criteriaAllCovered ? `Uncovered acceptance criteria: ${uncoveredCriteria.slice(0, 10).join(" | ")}\n` : ""}Validation_result:\n${reportLines.join("\n")}`,
					},
				],
				details: {
					planCount: normalized.length,
					paths: normalized.map((p: NormalizedPlanItem) => p.path),
					allPassed,
					criteriaAllCovered,
					uncoveredCriteria,
					validationResults,
				},
			};
		},
		renderCall(args, theme, context) {
			const text = (context.lastComponent as Text | undefined) ?? new Text("", 0, 0);
			const count = Array.isArray((args as any)?.plans) ? (args as any).plans.length : 0;
			text.setText(`${theme.fg("toolTitle", theme.bold("plan"))} ${theme.fg("accent", `${count} item(s)`)}`);
			return text;
		},
		renderResult(result, _options, theme, context) {
			const text = (context.lastComponent as Text | undefined) ?? new Text("", 0, 0);
			const output =
				result.content
					?.filter((c: any) => c.type === "text")
					.map((c: any) => c.text || "")
					.join("\n") ?? "";
			text.setText(output ? `\n${theme.fg("toolOutput", output)}` : "");
			return text;
		},
	};
}

export function createPlanTool(): AgentTool<typeof planSchema> {
	return wrapToolDefinition(createPlanToolDefinition());
}

export const planToolDefinition = createPlanToolDefinition();
export const planTool = createPlanTool();

let _shellToolsChecked = false;

function spawnErrorIsENOENT(err: unknown): boolean {
	return typeof err === "object" && err !== null && "code" in err && (err as { code: unknown }).code === "ENOENT";
}

function ensureShellValidationToolsAvailable(): void {
	if (_shellToolsChecked) return;
	const findProbe = spawnSync("find", [".", "-maxdepth", "0"], { encoding: "utf8" });
	if (findProbe.error && spawnErrorIsENOENT(findProbe.error)) {
		throw new Error("plan tool validation requires `find` on PATH");
	}
	const testProbe = spawnSync("test", ["-d", process.cwd()], { encoding: "utf8" });
	if (testProbe.error && spawnErrorIsENOENT(testProbe.error)) {
		throw new Error("plan tool validation requires `test` on PATH");
	}
	_shellToolsChecked = true;
}

/** Existence check via `test -f` (no Node fs). */
function fileExistsViaTest(absolutePath: string): boolean {
	const r = spawnSync("test", ["-f", absolutePath], { encoding: "utf8" });
	return r.status === 0;
}

/**
 * List repo-relative paths to regular files whose basename equals `fileName`, via `find`.
 * Mirrors typical harness ignores (node_modules, .git, dist, …).
 */
function findPathsByBasenameViaFind(repoRoot: string, fileName: string, maxMatches: number): string[] {
	if (!fileName || fileName === "." || fileName === "..") return [];
	const args = [
		".",
		"-type",
		"f",
		"-name",
		fileName,
		"!",
		"-path",
		"*/node_modules/*",
		"!",
		"-path",
		"*/.git/*",
		"!",
		"-path",
		"*/dist/*",
		"!",
		"-path",
		"*/build/*",
		"!",
		"-path",
		"*/.next/*",
		"!",
		"-path",
		"*/coverage/*",
		"!",
		"-path",
		"*/.venv/*",
		"!",
		"-path",
		"*/venv/*",
		"!",
		"-path",
		"*/__pycache__/*",
		"!",
		"-path",
		"*/.turbo/*",
		"!",
		"-path",
		"*/.cache/*",
	];
	const r = spawnSync("find", args, {
		cwd: repoRoot,
		encoding: "utf8",
		maxBuffer: 12 * 1024 * 1024,
	});
	if (r.error) {
		throw new Error(`plan tool: find failed: ${r.error.message}`);
	}
	const raw = r.stdout ?? "";
	const lines: string[] = raw
		.split("\n")
		.map((line: string) => line.trim())
		.filter((line: string) => line.length > 0);
	const normalized: string[] = lines.map((line: string) => (line.startsWith("./") ? line.slice(2) : line));
	const sorted = [...new Set<string>(normalized)].sort();
	return sorted.slice(0, maxMatches);
}
