import type { AgentTool } from "@mariozechner/pi-agent-core";
import { Text } from "@mariozechner/pi-tui";
import { type Static, Type } from "@sinclair/typebox";
import type { ToolDefinition } from "../extensions/types.js";
import { wrapToolDefinition } from "./tool-definition-wrapper.js";

/**
 * `editdone` is a pure signalling tool used in implement mode.
 *
 * The agent loop iterates planned files one-by-one. For each plan the model
 * applies `edit`/`write` calls until it is satisfied, then calls `editdone`
 * with a short evidence blurb. The agent loop consumes the `editdone` call
 * to advance to the next plan; after the Nth `editdone` (= plan count) the
 * run ends.
 *
 * The tool itself performs no filesystem work — it just echoes back a
 * confirmation message. All state transitions happen in the agent loop.
 */
const editdoneSchema = Type.Object(
	{
		filepath: Type.String({
			description: "Path of the planned file whose implementation is being declared complete.",
		}),
		plan: Type.String({
			description: "The plan text for this file (copy verbatim from the plan submission).",
		}),
		completedevidence: Type.String({
			description:
				"Short explanation (1-4 sentences) of why this plan is fully implemented — what edits were applied and how they satisfy the plan's acceptance criteria.",
		}),
	},
	{ additionalProperties: false },
);

export type EditDoneToolInput = Static<typeof editdoneSchema>;

export interface EditDoneToolDetails {
	filepath: string;
	completedevidence: string;
}

export function createEditDoneToolDefinition(): ToolDefinition<
	typeof editdoneSchema,
	EditDoneToolDetails,
	Record<string, never>
> {
	return {
		name: "editdone",
		label: "editdone",
		description:
			"Signal that the CURRENT plan is fully implemented. Call this ONLY in implement mode, and only after you have applied all `edit`/`write` calls needed for the current planned file. Required payload: { filepath, plan, completedevidence }. Calling `editdone` advances the agent to the next planned file (or ends the run after the last plan).",
		promptSnippet: "Signal completion of the current plan. Required: { filepath, plan, completedevidence }.",
		promptGuidelines: [
			"Only call `editdone` after you finished applying edits for the current planned file.",
			"`filepath` and `plan` must match the plan the agent is currently asking you to implement.",
			"`completedevidence` is a short justification of completion — summarize what you changed and why it satisfies the plan.",
		],
		parameters: editdoneSchema,
		async execute(_toolCallId, input: EditDoneToolInput) {
			const filepath = (input.filepath ?? "").trim();
			const completedevidence = (input.completedevidence ?? "").trim();
			return {
				content: [
					{
						type: "text" as const,
						text: `editdone received for \`${filepath}\`. Evidence: ${completedevidence || "(none)"}`,
					},
				],
				details: { filepath, completedevidence },
			};
		},
		renderCall(args, theme, context) {
			const text = (context.lastComponent as Text | undefined) ?? new Text("", 0, 0);
			const fp = typeof (args as any)?.filepath === "string" ? (args as any).filepath : "";
			text.setText(`${theme.fg("toolTitle", theme.bold("editdone"))} ${theme.fg("accent", fp)}`);
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

export function createEditDoneTool(): AgentTool<typeof editdoneSchema> {
	return wrapToolDefinition(createEditDoneToolDefinition());
}

export const editdoneToolDefinition = createEditDoneToolDefinition();
export const editdoneTool = createEditDoneTool();
