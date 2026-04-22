import type { Readable } from "node:stream";
import { StringDecoder } from "node:string_decoder";

/**
 * Serialize a single strict JSONL record.
 *
 * Framing is LF-only. Payload strings may contain other Unicode separators such as
 * U+2028 and U+2029. Clients must split records on `\n` only.
 */
export function serializeJsonLine(value: unknown): string {
	return `${safeJsonStringify(value)}\n`;
}

/**
 * JSON.stringify for JSONL streams: handles bigint and **true** cyclic references only.
 *
 * A naive WeakSet marks repeated object references as "circular", which breaks DAGs
 * (same object reachable twice) and corrupts events with `"[Circular]"` strings — breaking
 * harnesses that expect `agent_end` and valid shapes. This follows the stack-based approach
 * from json-stringify-safe: only values already on the current ancestor path are cycles.
 */
export function safeJsonStringify(value: unknown): string {
	try {
		return JSON.stringify(value, serializer());
	} catch (err) {
		const message = err instanceof Error ? err.message : String(err);
		return JSON.stringify({
			type: "json_serialization_error",
			error: message,
		});
	}
}

function serializer(): (this: unknown, key: string, value: unknown) => unknown {
	const stack: unknown[] = [];
	const keys: string[] = [];

	const cycleReplacer = function (this: unknown, _key: string, value: unknown): string {
		if (stack[0] === value) return "[Circular ~]";
		const idx = stack.indexOf(value);
		return `[Circular ~.${keys.slice(0, idx).join(".")}]`;
	};

	return function (this: unknown, key: string, value: unknown): unknown {
		// JSON.stringify throws on Symbol; functions are normally omitted but custom structures may pass them through.
		if (typeof value === "symbol" || typeof value === "function") return undefined;
		if (typeof value === "bigint") return value.toString();

		if (stack.length > 0) {
			const thisPos = stack.indexOf(this);
			if (~thisPos) {
				stack.splice(thisPos + 1);
				keys.splice(thisPos, Number.POSITIVE_INFINITY, key);
			} else {
				stack.push(this);
				keys.push(key);
			}
			if (~stack.indexOf(value)) {
				return cycleReplacer.call(this, key, value);
			}
		} else {
			stack.push(value);
		}

		return value;
	};
}

/**
 * Attach an LF-only JSONL reader to a stream.
 *
 * This intentionally does not use Node readline. Readline splits on additional
 * Unicode separators that are valid inside JSON strings and therefore does not
 * implement strict JSONL framing.
 */
export function attachJsonlLineReader(stream: Readable, onLine: (line: string) => void): () => void {
	const decoder = new StringDecoder("utf8");
	let buffer = "";

	const emitLine = (line: string) => {
		onLine(line.endsWith("\r") ? line.slice(0, -1) : line);
	};

	const onData = (chunk: string | Buffer) => {
		buffer += typeof chunk === "string" ? chunk : decoder.write(chunk);

		while (true) {
			const newlineIndex = buffer.indexOf("\n");
			if (newlineIndex === -1) {
				return;
			}

			emitLine(buffer.slice(0, newlineIndex));
			buffer = buffer.slice(newlineIndex + 1);
		}
	};

	const onEnd = () => {
		buffer += decoder.end();
		if (buffer.length > 0) {
			emitLine(buffer);
			buffer = "";
		}
	};

	stream.on("data", onData);
	stream.on("end", onEnd);

	return () => {
		stream.off("data", onData);
		stream.off("end", onEnd);
	};
}
