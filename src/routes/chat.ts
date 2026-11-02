
import { Hono } from "hono";
import { CHAT_TIMEOUT } from "../constants/timeouts";
import type { Env, Variables } from "../types/env";
import type {
  ChatCompletionRequest,
  ChatCompletionResponse,
  ChatCompletionUsage,
  OpenAIMessage,
} from "../types/openai";
import {
  invalidRequest,
  modelNotFound,
  upstreamError,
} from "../utils/errors";
import { getModels } from "../utils/model-discovery";
import { findModel } from "../utils/model-id";
import {
  buildUpstreamHeaders,
  buildUpstreamRequest,
  endpointUrl,
} from "../utils/upstream-request";
import {
  collectUpstream,
  formatCitations,
  inlineCitationLinks,
  streamUpstreamAsOpenAI,
} from "../utils/upstream-stream";

const chat = new Hono<{ Bindings: Env; Variables: Variables }>();

// ChatPlayground's maximum text length per message.
const MAX_MESSAGE_CHARS = 15_000;

// Keep upstream error messages useful but bounded.
const MAX_ERROR_DETAILS_CHARS = 1_000;

chat.post("/v1/chat/completions", async (c) => {
  const body = (await c.req.json().catch(() => null)) as
    | ChatCompletionRequest
    | null;

  if (!body || typeof body !== "object") {
    throw invalidRequest("Request body must be JSON.");
  }

  if (!body.model || typeof body.model !== "string") {
    throw invalidRequest("'model' is required.", "model");
  }

  if (!Array.isArray(body.messages) || body.messages.length === 0) {
    throw invalidRequest(
      "'messages' must be a non-empty array.",
      "messages",
    );
  }

  // Validate each message before sending it to ChatPlayground.
  // The limit applies to each message, not the entire conversation.
  for (let i = 0; i < body.messages.length; i++) {
    const message = body.messages[i];
    const chars = textChars(message.content);

    if (chars > MAX_MESSAGE_CHARS) {
      throw invalidRequest(
        `Message ${i} (${message.role}) exceeds ChatPlayground's ` +
          `${MAX_MESSAGE_CHARS}-character limit. ` +
          `Received ${chars} characters. Please reduce the message size.`,
        "messages",
      );
    }
  }

  const registry = await getModels(c.env);
  const model = findModel(body.model, registry);

  if (!model) {
    throw modelNotFound(body.model);
  }

  const { endpoint, body: upstreamBody } = buildUpstreamRequest(
    body,
    model,
  );

  const clerkUserId = c.get("clerkUserId");

  const upstream = await fetch(
    endpointUrl(endpoint, c.env.UPSTREAM_CHAT_URL),
    {
      method: "POST",
      headers: buildUpstreamHeaders(clerkUserId, c.env),
      body: JSON.stringify(upstreamBody),
      signal: AbortSignal.timeout(CHAT_TIMEOUT),
    },
  );

  // Preserve the actual upstream error instead of hiding it.
  if (!upstream.ok || !upstream.body) {
    const details = await upstream.text().catch(() => "");

    const errorDetails = details
      ? details.slice(0, MAX_ERROR_DETAILS_CHARS)
      : "Empty upstream response";

    throw upstreamError(
      upstream.status,
      `ChatPlayground returned HTTP ${upstream.status}: ${errorDetails}`,
    );
  }

  const id = `chatcmpl-${crypto.randomUUID()}`;
  const created = Math.floor(Date.now() / 1000);

  // Streaming response.
  if (body.stream) {
    const sse = streamUpstreamAsOpenAI(upstream.body, {
      id,
      model: model.id,
      created,
    });

    return new Response(sse, {
      status: 200,
      headers: {
        "content-type": "text/event-stream; charset=utf-8",
        "cache-control": "no-cache, no-transform",
      },
    });
  }

  // Non-streaming response.
  const { content: rawContent, citations } =
    await collectUpstream(upstream.body);

  // Rewrite citation markers as Markdown links.
  const content =
    citations.length === 0
      ? rawContent
      : inlineCitationLinks(rawContent, citations) +
        formatCitations(citations);

  const response: ChatCompletionResponse = {
    id,
    object: "chat.completion",
    created,
    model: model.id,
    choices: [
      {
        index: 0,
        message: { role: "assistant", content },
        finish_reason: "stop",
      },
    ],
    usage: estimateUsage(body.messages, rawContent),
  };

  return Response.json(response);
});

// Estimate token usage.
// ChatPlayground does not return token counts.
// Approximately 4 characters per token.
function estimateUsage(
  messages: OpenAIMessage[],
  completion: string,
): ChatCompletionUsage {
  const promptChars = messages.reduce(
    (sum, message) => sum + textChars(message.content),
    0,
  );

  const prompt = Math.ceil(promptChars / 4);
  const comp = Math.ceil(completion.length / 4);

  return {
    prompt_tokens: prompt,
    completion_tokens: comp,
    total_tokens: prompt + comp,
  };
}

// Count text characters in a message.
// Images are excluded from the character count.
function textChars(
  content: OpenAIMessage["content"],
): number {
  if (content == null) {
    return 0;
  }

  if (typeof content === "string") {
    return content.length;
  }

  let n = 0;

  for (const part of content) {
    if (part.type === "text") {
      n += part.text.length;
    }
  }

  return n;
}

export default chat;
