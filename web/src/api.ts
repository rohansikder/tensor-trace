export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function check(response: Response) {
  if (response.ok) return response;
  let message = `The runtime returned HTTP ${response.status}.`;
  try {
    const body = await response.json();
    message =
      typeof body.error === "string"
        ? body.error
        : body.error?.message || body.message || message;
  } catch {
    /* Preserve the useful HTTP status. */
  }
  throw new ApiError(message, response.status);
}

export async function request<T>(
  path: string,
  body?: unknown,
  signal?: AbortSignal,
): Promise<T> {
  const response = await check(
    await fetch(path, {
      method: body === undefined ? "GET" : "POST",
      headers:
        body === undefined ? undefined : { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal,
    }),
  );
  return response.json();
}

export async function stream<T, R>(
  path: string,
  body: unknown,
  signal: AbortSignal,
  onToken: (token: T) => void,
): Promise<R | undefined> {
  const response = await check(
    await fetch(path, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "text/event-stream",
      },
      body: JSON.stringify(body),
      signal,
    }),
  );
  if (!response.body)
    throw new Error("The browser did not receive a response stream.");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let result: R | undefined;
  let completed = false;
  const consume = (block: string) => {
    let event = "message";
    const values: string[] = [];
    for (const line of block.split("\n")) {
      if (line.startsWith("event:")) event = line.slice(6).trim();
      if (line.startsWith("data:")) values.push(line.slice(5).trimStart());
    }
    if (!values.length) return;
    const data = JSON.parse(values.join("\n"));
    if (event === "error")
      throw new Error(
        typeof data.error === "string"
          ? data.error
          : data.message || "Generation stopped with a runtime error.",
      );
    if (event === "token") onToken(data as T);
    if (event === "done") {
      result = data as R;
      completed = true;
    }
  };
  try {
    while (true) {
      const { done, value } = await reader.read();
      buffer += decoder.decode(value, { stream: !done }).replace(/\r\n/g, "\n");
      let split: number;
      while ((split = buffer.indexOf("\n\n")) >= 0) {
        consume(buffer.slice(0, split));
        buffer = buffer.slice(split + 2);
      }
      if (done) break;
    }
    if (buffer.trim()) consume(buffer);
    if (!completed && !signal.aborted)
      throw new Error("The generation stream ended before its final result.");
    return result;
  } finally {
    reader.releaseLock();
  }
}

export const message = (error: unknown) =>
  error instanceof Error
    ? error.message
    : "Something interrupted the runtime request.";
export const isAbort = (error: unknown) =>
  error instanceof DOMException && error.name === "AbortError";
