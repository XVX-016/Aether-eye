import type { NextRequest } from "next/server";

export const dynamic = "force-dynamic";

function buildBackendBase(): string {
    const raw = process.env.BACKEND_API_BASE_URL || "http://127.0.0.1:8000/api";
    const trimmed = raw.replace(/\/+$/, "");
    return trimmed.endsWith("/api") ? trimmed : `${trimmed}/api`;
}

function copyHeaders(input: Headers): Headers {
    const out = new Headers();
    input.forEach((value, key) => {
        const lower = key.toLowerCase();
        if (lower === "host" || lower === "content-length" || lower === "connection") {
            return;
        }
        out.set(key, value);
    });
    return out;
}

async function proxy(request: NextRequest, path: string[]) {
    const base = buildBackendBase();
    const query = request.nextUrl.search || "";
    const target = `${base}/${path.join("/")}${query}`;

    const method = request.method.toUpperCase();
    const needsBody = method !== "GET" && method !== "HEAD";
    const init: RequestInit = {
        method,
        headers: copyHeaders(request.headers),
        cache: "no-store",
        body: needsBody ? await request.arrayBuffer() : undefined,
    };

    const upstream = await fetch(target, init);
    const headers = new Headers();
    upstream.headers.forEach((value, key) => {
        const lower = key.toLowerCase();
        if (lower === "content-encoding" || lower === "transfer-encoding") {
            return;
        }
        headers.set(key, value);
    });

    return new Response(upstream.body, {
        status: upstream.status,
        statusText: upstream.statusText,
        headers,
    });
}

type RouteCtx = { params: Promise<{ path: string[] }> };

export async function GET(request: NextRequest, ctx: RouteCtx) {
    const { path } = await ctx.params;
    return proxy(request, path);
}

export async function POST(request: NextRequest, ctx: RouteCtx) {
    const { path } = await ctx.params;
    return proxy(request, path);
}

export async function PUT(request: NextRequest, ctx: RouteCtx) {
    const { path } = await ctx.params;
    return proxy(request, path);
}

export async function PATCH(request: NextRequest, ctx: RouteCtx) {
    const { path } = await ctx.params;
    return proxy(request, path);
}

export async function DELETE(request: NextRequest, ctx: RouteCtx) {
    const { path } = await ctx.params;
    return proxy(request, path);
}

