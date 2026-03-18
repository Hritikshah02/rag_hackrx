import { NextRequest, NextResponse } from 'next/server'

// Max duration: 60s on Vercel Hobby, 300s on Pro
export const maxDuration = 60

const WEBHOOK_API_URL = process.env.WEBHOOK_API_URL || 'http://localhost:8000'
const BEARER_TOKEN =
  process.env.BEARER_TOKEN ||
  '0834b150c8388abe371c886793946844e5847079871db13687754358e06d4b30'

export async function POST(request: NextRequest) {
  let body: unknown
  try {
    body = await request.json()
  } catch {
    return NextResponse.json({ detail: 'Invalid JSON body.' }, { status: 400 })
  }

  try {
    const upstream = await fetch(`${WEBHOOK_API_URL}/hackrx/run`, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${BEARER_TOKEN}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    })

    const data = await upstream.json()
    return NextResponse.json(data, { status: upstream.status })
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Upstream request failed.'
    return NextResponse.json({ detail: message }, { status: 502 })
  }
}
