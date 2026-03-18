import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Agentic Document QA System',
  description: 'Agentic RAG pipeline — accurate answers from any document in under 3 seconds.',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className="dark">
      <body className="scanlines">{children}</body>
    </html>
  )
}
