"""
Simple health check endpoint for Vercel
"""

def handler(request):
    """Vercel serverless function handler"""
    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        },
        'body': '{"status": "healthy", "version": "vercel-serverless", "dependencies": "zero-rust"}'
    }
