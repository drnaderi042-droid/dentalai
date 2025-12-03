import { NextResponse } from 'next/server'

export function middleware(request) {
  // Check if the request is using HTTP and redirect to HTTPS
  if (request.headers.get('x-forwarded-proto') === 'http') {
    const url = request.nextUrl.clone()
    url.protocol = 'https'
    return NextResponse.redirect(url)
  }

  // For local development, you might want to skip this
  // Uncomment the following lines if you want to skip redirection in development
  /*
  if (process.env.NODE_ENV === 'development') {
    return NextResponse.next()
  }
  */

  return NextResponse.next()
}

// Configure which paths this middleware should run on
export const config = {
  matcher: [
    /*
     * Match all request paths except for the ones starting with:
     * - api (API routes)
     * - _next/static (static files)
     * - _next/image (image optimization files)
     * - favicon.ico (favicon file)
     */
    '/((?!api|_next/static|_next/image|favicon.ico).*)',
  ],
}
