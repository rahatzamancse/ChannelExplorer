'use client'

import React from 'react'
import { Provider } from 'react-redux'
import { store } from '@/app/store'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { TourProvider } from '@reactour/tour'
import { tutorialSteps } from '@/tutorialSteps'
import Navigation from '@components/Navigation'

import 'bootstrap/dist/css/bootstrap.min.css'
import '@styles/index.css'

const queryClient = new QueryClient({
    defaultOptions: {
        queries: {
            refetchOnWindowFocus: false,
            retry: 1,
        },
    },
})

if (typeof window !== 'undefined') {
    const resizeObserverErr = (e: ErrorEvent) => {
        if (e.message === 'ResizeObserver loop completed with undelivered notifications.') {
            e.stopImmediatePropagation()
            e.preventDefault()
        }
    }
    window.addEventListener('error', resizeObserverErr, true)
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
    return (
        <html lang="en" suppressHydrationWarning>
            <body suppressHydrationWarning>
                <QueryClientProvider client={queryClient}>
                    <Provider store={store}>
                        <TourProvider steps={tutorialSteps}>
                            <Navigation />
                            {children}
                        </TourProvider>
                    </Provider>
                </QueryClientProvider>
            </body>
        </html>
    )
}
