import { useEffect, useRef } from 'react'

/**
 * Tracks object URLs and revokes them on unmount to prevent memory leaks.
 */
export function useObjectUrlTracker() {
    const urls = useRef<string[]>([])

    useEffect(() => {
        return () => {
            urls.current.forEach(url => URL.revokeObjectURL(url))
            urls.current = []
        }
    }, [])

    const track = (url: string) => {
        urls.current.push(url)
        return url
    }

    const trackAll = (urlList: string[]) => {
        urls.current.push(...urlList)
        return urlList
    }

    const revokeAll = () => {
        urls.current.forEach(url => URL.revokeObjectURL(url))
        urls.current = []
    }

    return { track, trackAll, revokeAll }
}
