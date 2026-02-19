import type { NextConfig } from 'next'

const nextConfig: NextConfig = {
    output: 'export',
    images: {
        unoptimized: true,
    },
    turbopack: {},
    webpack: (config) => {
        config.module.rules.push({
            test: /\.worker\.ts$/,
            use: { loader: 'worker-loader' },
        })
        return config
    },
}

export default nextConfig
