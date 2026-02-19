type TransposeArray = <T>(array: T[][]) => T[][]
const transposeArray: TransposeArray = (array) => array[0].map((_, j) => array.map((row) => row[j]))

function findIndicesOfMax(inp: number[], count: number): number[] {
    const outp: number[] = []
    for (let i = 0; i < inp.length; i++) {
        outp.push(i)
        if (outp.length > count) {
            outp.sort((a, b) => inp[b] - inp[a])
            outp.pop()
        }
    }
    return outp
}

function calcAllPairwiseDistance(arr: number[]): number {
    let sum = 0
    for (let i = 0; i < arr.length; i++) {
        for (let j = i + 1; j < arr.length; j++) {
            sum += Math.abs(arr[i] - arr[j])
        }
    }
    return sum
}

function calcVariance(inp: number[]): number {
    const mean = inp.reduce((a, b) => a + b, 0) / inp.length
    return inp.map(item => Math.pow(item - mean, 2)).reduce((a, b) => a + b, 0) / inp.length
}

function calcPairwiseDistance(arr1: number[], arr2: number[]): number {
    let sum = 0
    for (let i = 0; i < arr1.length; i++) {
        sum += Math.pow(arr1[i] - arr2[i], 2)
    }
    return Math.sqrt(sum)
}

function calcSumPairwiseDistance(...arrs: number[][]): number {
    let sum = 0
    for (let i = 0; i < arrs.length; i++) {
        for (let j = i + 1; j < arrs.length; j++) {
            sum += calcPairwiseDistance(arrs[i], arrs[j])
        }
    }
    return sum
}

function chunkify<T>(arr: T[], size: number): T[][] {
    return [...Array(Math.ceil(arr.length / size))].map((_, i) =>
        arr.slice(size * i, size + size * i)
    )
}

interface JaccardResult {
    intersection: number
    union: number
    similarity: number
}

function normalizeHeatmap(heatmap: number[][]): number[][] {
    return transposeArray(transposeArray(heatmap).map(row => {
        const mean = row.reduce((a, b) => a + b, 0) / row.length
        const meanShiftedRow = row.map(item => item - mean)
        const max = Math.max(...meanShiftedRow)
        const min = Math.min(...meanShiftedRow)
        return meanShiftedRow.map(item => (item - min) / (max - min))
    }))
}

function filterTopChannels(heatmap: number[][]): number[][] {
    const TOTAL_MAX_CHANNELS = (arr: number[]) => arr.length * 0.5
    const indicesMax = heatmap.map(arr => findIndicesOfMax(arr, TOTAL_MAX_CHANNELS(arr)))
    return heatmap.map((col, i) => {
        const newCol = [...col]
        col.forEach((_, j) => {
            if (!indicesMax[i].includes(j)) newCol[j] = 0
        })
        return newCol
    })
}

function calculateJaccardSimilarity(col1: number[], col2: number[]): JaccardResult {
    const intersection = col1
        .map((item, k) => (item > 0 && col2[k] > 0))
        .reduce((total, x) => total + (x ? 1 : 0), 0)
    const union = col1
        .map((item, k) => (item > 0 || col2[k] > 0))
        .reduce((total, x) => total + (x ? 1 : 0), 0)
    return { intersection, union, similarity: intersection / union }
}

function calculatePairwiseJaccard(heatmap: number[][]): JaccardResult[][] {
    return heatmap.map((col1, i) =>
        heatmap.map((col2, j) => {
            if (i === j) return { intersection: 1, union: 1, similarity: 1 }
            return calculateJaccardSimilarity(col1, col2)
        })
    )
}

self.onmessage = (e: MessageEvent) => {
    const { type, payload } = e.data

    if (type === 'computeJaccard') {
        const { heatmap } = payload
        const finalHeatmap = filterTopChannels(normalizeHeatmap(heatmap))
        const jDist = calculatePairwiseJaccard(finalHeatmap)
        self.postMessage({ type: 'jaccardResult', payload: jDist })
    }

    if (type === 'computeHeatmapProcessing') {
        const { heatmap, normalizeRow, examplePerClass, selectedClasses, sortBy, layerType, outEdgeWeight, totalMaxFraction } = payload as {
            heatmap: number[][], normalizeRow: boolean, examplePerClass: number,
            selectedClasses: number[], sortBy: string, layerType: string,
            outEdgeWeight: number[], totalMaxFraction: number
        }
        const nExamples = examplePerClass * selectedClasses.length

        const normalHeatmap = normalizeRow ? transposeArray(transposeArray(heatmap).map(row => {
            const mean = row.reduce((a, b) => a + b, 0) / row.length
            const shifted = row.map(item => item - mean)
            const max = Math.max(...shifted)
            const min = Math.min(...shifted)
            return (max - min === 0) ? shifted.map(() => 0) : shifted.map(item => (item - min) / (max - min))
        })) : heatmap.map((col: number[]) => {
            const mean = col.reduce((a, b) => a + b, 0) / col.length
            const shifted = col.map((item: number) => item - mean)
            const max = Math.max(...shifted)
            const min = Math.min(...shifted)
            return (max - min === 0) ? shifted.map(() => 0) : shifted.map((item: number) => (item - min) / (max - min))
        })

        const rawHeatmap = normalHeatmap.slice(0, nExamples)
        const h1 = transposeArray(transposeArray(rawHeatmap).map(row => [...row, calcVariance(row)]))

        const h2 = transposeArray(
            transposeArray(h1).map(row => [...row, (selectedClasses.length > 1 ? calcSumPairwiseDistance : calcAllPairwiseDistance)(
                ...chunkify(row.slice(0, nExamples), examplePerClass)
            )])
        )

        const maxFraction = totalMaxFraction || 0.2
        const indicesMax = h2.map((arr: number[]) => findIndicesOfMax(arr, arr.length * maxFraction))
        h2.forEach((col: number[], i: number) => {
            col.forEach((_: number, j: number) => {
                if (!indicesMax[i].includes(j)) col[j] = 0
            })
        })

        const h3 = transposeArray(
            transposeArray(h2).map(row => [
                ...row,
                calcAllPairwiseDistance(
                    selectedClasses.length > 1 ?
                        chunkify(row.slice(0, nExamples), examplePerClass)
                            .map((r: number[]) => r.reduce((prev: number, curr: number) => prev + (curr > 0 ? 1 : 0), 0)) :
                        row.slice(0, nExamples)
                )
            ])
        )

        let h4 = h3
        if (layerType === 'Conv2D' && outEdgeWeight) {
            h4 = transposeArray(transposeArray(h3).map((row: number[], i: number) => [...row, outEdgeWeight[i]]))
        }

        let SORT_BY = 0
        if (sortBy === 'count') SORT_BY = 2
        else if (sortBy === 'edge_weight') SORT_BY = 1
        else if (sortBy === 'pairwise') SORT_BY = 3
        else if (sortBy === 'variance') SORT_BY = 4

        const TOP_N = 40
        const finalHeatmapAll = SORT_BY === 0 ? h4 : transposeArray(transposeArray(h4).sort((a: number[], b: number[]) => b[b.length - SORT_BY] - a[a.length - SORT_BY]))
        const finalHeatmap = finalHeatmapAll.map((col: number[]) => col.slice(0, TOP_N))

        self.postMessage({ type: 'heatmapProcessingResult', payload: finalHeatmap })
    }
}

export {}
