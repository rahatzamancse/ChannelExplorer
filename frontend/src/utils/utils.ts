export const NodeColors: { [key: string]: string } = {
    'Conv2D': '#CBE4F9',
    'Dense': '#CDF5F6',
    'InputLayer': '#EFF9DA',
    'MaxPooling2D': '#FFFF00',
    'Flatten': '#F9EBDF',
    'Dropout': '#FF00FF',
    'Activation': '#FF8000',
    'GlobalAveragePooling2D': '#8000FF',
    'GlobalMaxPooling2D': '#0080FF',
    'BatchNormalization': '#D6CDEA',
    'Add': '#80FF00',
    'Concatenate': '#F9D8D6',
    'AveragePooling2D': '#800080',
    'ZeroPadding2D': '#808000',
    'UpSampling2D': '#008000',
    'Reshape': '#800000',
    'Permute': '#000080',
    'RepeatVector': '#808080',
    'Lambda': '#008080',
}

export function chunkify<T>(arr: T[], size: number): T[][] {
    return [...Array(Math.ceil(arr.length / size))].map((_, i) =>
        arr.slice(size * i, size + size * i)
    );
}

export function transposeArray<T>(array: T[][]): T[][] {
    return array[0].map((_, j) =>
        array.map((row) => row[j])
    );
}
export function findIndicesOfMax(inp: number[], count: number) {
    var outp = [];
    for (var i = 0; i < inp.length; i++) {
        outp.push(i); // add index to output array
        if (outp.length > count) {
            outp.sort(function(a, b) { return inp[b] - inp[a]; }); // descending sort the output array
            outp.pop(); // remove the last index (index of smallest element in output array)
        }
    }
    return outp;
}
export function calcAllPairwiseDistance(arr: number[]) {
    let sum = 0
    for (let i = 0; i < arr.length; i++) {
        for (let j = i + 1; j < arr.length; j++) {
            sum += Math.abs(arr[i] - arr[j])
        }
    }
    return sum
}

export function calcVariance(inp: number[]) {
    const mean = inp.reduce((a, b) => a + b, 0) / inp.length
    const variance = inp.map(item => Math.pow(item - mean, 2)).reduce((a, b) => a + b, 0) / inp.length
    return variance
}
export function calcPairwiseDistance(arr1: number[], arr2: number[]) {
    let sum = 0
    for (let i = 0; i < arr1.length; i++) {
        sum += Math.pow(arr1[i] - arr2[i], 2)
    }
    return Math.sqrt(sum)
}

export function calcSumPairwiseDistance(...arrs: number[][]): number {
    let sum = 0
    for (let i = 0; i < arrs.length; i++) {
        for (let j = i + 1; j < arrs.length; j++) {
            sum += calcPairwiseDistance(arrs[i], arrs[j])
        }
    }
    return sum
}

export function getRawHeatmap(heatmap: number[][], nExamples: number, nClasses: number) {
    return heatmap.slice(0, nExamples)
}

export function shortenName(name: string | unknown, len: number): string {
    let s: string
    if (typeof name !== 'string') {
        s = Array.isArray(name) ? (name[0] ?? '') : String(name ?? '')
    } else {
        s = name
    }
    s = s.split(": ")[0]
    return s.length <= len ? s : s.slice(0, len) + '...'
}
  
