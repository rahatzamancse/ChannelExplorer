import { AnalysisConfig } from "@features/analyzeSlice";
import { ModelGraph, Node } from "@types";

const API_URL = process.env.NEXT_PUBLIC_API_URL || process.env.REACT_APP_API_URL || "http://localhost:8000/api"
export const STATIC_MODE = process.env.NEXT_PUBLIC_STATIC_MODE === 'true'
const SD = '/static-data'

function parseModelGraphResponse(data: any): ModelGraph {
    const graph = data?.graph ?? {};
    const nodes = Array.isArray(graph?.nodes) ? graph.nodes : [];
    const links = Array.isArray(graph?.edges) ? graph.edges : Array.isArray(graph?.links) ? graph.links : [];
    const edgeWeights = data?.edge_weights ?? {};
    return {
        nodes: nodes.map((node: any) => ({
            id: node.id,
            label: node.label,
            layer_type: node.layer_type,
            name: node.name,
            input_shape: node.input_shape,
            kernel_size: node.kernel_size,
            output_shape: node.output_shape,
            tensor_type: node.tensor_type,
            pos: node.pos ? { x: node.pos.x, y: node.pos.y } : undefined,
            out_edge_weight: edgeWeights[node.name],
        })),
        edges: links.map((edge: any) => ({
            source: edge.source,
            target: edge.target,
        })),
        meta: {
            depth: data?.meta?.depth ?? 0,
        },
    };
}

export function getModelGraph(): Promise<ModelGraph> {
    if (STATIC_MODE) {
        return fetch(`${SD}/model.json`).then(r => r.json()).then(parseModelGraphResponse)
    }
    return fetch(`${API_URL}/model/`, {
        method: "GET",
        headers: { "Content-Type": "application/json" },
    })
        .then(response => response.json())
        .then(parseModelGraphResponse)
}

export function saveDataset(): Promise<string> {
    if (STATIC_MODE) return Promise.resolve("")
    return fetch(`${API_URL}/analysis/images/save`, {
        method: "POST",
    })
        .then(response => response.status === 200 ? response.blob() : Promise.reject(response))
        .then(blob => {
            const url = URL.createObjectURL(blob)
            const a = document.createElement('a')
            a.style.display = 'none'
            a.href = url
            a.download = 'dataset.zip'
            a.click()
            return url
        })
        .catch(err => {
            console.error(err)
            return ""
        })
}

export function getFeatureActivatedChannels(): Promise<{activated_channels: {[layer: string]: number[]}}> {
    if (STATIC_MODE) return Promise.resolve({ activated_channels: {} })
    return fetch(`${API_URL}/polygon/activated_channels`, {
        method: "GET",
        headers: { "Content-Type": "application/json" },
    })
        .then(response => response.json())
        .then(data => data)
}

export function getLabels(): Promise<string[]> {
    if (STATIC_MODE) {
        return fetch(`${SD}/labels.json`).then(r => r.json()).then(data => Array.isArray(data) ? data : [])
    }
    return fetch(`${API_URL}/labels/`, {
        method: "GET",
        headers: { "Content-Type": "application/json" },
    })
        .then(response => response.json())
        .then(data => (Array.isArray(data) ? data : []))
}

export function getFeatureHuntImage(): Promise<string> {
    if (STATIC_MODE) return Promise.resolve("")
    return fetch(`${API_URL}/polygon/getimage`, {
        method: "GET",
        headers: { "Content-Type": "application/json" },
    })
        .then(response => response.blob())
        .then(blob => URL.createObjectURL(blob))
}

export function submitBoxSelection(points: {x: number, y:number}[]) {
    if (STATIC_MODE) return Promise.resolve({})
    return fetch(`${API_URL}/polygon/points`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(points)
    })
        .then(response => response.json())
        .then(data => data)
}

export function submitFeatureHuntImage(file: File): Promise<string[]> {
    if (STATIC_MODE) return Promise.resolve([])
    const formData = new FormData()
    formData.append("file", file)
    return fetch(`${API_URL}/polygon/image`, {
        method: "POST",
        body: formData,
    })
        .then(response => response.json())
        .then(data => data)
}

export function getCluster(layer: string, useXMeans: boolean, kClusters?: number): Promise<{ labels: number[], centers: number[][], distances: number[], outliers: number[]}> {
    if (STATIC_MODE) {
        const file = (!useXMeans && kClusters === 3) ? 'cluster_kmeans3.json' : 'cluster_xmeans.json'
        return fetch(`${SD}/layers/${layer}/${file}`).then(r => r.json())
    }
    return fetch(`${API_URL}/analysis/layer/${layer}/cluster?` + new URLSearchParams({
        outlier_threshold: '2',
        use_xmeans: useXMeans.toString(),
        k_clusters: kClusters?.toString() || '-1',
    }))
        .then(response => response.json())
        .then(data => data)
}

export function getActivationsImages(node: Node, startFilter: number, nFilters: number, nImgs: number): Promise<string[][]> {
    const imgLayerTypes = ["Conv2D", "MaxPooling2D", "AveragePooling2D", "Conv2d", "Cat", "Add", "Concatenate"]
    if(!imgLayerTypes.includes(node.layer_type)) {
        return Promise.resolve([])
    }

    if (STATIC_MODE) {
        const result: string[][] = []
        Array.from(Array(nFilters).keys(), x => x + startFilter).forEach(filterIdx => {
            result.push(
                Array.from(Array(nImgs).keys()).map(imgIdx =>
                    `${SD}/images/activations/${node.name}/${imgIdx}_${filterIdx}.png`
                )
            )
        })
        return Promise.resolve(result)
    }
    
    const promises: Promise<string>[][] = []
    Array.from(Array(nFilters).keys(), x => x + startFilter).forEach(filterIdx => 
        promises.push(
            Array.from(Array(nImgs).keys()).map(imgIdx =>
                fetch(`${API_URL}/analysis/image/${imgIdx}/layer/${node.name}/filter/${filterIdx}`)
                .then(response => response.blob())
                .then(blob => URL.createObjectURL(blob))
            )
        )
    )
    
    return Promise.all(promises.map(p => Promise.all(p)))
}

export function getAnalysisLayerCoords(node: string, method: string = 'mds', distance: string = 'euclidean', normalization: string = 'none', takeSummary: boolean = true): Promise<[number, number][]> {
    if (STATIC_MODE) {
        return fetch(`${SD}/layers/${node}/embedding.json`).then(r => r.json())
    }
    return fetch(`${API_URL}/analysis/layer/${node}/embedding?` + new URLSearchParams({
        normalization: normalization,
        method: method,
        distance: distance,
        take_summary: takeSummary.toString(),
    }))
        .then(response => response.json())
        .then(data => data)
}

export function getAnalysisDistanceMatrix(node: string): Promise<number[][]> {
    if (STATIC_MODE) {
        return fetch(`${SD}/layers/${node}/distance.json`).then(r => r.json())
    }
    return fetch(`${API_URL}/analysis/layer/${node}/embedding/distance`)
        .then(response => response.json())
        .then(data => data)
}

export function getAllDistances(): Promise<number[][]> {
    if (STATIC_MODE) {
        return fetch(`${SD}/alldistances.json`).then(r => r.json())
    }
    return fetch(`${API_URL}/analysis/alldistances`)
        .then(response => response.json())
        .then(data => data)
}

export function getAnalysisHeatmap(node: string): Promise<number[][]> {
    if (STATIC_MODE) {
        return fetch(`${SD}/layers/${node}/heatmap.json`).then(r => r.json())
    }
    return fetch(`${API_URL}/analysis/layer/${node}/heatmap`)
        .then(response => response.json())
        .then(data => data)
}

export function analyze(labels: number[], examplePerClass: number, shuffled: boolean): Promise<{ message: string, task_id: string }> {
    if (STATIC_MODE) {
        return Promise.resolve({ message: 'Static demo', task_id: 'static' })
    }
    return fetch(`${API_URL}/analysis?examplePerClass=${examplePerClass}&shuffle=${shuffled}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(labels)
    })
    .then(response => response.json())
    .then(data => data)
}

export function getTaskStatus(task_id: string): Promise<{ message: string, task_id: string, payload: null | AnalysisConfig }> {
    if (STATIC_MODE) {
        return fetch(`${SD}/config.json`).then(r => r.json()).then(data => ({
            message: 'done',
            task_id: 'static',
            payload: {
                selectedClasses: data.selectedClasses,
                examplePerClass: data.examplePerClass,
                selectedImages: [],
                shuffled: data.shuffled,
                predictions: data.predictions,
            },
        }))
    }
    return fetch(`${API_URL}/taskStatus?task_id=${task_id}`, {
        method: "GET",
        headers: { "Content-Type": "application/json" },
    })
    .then(response => response.json())
    .then(data => data)
}

export function getPredictions(): Promise<number[]> {
    if (STATIC_MODE) {
        return fetch(`${SD}/predictions.json`).then(r => r.json())
    }
    return fetch(`${API_URL}/analysis/predictions`)
        .then(response => response.json())
        .then(data => data)
}

export function getDenseArgmax(layer: string): Promise<number[]> {
    if (STATIC_MODE) {
        return fetch(`${SD}/layers/${layer}/argmax.json`).then(r => r.json())
    }
    return fetch(`${API_URL}/analysis/layer/${layer}/argmax`)
        .then(response => response.json())
        .then(data => data)
}


export function getInputImages(imgIdxs: number[]): Promise<string[]> {
    if(imgIdxs.length === 0) {
        return Promise.resolve([])
    }
    if (STATIC_MODE) {
        return Promise.resolve(imgIdxs.map(i => `${SD}/images/input/${i}.png`))
    }
    return Promise.all(imgIdxs.map(
        i => fetch(`${API_URL}/analysis/images/${i}`, {
            method: "GET",
            headers: { "Content-Type": "application/json" },
        })
            .then(response => response.blob())
            .then(blob => URL.createObjectURL(blob))
        ))
}

export function getInputImageURL(imgIdx: number): string {
    if (STATIC_MODE) return `${SD}/images/input/${imgIdx}.png`
    return `${API_URL}/analysis/images/${imgIdx}`
}

export function getActivationOverlay(imgIdxs: number[], node: string, channel: number): Promise<string[]> {
    if (STATIC_MODE) {
        return Promise.resolve(imgIdxs.map(i => `${SD}/images/overlays/${node}/${channel}_${i}.png`))
    }
    return Promise.all(imgIdxs.map(
        i => fetch(`${API_URL}/analysis/layer/${node}/${channel}/heatmap/${i}`, {
            method: "GET",
            headers: { "Content-Type": "application/json" },
        })
            .then(response => response.blob())
            .then(blob => URL.createObjectURL(blob))))
}

export function getKernel(node: string, channel: number): Promise<string> {
    if (STATIC_MODE) {
        return Promise.resolve(`${SD}/images/kernels/${node}/${channel}.png`)
    }
    return fetch(`${API_URL}/analysis/layer/${node}/${channel}/kernel`, {
        method: "GET",
        headers: { "Content-Type": "application/json" },
    })
        .then(response => response.blob())
        .then(blob => URL.createObjectURL(blob))
}

export function getAllEmbedding(): Promise<[number, number][]> {
    if (STATIC_MODE) {
        return fetch(`${SD}/allembedding.json`).then(r => r.json())
    }
    return fetch(`${API_URL}/analysis/allembedding`)
        .then(response => response.json())
        .then(data => data)
}

export function getConfiguration(): Promise<AnalysisConfig> {
    if (STATIC_MODE) {
        return fetch(`${SD}/config.json`).then(r => r.json()).then(data => ({
            selectedClasses: data.selectedClasses,
            examplePerClass: data.examplePerClass,
            selectedImages: [],
            shuffled: data.shuffled,
            predictions: data.predictions,
        }))
    }
    return fetch(`${API_URL}/loaded_analysis`, {
        method: "GET",
        headers: { "Content-Type": "application/json" },
    })
        .then(response => response.json())
        .then(data => ({
            selectedClasses: data.selectedClasses,
            examplePerClass: data.examplePerClass,
            selectedImages: [],
            shuffled: data.shuffled,
            predictions: data.predictions,
        }))
}
