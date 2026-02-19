import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import * as api from '@api'
import { Node } from '@types'

export function useLabels() {
    return useQuery({
        queryKey: ['labels'],
        queryFn: api.getLabels,
        staleTime: Infinity,
    })
}

export function useModelGraph() {
    return useQuery({
        queryKey: ['modelGraph'],
        queryFn: api.getModelGraph,
        staleTime: Infinity,
    })
}

export function useAnalysisHeatmap(nodeName: string, enabled: boolean) {
    return useQuery({
        queryKey: ['heatmap', nodeName],
        queryFn: () => api.getAnalysisHeatmap(nodeName),
        enabled,
        staleTime: 5 * 60 * 1000,
    })
}

export function useAnalysisLayerCoords(nodeName: string, method = 'umap', distance = 'euclidean', normalization = 'none', takeSummary = true) {
    return useQuery({
        queryKey: ['layerCoords', nodeName, method, distance, normalization, takeSummary],
        queryFn: () => api.getAnalysisLayerCoords(nodeName, method, distance, normalization, takeSummary),
        staleTime: 5 * 60 * 1000,
    })
}

export function usePredictions() {
    return useQuery({
        queryKey: ['predictions'],
        queryFn: api.getPredictions,
        staleTime: 5 * 60 * 1000,
    })
}

export function useDistanceMatrix(nodeName: string, enabled: boolean) {
    return useQuery({
        queryKey: ['distanceMatrix', nodeName],
        queryFn: () => api.getAnalysisDistanceMatrix(nodeName),
        enabled,
        staleTime: 5 * 60 * 1000,
    })
}

export function useCluster(nodeName: string, useXMeans: boolean, kClusters: number, enabled: boolean) {
    return useQuery({
        queryKey: ['cluster', nodeName, useXMeans, kClusters],
        queryFn: () => api.getCluster(nodeName, useXMeans, kClusters),
        enabled,
        staleTime: 5 * 60 * 1000,
    })
}

export function useTaskStatus(taskId: string | null) {
    return useQuery({
        queryKey: ['taskStatus', taskId],
        queryFn: () => api.getTaskStatus(taskId!),
        enabled: !!taskId,
        refetchInterval: (query) => {
            if (query.state.data?.payload !== null) return false
            return 1000
        },
    })
}

export function useConfiguration() {
    return useQuery({
        queryKey: ['configuration'],
        queryFn: api.getConfiguration,
        staleTime: 30 * 1000,
    })
}

export function useDenseArgmax(nodeName: string, layerType: string) {
    return useQuery({
        queryKey: ['denseArgmax', nodeName],
        queryFn: () => api.getDenseArgmax(nodeName),
        enabled: layerType === 'Dense' || layerType === 'Linear',
        staleTime: 5 * 60 * 1000,
    })
}
