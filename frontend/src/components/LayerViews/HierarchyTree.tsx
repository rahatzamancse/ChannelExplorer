import React from 'react';
import { Node } from '@types';
import { useAppSelector } from '@hooks';
import { selectAnalysisResult } from '@features/analyzeSlice';
import * as api from '@api';
import * as d3 from 'd3';
import { 
    normalizeHeatmap, 
    filterTopChannels, 
    calculatePairwiseJaccard 
} from '@utils/heatmapProcessing'
import ImageToolTip from '@components/ImageToolTip';

interface HierarchyTreeProps {
    node: Node;
}

type ClusterNode = {
    id: number,
    children: ClusterNode[],
    isLeaf: boolean,
    instanceIds: number[],
    type: 'subclass' | 'superclass' | 'mainclass',
    level?: number,
    name?: string,
    classIdx?: number
}

const HierarchyTree = React.memo(function HierarchyTree({ node }: HierarchyTreeProps) {
    const [heatmap, setHeatmap] = React.useState<number[][]>([])
    const [classLabels, setClassLabels] = React.useState<string[]>([])
    const analyzeResult = useAppSelector(selectAnalysisResult)
    const svgRef = React.useRef<SVGSVGElement>(null)
    const [clusters, setClusters] = React.useState<{labels: number[], centers: number[][], distances: number[], outliers: number[]} | null>(null)
    const width = 200;
    const height = 500;

    const [nodeList, setNodeList] = React.useState<ClusterNode[]>([]);

    // Add state for hoveredNode
    const [hoveredNode, setHoveredNode] = React.useState<ClusterNode | null>(null);

    // Data fetching effect
    React.useEffect(() => {
        if (analyzeResult.examplePerClass === 0) return
        api.getAnalysisHeatmap(node.name).then(setHeatmap)
        api.getLabels().then(setClassLabels)
        api.getCluster(node.name, false, 3).then(setClusters)
    }, [node.name, analyzeResult])
    

    const nClasses = analyzeResult.selectedClasses.length
    const nClusters = clusters?.centers.length || 0

    // Tree computation effect
    React.useEffect(() => {
        if (heatmap.length === 0 || !clusters || nClusters === 0) return
        let newNodeList: ClusterNode[] = analyzeResult.selectedClasses.map((classId, idx) => ({
            id: idx,
            children: [],
            isLeaf: true,
            instanceIds: Array.from(
                {length: analyzeResult.examplePerClass}, 
                (_, i) => idx * analyzeResult.examplePerClass + i
            ),
            type: 'mainclass',
            name: classLabels[classId],
            classIdx: classId
        }))
        let nextNodeId = newNodeList.length;
            
        // Subcluster
        // Create distribution of classes across clusters
        const classDistribution = Array(nClasses).fill(0).map(() => Array(nClusters).fill(0))
        
        // For each class, count instances in each cluster
        for (let classIdx = 0; classIdx < nClasses; classIdx++) {
            const start = classIdx * analyzeResult.examplePerClass
            const end = start + analyzeResult.examplePerClass
            
            // Count which cluster each instance belongs to
            for (let i = start; i < end; i++) {
                const clusterIdx = clusters.labels[i]
                if (clusterIdx >= 0) { // Ignore outliers (-1)
                    classDistribution[classIdx][clusterIdx]++
                }
            }
        }
        
        // For each class, check distribution across clusters
        classDistribution.forEach((distribution, classIdx) => {
            const thisNode = newNodeList.find(node => node.id === classIdx)!
            const total = distribution.reduce((a, b) => a + b, 0);
            const significantClusters = distribution
                .map((count, clusterIdx) => ({count, clusterIdx}))
                .filter(({count}) => (count / total) > 0.1 && count > 2); // More than 10% threshold and at least 3 instances

            // If class has multiple significant clusters, create child nodes
            if (significantClusters.length > 1) {
                // Convert parent to non-leaf
                thisNode.isLeaf = false;
                
                // Create child nodes for each significant cluster
                thisNode.children = significantClusters.map(({clusterIdx}, index) => ({
                    id: nextNodeId++,
                    children: [],
                    isLeaf: true,
                    instanceIds: thisNode.instanceIds.filter(id => clusters.labels[id] === clusterIdx),
                    type: 'subclass'
                } as ClusterNode));
            }
        });
        
        // Supercluster
        const finalHeatmap = filterTopChannels(normalizeHeatmap(heatmap))
        const jDist = calculatePairwiseJaccard(finalHeatmap)
            .map(row => row.map(item => item.similarity))
        
        // Calculate similarity between multiple clusters using their instance IDs
        const calculateSimilarities = (...clusterNodes: ClusterNode[]): number[][] => {
            // Create result matrix of size clusterNodes.length x clusterNodes.length
            const result = Array(clusterNodes.length).fill(0)
                .map(() => Array(clusterNodes.length).fill(0));
            
            // Calculate similarity for each pair of clusters
            for (let i = 0; i < clusterNodes.length; i++) {
                for (let j = i; j < clusterNodes.length; j++) {
                    const instanceIds1 = clusterNodes[i].instanceIds;
                    const instanceIds2 = clusterNodes[j].instanceIds;
                    
                    let sum = 0;
                    let count = 0;
                    
                    // For each instance in first cluster
                    for (let idx1 of instanceIds1) {
                        // For each instance in second cluster
                        for (let idx2 of instanceIds2) {
                            sum += jDist[idx1][idx2];
                            count++;
                        }
                    }
                    
                    // Fill both sides of the symmetric matrix
                    result[i][j] = sum / count;
                    if (i !== j) {
                        result[j][i] = result[i][j];
                    } else {
                        result[i][j] = 1;
                    }
                }
            }
            
            return result;
        }

        while (newNodeList.length > 1) {
            // Calculate similarity matrix between all clusters
            const similarities = calculateSimilarities(...newNodeList);

            // Find most similar pair
            let maxSim = -Infinity;
            let [mergeIdx1, mergeIdx2] = [-1, -1];
            for (let i = 0; i < similarities.length; i++) {
                for (let j = i + 1; j < similarities.length; j++) {
                    if (similarities[i][j] > maxSim) {
                        maxSim = similarities[i][j];
                        mergeIdx1 = i;
                        mergeIdx2 = j;
                    }
                }
            }
            
            // Create new merged node
            const newNode: ClusterNode = {
                id: nextNodeId++,
                children: [newNodeList[mergeIdx1], newNodeList[mergeIdx2]],
                isLeaf: false,
                instanceIds: newNodeList[mergeIdx1].instanceIds.concat(newNodeList[mergeIdx2].instanceIds),
                type: 'superclass'
            };

            // Update node list - remove merged nodes and add new node
            newNodeList = newNodeList.filter((_, idx) => 
                idx !== mergeIdx1 && idx !== mergeIdx2
            ).concat(newNode);
        }

        setNodeList(newNodeList)
    }, [heatmap, analyzeResult, clusters])

    // 4. Visualization effect
    React.useEffect(() => {
        if (!svgRef.current || nodeList.length === 0 || !clusters) return

        d3.select(svgRef.current).selectAll("*").remove()

        const root = d3.hierarchy(nodeList[0])
        const layout = d3.tree<ClusterNode>()
            .size([width - 40, height - 40])

        const treeData = layout(root)

        // Find max depth of nodes with mainClass=true
        let maxDepth = -1;
        treeData.each(node => {
            const depth = node.depth;
            if (node.data.type === 'mainclass' && depth > maxDepth) {
                maxDepth = depth;
            }
        });

        // Assign levels based on depth
        treeData.each(node => { // bfs
            if (node.data.type === 'mainclass') {
                node.data.level = maxDepth;
            }
            else {
                node.data.level = node.parent ? node.parent.data.level! + 1 : 0;
            }
        });
        
        treeData.children![0].x = -5
        
        const yGap = 60;
        
        treeData.each(node => {
            node.y = node.data.level! * yGap;
            node.x += 10;
        });
        
        
        const svg = d3.select(svgRef.current)
            .append("g")
            .attr("transform", `translate(20,20)`)

        // Add links
        svg.selectAll("path.link")
            .data(treeData.links())
            .enter()
            .append("path")
            .attr("class", "link")
            .attr("fill", "none")
            .attr("stroke", d => d.source.data.type === 'mainclass' ? "#4CAF50" : "#ccc")
            .attr("stroke-width", d => {
                if (d.source.data.type === 'mainclass') {
                    // Calculate total instances in child nodes
                    const childInstances = d.target.data.instanceIds.length;
                    const parentInstances = d.source.data.instanceIds.length;
                    // Width proportional to percentage of parent's instances
                    return (childInstances / parentInstances) * 5;
                }
                return 4;
            })
            .attr("d", d => d3.linkVertical()({
                source: [d.source.x!, d.source.y!],
                target: [d.target.x!, d.target.y!]
            }))

        const nodes = svg.selectAll("g.node")
            .data(treeData.descendants())
            .enter()
            .append("g")
            .attr("class", "node")
            .attr("transform", d => `translate(${d.x},${d.y})`)

        // Add circles for nodes
        nodes.append("circle")
            .attr("r", 8)
            .attr("fill", d => (d.data as ClusterNode).type === 'mainclass' ? "#4CAF50" : d.data.type === 'subclass' ? "#2196F3" : "#FF9800")
            .attr("stroke", d => (d.data as ClusterNode).id === hoveredNode?.id ? "black" : "none")
            .attr("data-tooltip-id", "image-tooltip")
            .on("mouseenter", (event, d) => setHoveredNode(d.data))
            .on("mouseleave", () => setHoveredNode(null));

        // Add labels
        nodes.append("text")
            .attr("dy", "-0.7em")
            .attr("x", 0)
            .attr("text-anchor", "middle")
            .attr("transform", "rotate(-30)")
            .text(d => {
                const nodeData = d.data as ClusterNode
                if (nodeData.type === 'mainclass') {
                    return nodeData.classIdx || ""
                }
                return ""
            })
            .style("font-size", "14px")
        
        // Add horizontal lines through main class nodes
        nodes.filter(d => (d.data as ClusterNode).type === 'mainclass')
            .append("line")
            .attr("x1", -40)
            .attr("y1", 0)
            .attr("x2", 40) 
            .attr("y2", 0)
            .attr("stroke", "#4CAF50")
            .attr("stroke-width", 3)
            .attr("stroke-opacity", 0.4)
            .attr("stroke-dasharray", "4,4");

        // Add text labels
        // svg.append("text")
        //     .attr("x", 20)
        //     .attr("y", height/2 - 10)
        //     .text("superclass")
        //     .style("font-size", "12px");

        // svg.append("text")
        //     .attr("x", 20) 
        //     .attr("y", height/2 + 10)
        //     .text("subclass")
        //     .style("font-size", "12px");
            //     

        // Add legend
        svg.append("g")
            .attr("transform", `translate(-14, 0)`)
            .call(g => {
                // Main class
                g.append("circle")
                    .attr("r", 6)
                    .attr("fill", "#4CAF50")
                    .attr("cx", 0)
                    .attr("cy", 0);
                g.append("text")
                    .attr("x", 15)
                    .attr("y", 4)
                    .text("Main class")
                    .style("font-size", "12px");

                // Subclass
                g.append("circle") 
                    .attr("r", 6)
                    .attr("fill", "#2196F3")
                    .attr("cx", 0)
                    .attr("cy", 20);
                g.append("text")
                    .attr("x", 15)
                    .attr("y", 24)
                    .text("Subclass")
                    .style("font-size", "12px");

                // Superclass
                g.append("circle")
                    .attr("r", 6) 
                    .attr("fill", "#FF9800")
                    .attr("cx", 0)
                    .attr("cy", 40);
                g.append("text")
                    .attr("x", 15)
                    .attr("y", 44)
                    .text("Superclass")
                    .style("font-size", "12px");
            });



    }, [nodeList, classLabels, analyzeResult.selectedClasses, width, height, clusters, hoveredNode])
    

    return <>
        <div style={{display: "flex", flexDirection: "column", alignItems: "center"}}>
            <svg ref={svgRef} width={width} height={height}></svg>
            <button 
                onClick={() => api.getCluster(node.name, false, 3).then(setClusters)}
                style={{
                    marginTop: "10px",
                    padding: "8px 16px",
                    backgroundColor: "#4CAF50",
                    color: "white",
                    border: "none",
                    borderRadius: "4px",
                    cursor: "pointer",
                    fontSize: "14px",
                    transition: "background-color 0.2s"
                }}
                onMouseOver={(e) => e.currentTarget.style.backgroundColor = "#45a049"}
                onMouseOut={(e) => e.currentTarget.style.backgroundColor = "#4CAF50"}
            >
                Resubclass
            </button>
            {hoveredNode && hoveredNode.instanceIds.length > 0 &&
                <ImageToolTip
                    imgs={[...hoveredNode.instanceIds]
                        .sort(() => Math.random() - 0.5)
                        .slice(0, 9)} // Show 9 random images
                    imgType={'raw'}
                    imgData={{}}
                    label={`${hoveredNode.instanceIds.length} images`}
                />
            }
        </div>
    </>
})

export default HierarchyTree;
