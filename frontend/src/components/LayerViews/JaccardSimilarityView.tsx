import React from 'react'
import { Node } from '@types'
import { useAppSelector } from '@hooks'
import { selectAnalysisResult } from '@features/analyzeSlice'
import * as api from '@api'
import { shortenName } from '@utils/utils'
import * as d3 from 'd3'
import ImageToolTip from '@components/ImageToolTip'
import { 
    normalizeHeatmap, 
    filterTopChannels, 
    calculatePairwiseJaccard 
} from '@utils/heatmapProcessing'

const JACCARD_SVG_PADDING = { top: 10, right: 10, bottom: 10, left: 10 }

const JaccardSimilarityView = React.memo(function JaccardSimilarityView({ node, width, height }: { node: Node, width: number, height: number }) {
    const [heatmap, setHeatmap] = React.useState<number[][]>([])
    const svgRef = React.useRef<SVGSVGElement>(null)
    const analyzeResult = useAppSelector(selectAnalysisResult)
    const [hoveredItem, setHoveredItem] = React.useState<[number, number]>([-1, -1])
    const [classLabels, setClassLabels] = React.useState<string[]>([])

    const svgPadding = JACCARD_SVG_PADDING

    React.useEffect(() => {
        if (analyzeResult.examplePerClass === 0) return
        if (['Conv2D', 'Concatenate', 'Dense', 'Conv2d', 'Cat', 'Linear', 'Add'].some(l => node.layer_type.includes(l))) {
            api.getAnalysisHeatmap(node.name).then(setHeatmap)
        }
        api.getLabels().then(setClassLabels)
    }, [node.name, analyzeResult])

    const computed = React.useMemo(() => {
        if (heatmap.length === 0) return null
        const nExamples = analyzeResult.examplePerClass * analyzeResult.selectedClasses.length
            
        const finalHeatmap = filterTopChannels(normalizeHeatmap(heatmap))
        const jDist = calculatePairwiseJaccard(finalHeatmap)
        
        const minJDist = Math.min(...jDist.map(col => Math.min(...col.map(item => item.similarity))))
        const maxJDistWithout1 = Math.max(...jDist.map(col => Math.max(...col.map(item => 
            item.similarity === 1 ? 0 : item.similarity))))

        const jDistColorScale = d3.scaleLinear()
            .domain([minJDist, maxJDistWithout1])
            .range([0, 1])
            .clamp(true)

        const jDistColors = jDist.map(col => col.map(item => d3.interpolateBlues(jDistColorScale(item.similarity))))
        
        const cellWidth = (width - svgPadding.left - svgPadding.right) / nExamples
        const cellHeight = (height - svgPadding.top - svgPadding.bottom) / nExamples

        const labelScale = d3.scaleLinear()
            .domain([0, analyzeResult.selectedClasses.length - 1])
            .range([
                svgPadding.left + (cellWidth * analyzeResult.examplePerClass) / 2,
                width - (cellWidth * analyzeResult.examplePerClass) / 2 - svgPadding.right,
            ])

        return { nExamples, jDist, jDistColors, cellWidth, cellHeight, labelScale }
    }, [heatmap, analyzeResult, width, height, svgPadding])

    if (!computed) return null
    const { nExamples: _nExamples, jDist, jDistColors, cellWidth, cellHeight, labelScale } = computed
        
        
    
    return <>
        <svg width={width+20} height={height+20} ref={svgRef} style={{
            backgroundColor: "white"
        }}>
            <defs>
                <pattern id="diagonalHatch" patternUnits="userSpaceOnUse" width="4" height="4">
                    <path d="M-1,1 l2,-2
                            M0,4 l4,-4
                            M3,5 l2,-2"
                        style={{stroke:'black', strokeWidth:1}}
                    />
                </pattern>
            </defs>
            <g transform='translate(20 20)'>
            <g>
                {jDistColors.map((col, i) => col.map((elem, j) => (
                    <rect
                        key={`${i}-${j}`}
                        x={i * cellWidth + svgPadding.left}
                        y={j * cellHeight + svgPadding.top}
                        width={cellWidth}
                        height={cellHeight}
                        fill={i === j?"url(#diagonalHatch)":elem}
                        onMouseEnter={() => {
                            setHoveredItem([i,j])
                        }}
                        onMouseLeave={() => {
                            setHoveredItem([-1, -1])
                        }}
                        data-tooltip-id="image-tooltip"
                    />
                )))}
            </g>
            <g>
                {/* Add a line seperator between each class */}
                {Array.from({ length: analyzeResult.selectedClasses.length-1 }, (_, i) => (
                    <React.Fragment key={i}>
                        <line
                            x1={labelScale(i + 1) - (cellWidth * analyzeResult.examplePerClass) / 2}
                            y1={svgPadding.top}
                            x2={labelScale(i + 1) - (cellWidth * analyzeResult.examplePerClass) / 2}
                            y2={height - svgPadding.bottom}
                            stroke="black"
                        />
                        <line
                            x1={svgPadding.top}
                            y1={labelScale(i + 1) - (cellWidth * analyzeResult.examplePerClass) / 2}
                            x2={height - svgPadding.bottom}
                            y2={labelScale(i + 1) - (cellWidth * analyzeResult.examplePerClass) / 2}
                            stroke="black"
                        />
                    </React.Fragment>
                ))}
            </g>
            </g>
            <g>
                {/* Add title for each class */}
                {analyzeResult.selectedClasses.map((label, i) => (
                    <text key={i}
                        textAnchor='start'
                        style={{
                            fontSize: "10px",
                        }}
                        transform={`
                            translate(${labelScale(i)-8}, ${svgPadding.top+16})
                        `}
                    >
                        {classLabels.length>0?shortenName(classLabels[label], 1/analyzeResult.selectedClasses.length*60):label}
                    </text>
                ))}
            </g>
            <g>
                {/* Add title for each class */}
                {analyzeResult.selectedClasses.map((label, i) => (
                    <text key={i}
                        textAnchor='end'
                        style={{
                            transformOrigin: `0% 0%`,
                            fontSize: "10px",
                        }}
                        transform={`
                            translate(${svgPadding.top+14}, ${labelScale(i)+50})
                            rotate(-90 0 0)
                        `}
                    >
                        {classLabels.length>0?shortenName(classLabels[label], 1/analyzeResult.selectedClasses.length*60):label}
                    </text>
                ))}
            </g>
            <g />
        </svg>
        {hoveredItem[0] !== -1 && <ImageToolTip
            imgs={hoveredItem}
            imgType={'raw'}
            imgData={{}}
            label={`Jaccard similarity: ${jDist[hoveredItem[0]][hoveredItem[1]].intersection}/${jDist[hoveredItem[0]][hoveredItem[1]].union}`}
        />}
    </>
})

export default JaccardSimilarityView