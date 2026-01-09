import React, { FC } from 'react'
import * as api from '@api'
import { Node } from '@types'
import * as d3 from 'd3'
import { useAppSelector } from '@hooks'
import { selectAnalysisResult } from '@features/analyzeSlice'
import { calcAllPairwiseDistance, calcSumPairwiseDistance, calcVariance, chunkify, findIndicesOfMax, getRawHeatmap, shortenName, transposeArray } from '@utils/utils'
import ImageToolTip from '@components/ImageToolTip'
import '@styles/activation_heatmap.css'

interface Props {
    node: Node;
    minWidth: number;
    minHeight: number;
    normalizeRow?: boolean;
    totalMaxChannels?: (arr: number[]) => number;
}

const CELL_MIN_WIDTH = 12
const CELL_SUMMARY_MIN_WIDTH = 20
const CELL_MIN_HEIGHT = 6

const ActivationHeatmapView: FC<Props> = ({ 
    node, 
    minWidth, 
    minHeight, 
    normalizeRow = true, 
    totalMaxChannels = (arr: number[]) => arr.length * 0.2 
}) => {
    const [heatmap, setHeatmap] = React.useState<number[][]>([])
    const svgRef = React.useRef<SVGSVGElement>(null)
    const analyzeResult = useAppSelector(selectAnalysisResult)
    const [globalColorScale, _setGlobalColorScale] = React.useState(false)
    const [hoveredItem, setHoveredItem] = React.useState<[number, number]>([-1, -1])
    const [sortBy, setSortBy] = React.useState<'count' | 'pairwise' | 'variance' | 'edge_weight' | 'none'>('pairwise')
    const [classNames, setClassNames] = React.useState<string[]>([])

    const svgPadding = { top: 10, right: 10, bottom: 10, left: 10 }

    React.useEffect(() => {
        if (analyzeResult.examplePerClass === 0) return
        if (['Conv2D', 'Concatenate', 'Dense', 'Conv2d', 'Linear', 'Cat', 'Add',].some(l => node.layer_type.includes(l))) {
            api.getAnalysisHeatmap(node.name).then(setHeatmap)
        }
        api.getLabels().then(setClassNames)
    }, [node.name, analyzeResult])

    if (heatmap.length === 0) return null
    const nExamples = analyzeResult.examplePerClass * analyzeResult.selectedClasses.length

    // Normalize all rows in heatmap
    const normalHeatmap = normalizeRow ? transposeArray(transposeArray(heatmap).map(row => {
        // Mean shift
        const mean = row.reduce((a, b) => a + b, 0) / row.length
        const meanShiftedRow = row.map(item => item - mean)

        // Normalize
        const max = Math.max(...meanShiftedRow)
        const min = Math.min(...meanShiftedRow)
        const normalRow = (max - min === 0) ? meanShiftedRow.map(item => 0) : meanShiftedRow.map(item => (item - min) / (max - min))
        return normalRow
    })) : heatmap.map(col => {
        // Mean shift
        const mean = col.reduce((a, b) => a + b, 0) / col.length
        const meanShiftedCol = col.map(item => item - mean)

        // Normalize
        const max = Math.max(...meanShiftedCol)
        const min = Math.min(...meanShiftedCol)
        const normalCol = (max - min === 0) ? meanShiftedCol.map(item => 0) : meanShiftedCol.map(item => (item - min) / (max - min))
        return normalCol
    })

    // Add variance as a new column
    const h1 = transposeArray(transposeArray(
        getRawHeatmap(normalHeatmap, nExamples, analyzeResult.selectedClasses.length))
        .map(row => [...row, calcVariance(row)])
    )

    // Add pairwise distance as a new column
    const h2 = transposeArray(
        transposeArray(h1).map(row => [...row, (analyzeResult.selectedClasses.length > 1 ? calcSumPairwiseDistance : calcAllPairwiseDistance)(
            ...chunkify(row.slice(0, nExamples), analyzeResult.examplePerClass)
        )])
    )

    // Choose only first totalMaxChannels elements of each row
    const indicesMax = h2.map(arr => findIndicesOfMax(arr, totalMaxChannels!(arr)))
    h2.forEach((col, i) => {
        // Make all other elements of arr 0 except the max elements
        col.forEach((_, j) => {
            if (!indicesMax[i].includes(j)) {
                col[j] = 0
            }
        })
    })


    // Calculate difference between number of activated channels and number of examples
    const h3 = transposeArray(
        transposeArray(h2).map(row => [
            ...row,
            calcAllPairwiseDistance(
                analyzeResult.selectedClasses.length > 1 ?
                    chunkify(row.slice(0, nExamples), analyzeResult.examplePerClass)
                        .map(row => row.reduce((prev, curr) => prev + (curr > 0 ? 1 : 0), 0)) :
                    row.slice(0, nExamples)
            )
        ])
    )

    // Add edge weight as a new column
    let h4
    if (['Conv2D'].includes(node.layer_type)) {
        h4 = transposeArray(
            transposeArray(h3).map((row, i) => [
                ...row,
                node.out_edge_weight[i]
            ])
        )
    } else {
        h4 = h3
    }

    // Sort all colors by the last summary columns
    // 0: Keep the original order
    // 1: if there are multiple classes: Sum of pairwise activation count (after taking totalMaxChannels) distance between all classes
    // 1: if there is only one class: Sum of pairwise activation value (after making all values 0 except totalMaxChannels) distance between all examples
    // 2: if there are multiple classes: Sum of pairwise activation value distance between all classes
    // 2: if there is only one class: Sum of pairwise activation value distance between all examples
    // 3: Variance of all examples
    let SORT_BY = 0
    if (sortBy === 'count') SORT_BY = 2
    else if (sortBy === 'edge_weight') SORT_BY = 1
    else if (sortBy === 'pairwise') SORT_BY = 3
    else if (sortBy === 'variance') SORT_BY = 4

    const finalHeatmapAll = SORT_BY === 0 ? h4 : transposeArray(transposeArray(h4).sort((a, b) => b[b.length - SORT_BY] - a[a.length - SORT_BY]))

    const TOP_N = 40
    const finalHeatmap = finalHeatmapAll.map(col => col.slice(0, TOP_N))

    const extraCols = finalHeatmap.length - analyzeResult.selectedClasses.length * analyzeResult.examplePerClass

    // Apply the colorScale to finalHeatmap
    const colorScales = finalHeatmap.slice(0, finalHeatmap.length - extraCols).map(col => d3.scaleLinear<number>()
        .domain(globalColorScale ? [
            Math.min(...finalHeatmap.slice(0, finalHeatmap.length - extraCols).map(col => Math.min(...col))),
            Math.max(...finalHeatmap.slice(0, finalHeatmap.length - extraCols).map(col => Math.max(...col))),
        ] : [
            Math.min(...col),
            Math.max(...col),
        ])
        .range([0, 1])
        .clamp(true)
    )
    // Apply color scale to stats for last 4 columns (variance, pairwise, count, edge weight) with green color
    const statsColorScale = finalHeatmap.slice(-extraCols).map(col => d3.scaleLinear<number>()
        .domain([
            Math.min(...col),
            Math.max(...col),
        ])
        .range([0, 1])
    )
    const allColors = finalHeatmap.map((col, i) => col.map(elem => {
        if (i < analyzeResult.selectedClasses.length * analyzeResult.examplePerClass) {
            return colorScales[i](elem)
        }
        return statsColorScale[i - analyzeResult.selectedClasses.length * analyzeResult.examplePerClass](elem)
    }))

    // Apply the color scale to allColors
    const heatmapColor = transposeArray(transposeArray(allColors)).map((row, i) => {
        if (i < analyzeResult.selectedClasses.length * analyzeResult.examplePerClass) {
            return row.map(d3.interpolateBlues)
        }
        return row.map(d3.interpolateGreens)
    })
    
    // Drawing parameters
    const curCellWidth = (minWidth - svgPadding.left - svgPadding.right) / heatmapColor.length
    const curCellHeight = (minHeight - svgPadding.top - svgPadding.bottom) / heatmapColor[0].length
    
    const width = curCellWidth >= CELL_MIN_WIDTH ? minWidth : CELL_MIN_WIDTH * heatmapColor.length + svgPadding.left + svgPadding.right
    const height = curCellHeight >= CELL_MIN_HEIGHT ? minHeight : CELL_MIN_HEIGHT * heatmapColor[0].length + svgPadding.top + svgPadding.bottom
    
    const cellWidth = curCellWidth >= CELL_MIN_WIDTH ? curCellWidth : CELL_MIN_WIDTH
    const cellHeight = curCellHeight >= CELL_MIN_HEIGHT ? curCellHeight : CELL_MIN_HEIGHT
    const cellSummaryWidth = curCellWidth >= CELL_SUMMARY_MIN_WIDTH ? curCellWidth : CELL_SUMMARY_MIN_WIDTH
    
    const labelScale = d3.scaleLinear()
        .domain([0, analyzeResult.selectedClasses.length - 1])
        .range([
            svgPadding.left + (cellWidth * analyzeResult.examplePerClass) / 2,
            width - svgPadding.right - (cellWidth * analyzeResult.examplePerClass) / 2 - cellWidth * extraCols
        ])

    const statLabelScale = d3.scaleLinear()
        .domain([0, extraCols - 1])
        .range([
            svgPadding.left + cellWidth * analyzeResult.selectedClasses.length * analyzeResult.examplePerClass + (cellWidth) / 2,
            width - svgPadding.right - cellWidth / 2
        ])
        
    const sortById2Labels = {
        'none': 'None',
        'count': 'Count',
        'pairwise': 'Pairwise',
        'variance': 'Variance',
        'edge_weight': 'Edge Weight'
    }


    return <>
        <svg width={width+30} height={height + 50} ref={svgRef} style={{
            backgroundColor: "white"
        }}>
            <g transform='translate(0, 50)'>
                <g>
                    {heatmapColor.map((col, i) => <>
                        {/* <rect
                            key={`${i}-activated`}
                            x={i * cellWidth + svgPadding.left}
                            // y={j * cellHeight + svgPadding.top}
                            width={50}
                            height={cellHeight}
                            fill="black"
                        /> */}
                    </>)}
                    {heatmapColor.map((col, i) => <>
                        {col.map((elem, j) => 
                            <rect
                                key={`${i}-${j}`}
                                x={i * cellWidth + svgPadding.left}
                                y={j * cellHeight + svgPadding.top}
                                width={(i < analyzeResult.selectedClasses.length * analyzeResult.examplePerClass) ? cellWidth : cellSummaryWidth}
                                height={cellHeight}
                                fill={elem}
                                onMouseEnter={() => {
                                    if (['Conv2D', 'Concatenate', 'Conv2d', 'Cat'].some(l => node.layer_type.includes(l)))
                                        setHoveredItem([i, j])
                                }}
                                onMouseLeave={() => {
                                    if (['Conv2D', 'Concatenate', 'Conv2d', 'Cat'].some(l => node.layer_type.includes(l)))
                                        setHoveredItem([-1, -1])
                                }}
                                data-tooltip-id="image-tooltip"
                            />
                        )}
                    </>)}
                </g>
                <g>
                    <line
                        x1={svgPadding.left}
                        y1={height - svgPadding.bottom}
                        x2={width - svgPadding.right - cellWidth * extraCols}
                        y2={height - svgPadding.bottom}
                        stroke="black"
                    />
                    <line
                        x1={svgPadding.left}
                        y1={height - svgPadding.bottom}
                        x2={svgPadding.left}
                        y2={0}
                        stroke="black"
                    />
                    <line
                        x1={width - svgPadding.right - cellWidth * extraCols}
                        y1={height - svgPadding.bottom}
                        x2={width - svgPadding.right - cellWidth * extraCols}
                        y2={0}
                        stroke="black"
                    />
                    <text transform={`translate(${(width - svgPadding.right - cellWidth * extraCols) / 2}, ${height - 1})`}>
                        Images
                    </text>
                    <text
                        textAnchor='middle'
                        transform={`translate(${0}, ${height / 2}) rotate(90)`}>
                        Channel Activation
                    </text>
                    {/* Add a line seperator between each class */}
                    {Array.from({ length: analyzeResult.selectedClasses.length - 1 }, (_, i) => (
                        <line
                            key={i}
                            x1={labelScale(i + 1) - (cellWidth * analyzeResult.examplePerClass) / 2}
                            y1={svgPadding.top - 10}
                            x2={labelScale(i + 1) - (cellWidth * analyzeResult.examplePerClass) / 2}
                            y2={height - svgPadding.bottom}
                            stroke="black"
                        />
                    ))}
                </g>
            </g>
            <g>
                {/* Add title for each class */}
                {analyzeResult.selectedClasses.map((label, i) => (
                    <text key={label+i}
                        textAnchor='start'
                        style={{
                            transformOrigin: `0% 0%`,
                            fontSize: '10px'
                        }}
                        transform={`
                            translate(${labelScale(i) - 10}, ${svgPadding.top + 45})
                            rotate(-45 0 0)
                        `}
                    >
                        <title>{classNames.length > 0 ? classNames[label] : label}</title>
                        <tspan>{shortenName(classNames.length > 0 ? classNames[label] : label.toString(), 10)}</tspan>
                    </text>
                ))}
            </g>
            <g>
                {/* Add title for each stats */}
                {["Variance", "Pairwise", "Count", "Edge Weight"].map((label, i) => <>
                    <text key={i}
                        onMouseEnter={(e) => {
                            e.currentTarget.classList.add('hovered')
                        }}
                        onMouseLeave={(e) => {
                            e.currentTarget.classList.remove('hovered')
                        }}
                        textAnchor='start'
                        style={{
                            transformOrigin: `0% 0%`,
                            fontSize: '10px',
                        }}
                        transform={`
                            translate(${statLabelScale(i) - 6}, ${svgPadding.top + 45})
                            rotate(-45 0 0)
                        `}
                        onClick={() => {
                            if (label === 'Variance') setSortBy('variance')
                            else if (label === 'Pairwise') setSortBy('pairwise')
                            else if (label === 'Count') setSortBy('count')
                            else if (label === 'Edge Weight') setSortBy('edge_weight')
                        }}
                        className={'activation-stats' + (label === sortById2Labels[sortBy] ? ' selected' : '')}
                    >
                        <tspan>{label}</tspan>
                    </text>
                    {label === sortById2Labels[sortBy] && (
                        <text
                            textAnchor='start'
                            style={{
                                transformOrigin: `0% 0%`,
                                fontSize: '10px',
                                fill: 'black'
                            }}
                            transform={`
                                translate(${statLabelScale(i) - 3}, ${svgPadding.top + 20})
                            `}
                        >
                        </text>
                    )}
                </>)}
            </g>
            <g>
                {/* Add a highlighting line above and below the rects of hoveredItem */}
                {/* {hoveredItem[1] !== -1 && (
                    <>
                        <line
                            x1={svgPadding.left}
                            y1={svgPadding.top + hoveredItem[1] * cellHeight}
                            x2={width - svgPadding.right}
                            y2={svgPadding.top + hoveredItem[1] * cellHeight}
                            stroke="yellow"
                        />
                        <line
                            x1={svgPadding.left}
                            y1={svgPadding.top + (hoveredItem[1] + 1) * cellHeight}
                            x2={width - svgPadding.right}
                            y2={svgPadding.top + (hoveredItem[1] + 1) * cellHeight}
                            stroke="yellow"
                        />
                    </>
                )} */}
            </g>
        </svg>
        {hoveredItem[0] !== -1 && hoveredItem[0] < nExamples && <ImageToolTip
            imgs={[hoveredItem[0]]}
            imgType={'overlay'}
            imgData={{
                layer: node.name,
                channel: hoveredItem[1]
            }}
            label={`Layer: ${node.name}, Channel: ${hoveredItem[1]}`}
        />}
    </>
}

export default ActivationHeatmapView