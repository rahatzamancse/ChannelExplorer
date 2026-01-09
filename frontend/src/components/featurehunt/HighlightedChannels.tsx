import React from 'react'
import * as api from '@api'
import { useAppSelector } from '@hooks'
import { selectFeatureHunt } from '@features/featureHuntSlice'

const HighlightedChannels = () => {
    const [heatmap, setHeatmap] = React.useState<number[][]>([])
    const svgRef = React.useRef<SVGSVGElement>(null)
    const featureHuntState = useAppSelector(selectFeatureHunt)
    
    
    const _exampleActivatedChannels = {
        "activated_channels": {
            "conv2d_1": [
                0,
                1,
            ],
        }
    }
    
    React.useEffect(() => {
        api.getFeatureActivatedChannels()
            .then((activateds) => {
                const max = Math.max(...Object.values(activateds['activated_channels']).flat())
                const matrix: number[][] = new Array( Object.keys(activateds['activated_channels']).length).fill(false).map(() => new Array(max).fill(0))
                
                
                Object.values(activateds['activated_channels']).forEach((channels, i) => {
                    channels.forEach((channel: number) => {
                        matrix[i][channel] = 1
                    })
                })
                setHeatmap(matrix)
            })
    }, [featureHuntState.uploadComplete])
    

    const svgPadding = { top: 10, right: 10, bottom: 10, left: 10 }

    if (heatmap.length === 0) return null
        
    // Drawing parameters
    const width = 600
    const height = 3000
    const cellWidth = (width - svgPadding.left - svgPadding.right) / 13
    const cellHeight = (height - svgPadding.top - svgPadding.bottom) / 1000

    return <div className="rsection" style={{
        display: "flex",
        flexDirection: "column",
        width: "100%",
        minWidth: "300px",
        // maxWidth: "500px",
        minHeight: "90vh",
        maxHeight: "90vh",
        padding: "20px",
    }}>
        <svg width={width} height={height} ref={svgRef} style={{
            backgroundColor: "white"
        }}>
            <g>
                {heatmap.map((col, i) => col.map((elem, j) => (
                    <rect
                        key={`${i}-${j}`}
                        x={i * cellWidth + svgPadding.left}
                        y={j * cellHeight + svgPadding.top}
                        width={cellWidth}
                        height={cellHeight}
                        fill={elem===1?'blue':'lightgray'}
                        // onMouseEnter={() => {
                        //     if (['Conv2D', 'Concatenate'].some(l => node.layer_type.includes(l)))
                        //         setHoveredItem([i, j])
                        // }}
                        // onMouseLeave={() => {
                        //     if (['Conv2D', 'Concatenate'].some(l => node.layer_type.includes(l)))
                        //         setHoveredItem([-1, -1])
                        // }}
                        // data-tooltip-id="image-tooltip"
                    />
                )))}
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
    </div>
}

export default HighlightedChannels