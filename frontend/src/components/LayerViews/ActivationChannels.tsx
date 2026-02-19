import React from 'react'
import { Node } from '@types'

import * as api from '@api'
import { useAppSelector } from '@hooks'
import { selectAnalysisResult } from '@features/analyzeSlice';
import ModalImage from 'react-modal-image'
import '@styles/layer_activation.css'


const ActivationChannels = React.memo(function ActivationChannels({ node }: { node: Node }) {
    const analyzeResult = useAppSelector(selectAnalysisResult)
    const nImgs = analyzeResult.selectedClasses.length * analyzeResult.examplePerClass
    const [currentPage, setCurrentPage] = React.useState<number>(0)
    const [activations, setActivations] = React.useState<string[][]>([])
    const filtersPerPage = 5
    const nPages = Math.ceil(node.output_shape[3] / filtersPerPage)
    
    React.useEffect(() => {

        if (nImgs === 0) return
        const imgNodes = ['Conv2D', 'Concatenate', 'Add', 'Cat', 'Conv2d']
        if (!imgNodes.some(imgNode => node.layer_type.toLowerCase().includes(imgNode.toLowerCase()))) return
        api.getActivationsImages(
            node,
            currentPage === nPages-1? node.output_shape[3] - (nPages-1)*filtersPerPage : currentPage*filtersPerPage,
            filtersPerPage,
            nImgs
        ).then(setActivations)
    }, [node, currentPage])
    
    const nextPage = () => {
        if (currentPage === nPages - 1) return
        setCurrentPage(currentPage + 1)
    }
    const prevPage = () => {
        if (currentPage === 0) return
        setCurrentPage(currentPage - 1)
    }
    
    return <div style={{
        display: "flex",
        flexDirection: "column",
        width: "100%",
        overflowY: "scroll",
        overflowX: "hidden",
    }}>
        <div style={{
            display: "flex",
            flexWrap: "wrap",
            justifyContent: "center",
            flexDirection: "column",
        }}>
            {activations.map((row, i) => 
                <div key={i} style={{ display: "flex", flexDirection: "row" }}>
                    {row.map((image, j) =>
                        <span key={`${i}-${j}`} onErrorCapture={e => {
                            const el = e.currentTarget as HTMLElement
                            if ((e.target as HTMLElement).tagName === 'IMG') el.style.display = 'none'
                        }}>
                            <ModalImage
                                className='raw-activation-img'
                                hideZoom={false}
                                small={image}
                                large={image}
                            />
                        </span>
                    )}
                </div>
            )}
        </div>
        <div style={{
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            marginTop: "10px"
        }}>
            <button onClick={prevPage} disabled={currentPage === 0}>Previous</button>
            {/* Add page numbers */}
            <span style={{padding: "0 10px"}}>{currentPage+1}/{nPages}</span>
            <button onClick={nextPage} disabled={currentPage === nPages-1}>Next</button>
        </div>
    </div>
})

export default ActivationChannels