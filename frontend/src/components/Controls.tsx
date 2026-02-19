import React from 'react'
import Form from 'react-bootstrap/Form';
import { ProgressBar } from 'react-bootstrap';
import * as api from '@api'
import { useAppDispatch, useAppSelector } from '@hooks'
import { selectAnalysisResult, setAnalysisResult } from '@features/analyzeSlice';
import { chunkify, shortenName } from '@utils/utils';
import { Modal, Spinner } from 'react-bootstrap';
import { useTour } from '@reactour/tour';
import ModalImage from 'react-modal-image';
import '@styles/control.css'
import '@styles/scrollbar.css'

const Controls = React.memo(function Controls() {
    const [isProcessing, setIsProcessing] = React.useState<boolean>(false)
    const [processingMessage, setProcessingMessage] = React.useState<string>("")
    const [classes, setClasses] = React.useState<string[]>([])
    const analysisResult = useAppSelector(selectAnalysisResult)
    const [inputImages, setInputImages] = React.useState<string[]>([])
    const [shuffled, setShuffled] = React.useState<boolean>(false)
    const [nExamplePerClass, setNExamplePerClass] = React.useState<number>(5)
    const [curClassProgressSelected, setCurClassProgressSelected] = React.useState<number>(0)
    const { setIsOpen } = useTour();

    const dispatch = useAppDispatch()

    const checkboxRefs = React.useRef<HTMLInputElement[]>([])
    const nExamplePerClassRef = React.useRef<HTMLInputElement>(null)
    const pollTimerRef = React.useRef<ReturnType<typeof setTimeout> | null>(null)
    const [progressValues, setProgressValues] = React.useState<number[]>(classes.map(() => 0))

    React.useEffect(() => {
        return () => {
            if (pollTimerRef.current) clearTimeout(pollTimerRef.current)
        }
    }, [])

    const checkTaskStatus = React.useCallback((task_id: string) => {
        setIsProcessing(true)
        pollTimerRef.current = setTimeout(() => {
            api.getTaskStatus(task_id).then((res) => {
                if(res.payload === null) {
                    setProcessingMessage(res.message)
                    checkTaskStatus(task_id)
                } else {
                    setIsProcessing(false)
                    setProcessingMessage(res.message)
                    pollTimerRef.current = null

                    dispatch(setAnalysisResult(res.payload))
                    api.getInputImages([...Array(res.payload.selectedClasses.length*res.payload.examplePerClass).keys()]).then(setInputImages)
                    setNExamplePerClass(res.payload.examplePerClass)
                    if(nExamplePerClassRef.current !== null) nExamplePerClassRef.current.value = res.payload.examplePerClass.toString()
                    setShuffled(res.payload.shuffled)
                    if (!localStorage.getItem('firstVisit')) {
                        localStorage.setItem('firstVisit', 'true');
                        setTimeout(() => {
                            setIsOpen(true);
                        }, 2000)
                    }
                }
            })
        }, 1000)
    }, [dispatch, setIsOpen])

    React.useEffect(() => {
        api.analyze([0, 1, 2, 3], 5, false)
            .then(({ message, task_id }) => {
                checkTaskStatus(task_id)
            })
        api.getLabels().then(setClasses)
    }, [])
    
    React.useEffect(() => {
        api.getConfiguration().then((res) => {
            checkboxRefs.current.forEach((checkbox, index) => {
                checkbox.checked = res.selectedClasses.includes(index)
            })
        })
    }, [classes])
    
    React.useEffect(() => {
        const curProgressValues = Array.from({length: classes.length}, () => Math.random()*100)
        
        const classSimilarities = [
            [10, 7, 4, 3, 3.5, 3.7, 3.7, 3, 2.9, 2.6],
            [7, 10, 4, 3, 3.5, 3.7, 3.7, 3, 2.9, 2.6],
            [4, 4, 100, 8, 7, 4, 5, 3.7, 2.0, 3],
        ]
        
        if(classSimilarities.length > curClassProgressSelected) {
            for(let i = 0; i < classSimilarities[curClassProgressSelected].length; i++) {
                curProgressValues[i] = classSimilarities[curClassProgressSelected][i]*10
            }
        }
        
        setProgressValues(curProgressValues)
    }, [classes, curClassProgressSelected])
    
    return <><div className="rsection tutorial-control" style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "flex-start",
        width: "20%",
        minWidth: "280px",
        maxWidth: "500px",
        minHeight: "90vh",
        maxHeight: "94vh",
        padding: "8px",
    }}>
        <Form style={{ width: "100%" }}>
            <button title='Reload the last loaded state of the Neural Network' type="button" className="btn btn-primary" style={{
                width: "100%",
                marginBottom: "20px",
                alignSelf: "flex-end"
            }} onClick={(e) => {
                if(checkboxRefs.current.length === 0) return
                api.getConfiguration().then((res) => {
                    setNExamplePerClass(res.examplePerClass)
                    if(nExamplePerClassRef.current !== null) nExamplePerClassRef.current.value = res.examplePerClass.toString()
                    checkboxRefs.current.forEach((checkbox, index) => {
                        checkbox.checked = res.selectedClasses.includes(index)
                    })
                    setShuffled(res.shuffled)
                    api.getInputImages([...Array(res.selectedClasses.length*res.examplePerClass).keys()]).then(setInputImages)
                    dispatch(setAnalysisResult(res))
                })
            }}><span className="glyphicon glyphicon-refresh">Refresh</span></button>
        </Form>
        <div style={{
            border: "1px solid lightgray",
            borderRadius: "20px",
            padding: "12px",
            width: "100%",
        }}>
            <h5 className="mb-3">Select Labels to Analyze</h5>
            <Form.Check type="switch" id="shufflecheck" label="Shuffle" checked={shuffled} onChange={e => setShuffled(e.target.checked)} className='mb-3 tutorial-shuffle' />
            <Form.Control ref={nExamplePerClassRef} id="nExamplePerClass" className="mb-1  tutorial-image-per-class" type="number" min={1} max={50} placeholder='# Image Per Class' />
            {/* <Form.Label htmlFor="nExamplePerClass" style={{ float: 'left', width: '40%'}}>Image per Class</Form.Label> */}
            <div style={{
                display: "flex",
                flexDirection: "column",
                alignItems: "flex-start",
                justifyContent: "flex-start",
                overflowY: "scroll",
                overflowX: "hidden",
                height: "20vh",
                width: "100%",
            }} className='tutorial-selected-classes scrollbar'>
                {classes.map((label, index) => (
                    <div style={{
                        position: 'relative',
                    }} key={index}>
                        <Form.Check
                            key={index}
                            type="checkbox"
                            label={label}
                            id={`checkbox_${index}`}
                            ref={(ref: HTMLInputElement) => {checkboxRefs.current[index] = ref}}
                        />
                        <ProgressBar
                            now={progressValues[index]}
                            style={{
                                position: 'absolute',
                                top: 5,
                                left: 24,
                                width: '180px',
                                height: '16px',
                                opacity: 0.3,
                            }}
                            variant={index === curClassProgressSelected ? 'danger' : 'primary'}
                            onClick={e => {
                                e.stopPropagation()
                                setCurClassProgressSelected(index)
                            }
                        }/>
                    </div>
                ))}
            </div>
            <button className="btn btn-primary mt-2 tutorial-analyze"
            style={{
                width: "100%",
            }}
                
            onClick={() => {
                const checkedLabels = checkboxRefs.current.map((checkbox, index) => checkbox.checked).reduce<number[]>((out, bool, index) => bool?out.concat(index):out, [])
                // const checkedLabels = Array.from(document.querySelectorAll("input[type=checkbox]:checked")).map(checkbox => parseInt(checkbox.id.split("_")[1]))
                
                const currentNExamplePerClass = parseInt(nExamplePerClassRef.current?.value || "5")

                api.analyze(checkedLabels, currentNExamplePerClass, shuffled)
                    .then(({ message, task_id }) => {
                        checkTaskStatus(task_id)
                    })
            }}>Analyze</button>
        </div>
        
        <hr />
        <h4 className="mt-2">Input Images</h4>
        {/* Save all images */}
        <button className="btn btn-primary mt-2" style={{
            width: "100%",
        }} onClick={() => {
            api.saveDataset()
        }}>Save All Images</button>
        {inputImages.length > 0 ? <div style={{
            display: "flex",
            paddingLeft: "8px",
            paddingTop: "20px",
            paddingRight: "2px",
            flexDirection: "column",
            marginTop: "20px",
            alignItems: "flex-start",
            justifyContent: "flex-start",
            width: "100%",
            overflowY: "scroll",
            border: "1px solid lightgray",
            borderRadius: "20px",
        }} className='tutorial-input-images scrollbar scrollbar-wide'>
            {chunkify(inputImages, nExamplePerClass).map((chunk, i) => <div key={'outerdiv'+i} style={{
                width: "100%",
                marginBottom: "20px",
            }}>
                { i === 0? <div>
                    <h5>External Images</h5>
                    <div className='scrollbar'
                        key={'div-start'+i}
                        style={{
                            display: "flex",
                            flexDirection: "row",
                            alignItems: "flex-start",
                            justifyContent: "flex-start",
                            overflowY: "scroll",
                            maxHeight: "20vh",
                            width: "100%",
                            flexWrap: "wrap",
                        }}
                    >
                        {/* <div style={{
                            display: "flex",
                            flexDirection: "column",
                            alignItems: "center",
                            justifyContent: "center",
                        }}>
                            <ModalImage hideZoom={false} className='input-image' small={"https://sharkfreewaters.files.wordpress.com/2012/05/shark1.jpg"} large={"https://sharkfreewaters.files.wordpress.com/2012/05/shark1.jpg"} alt={"Adversarial Shark"} />
                            <span style={{ color: "black", }}>
                                {shortenName("trilobite", 10)}
                            </span>
                        </div>
                        <div style={{
                            display: "flex",
                            flexDirection: "column",
                            alignItems: "center",
                            justifyContent: "center",
                        }}>
                            <ModalImage hideZoom={false} className='input-image' small={"assets/panda-gibbon-adversarial.png"} large={"assets/panda-gibbon-adversarial.png"} alt={"Adversarial Panda"} />
                            <span style={{ color: "black", }}>
                                {shortenName("trilobite", 10)}
                            </span>
                        </div> */}
                        <div style={{
                            display: "flex",
                            flexDirection: "column",
                            alignItems: "center",
                            justifyContent: "center",
                        }}>
                            <ModalImage hideZoom={false} className='input-image' small={"assets/upload-external.png"} large={"assets/upload-external.png"} alt={"Upload External"} />
                        </div>
                    </div>
                </div>:null}
                <h5 key={'h5'+i}>Class: {classes[analysisResult.selectedClasses[i]]}</h5>
                <div className='scrollbar'
                    key={'div'+i}
                    style={{
                        display: "flex",
                        flexDirection: "row",
                        alignItems: "flex-start",
                        justifyContent: "flex-start",
                        overflowY: "scroll",
                        maxHeight: "20vh",
                        width: "100%",
                        flexWrap: "wrap",
                    }}
                >
                    {chunk.map((image,j) => <div style={{
                        display: "flex",
                        flexDirection: "column",
                        alignItems: "center",
                        justifyContent: "center",
                    }} key={'chunk_div'+j}>
                        <ModalImage hideZoom={false} className='input-image' small={image} large={image} alt={classes[analysisResult.selectedClasses[i]]} />
                        <span style={{
                            color: analysisResult.selectedClasses[i] === analysisResult.predictions[i*analysisResult.examplePerClass+j] ? "green" : "red",
                        }}>
                            {classes[analysisResult.predictions[i*analysisResult.examplePerClass+j]] && shortenName(classes[analysisResult.predictions[i*analysisResult.examplePerClass+j]], 10)}
                        </span>
                    </div>)}
                </div>
            </div>)}
        </div>:null}
    </div>
    
    <Modal show={isProcessing} centered>
        <Modal.Header>
            <Modal.Title>Running Model</Modal.Title>
        </Modal.Header>
        <Modal.Body>
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
                <Spinner animation="border" role="status">
                    {/* <span className="sr-only">Loading...</span> */}
                </Spinner>
                <p>{processingMessage}</p>
            </div>
        </Modal.Body>
    </Modal>
    

    </>
})

export default Controls
            
