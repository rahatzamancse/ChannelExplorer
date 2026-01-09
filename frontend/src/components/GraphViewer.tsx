import React from 'react'
import * as api from '@api'
import { useTour } from '@reactour/tour';
import { useStoreApi } from 'reactflow';

import ReactFlow, {
    MiniMap,
    Controls,
    Background,
    useNodesState,
    useEdgesState,
    Node,
    Edge,
    useReactFlow,
    useUpdateNodeInternals,
    ControlButton,
} from 'reactflow';
import { toPng } from 'html-to-image';

import 'reactflow/dist/style.css';
import LayerNode from './LayerNode';
import { Node as BaseNode } from '../types'

const initialNodes: Node[] = [
  { id: '1', position: { x: 0, y: 0 }, data: { label: 'Loading Graph\nPlease Wait' } },
];
const initialEdges: Edge[] = [];

const GRAPH_HEIGHT_FACTOR = 200
const GRAPH_WIDTH_FACTOR = 40

const nodeTypes = { layerNode: LayerNode };



function GraphViewer() {
    const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
    const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);
    const [layoutHorizontal, setLayoutHorizontal] = React.useState(true);
    const { isOpen, currentStep } = useTour()
    
    const reactFlowInstance = useReactFlow();
    const updateNodeInternals = useUpdateNodeInternals()
    const flowRef = React.useRef<HTMLDivElement>(null);
    
    React.useEffect(() => {
        api.getModelGraph().then(modelGraph => {
            const max_depth = modelGraph.meta.depth * 2.5
            
            // TODO: Get the width of the model tree
            const max_width = 20
            
            const getX = (node: BaseNode) => node.pos?node.pos.y*-1*GRAPH_HEIGHT_FACTOR*max_depth:0
            const getY = (node: BaseNode) => node.pos?node.pos.x*GRAPH_WIDTH_FACTOR*max_width:0
            
            let firstCNNSet = false
            
            setNodes(modelGraph.nodes.map(node => {
                const resNode = {
                    id: node.id,
                    position: { 
                        x: layoutHorizontal?getX(node):getY(node),
                        y: layoutHorizontal?getY(node):getX(node), 
                    },
                    data: {
                        label: node.label,
                        layer_type: node.layer_type,
                        name: node.name,
                        input_shape: node.input_shape,
                        kernel_size: node.kernel_size,
                        output_shape: node.output_shape,
                        tensor_type: node.tensor_type,
                        out_edge_weight: node.out_edge_weight,
                        layout_horizontal: layoutHorizontal,
                        tutorial_node: false,
                    },
                    type: 'layerNode',
                }
                if (!firstCNNSet && node.layer_type.toLowerCase().includes('conv2d')) {
                    firstCNNSet = true
                    resNode.data.tutorial_node = true
                }
                return resNode
            }))
            setEdges(modelGraph.edges.map(edge => ({
                id: `${edge.source}-${edge.target}`,
                source: edge.source,
                target: edge.target,
                animated: true,
                label: ''
            })))
            
        })
    }, [])
    
    React.useEffect(() => {
        setNodes(val => val.map(node => ({
                ...node,
                position: {
                    ...node.position,
                    x: node.position.y,
                    y: node.position.x
                },
                data: {
                    ...node.data,
                    layout_horizontal: layoutHorizontal
                }
            }))
        )
        
        nodes.forEach(node => {
            updateNodeInternals(node.id)
        })
        
        setTimeout(() => {
            reactFlowInstance.fitView({ duration: 800 })
        }, 1000)
        
    }, [layoutHorizontal])
    
    const store = useStoreApi();
    
    const focusNode = (node: Node) => {
        const { nodeInternals } = store.getState();
        const nodes = Array.from(nodeInternals).map(([, node]) => node);

        if (nodes.length > 0) {
            const nodeToZoom = nodes.find(n => n.id === node.id);
            if (!nodeToZoom || !nodeToZoom.width || !nodeToZoom.height) return;

            const x = nodeToZoom.position.x + nodeToZoom.width / 2;
            const y = nodeToZoom.position.y + nodeToZoom.height / 2;
            const zoom = 1.85;

            reactFlowInstance.setCenter(x, y, { zoom, duration: 1000 });
        }
    };
    
    React.useEffect(() => {
        if (!isOpen) return
        if (currentStep === 12) {
            reactFlowInstance.fitView({ duration: 800 })
        }
        else if(currentStep === 13) {
            focusNode(nodes.find(node => node.data.tutorial_node === true)!)
        }
    }, [isOpen, currentStep])

    

    const positions = nodes.map(node => node.position)
    const minX = Math.min(...positions.map(pos => pos.x)) - 500
    const minY = Math.min(...positions.map(pos => pos.y)) - 500
    const maxX = Math.max(...positions.map(pos => pos.x)) + 500
    const maxY = Math.max(...positions.map(pos => pos.y)) + 500
    const maxRange = Math.max((maxX-minX)/2, (maxY-minY)/2) * 1.2
    const center = { x: minX + (maxX-minX)/2, y: minY + (maxY-minY)/2 }
    const translationExtent: [[number,number],[number,number]] = [
        [center.x - maxRange, center.y - maxRange],
        [center.x + maxRange, center.y + maxRange]
    ]
    
    return <div className="rsection tutorial-main-view" style={{
        display: "flex",
        width: "100%",
        minWidth: "600px",
        height: "92vh",
        minHeight: "1000px",
    }}>
        <ReactFlow
            nodes={nodes}
            nodeTypes={nodeTypes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            fitView
            fitViewOptions={{ padding: 0.1 }}
            attributionPosition="bottom-right"
            minZoom={0.1}
            maxZoom={10}
            translateExtent={translationExtent}
            elevateEdgesOnSelect
            ref={flowRef}
        >
            <MiniMap pannable zoomable style={{
                border: '1px solid #000',
            }}/>
            <Controls className='tutorial-main-view-controls' >
                <ControlButton onClick={() => setLayoutHorizontal(val => !val)} title='Change layout between vertical and horizontal.'>
                    {layoutHorizontal ? "H" : "V"}
                </ControlButton>
                <ControlButton title="Export an image of the network current view." onClick={() => {
                    if(flowRef.current === null) return
                    toPng(flowRef.current, {
                        filter: node => !(
                            node?.classList?.contains('react-flow__minimap') ||
                            node?.classList?.contains('react-flow__controls')
                        ),
                        // canvasWidth: 5000,
                        // canvasHeight: 1000,
                    }).then(dataUrl => {
                        const a = document.createElement('a');
                        a.setAttribute('download', 'reactflow.png');
                        a.setAttribute('href', dataUrl);
                        a.click();
                    });
                }}>
                    <img src="assets/export.png" alt="Export" width="16px" height="16px" />
                </ControlButton>
                <ControlButton title="Load layout from a file" onClick={() => {
                    const input = document.createElement('input');
                    input.type = 'file';
                    input.onchange = async (e) => {
                        const file = (e.target as HTMLInputElement).files?.[0];
                        if (file) {
                            const reader = new FileReader();
                            reader.onload = async (e) => {
                                if (e.target) {
                                    const text = (e.target as any).result;
                                    if (text) {
                                        const flow = JSON.parse(text);
                                        if (flow) {
                                            const { x = 0, y = 0, zoom = 1 } = flow.viewport;
                                            setNodes(flow.nodes || []);
                                            setEdges(flow.edges || []);
                                            reactFlowInstance.setViewport({ x, y, zoom });
                                        }
                                    }
                                }
                            };
                            reader.readAsText(file);
                        }
                    };
                    input.click();
                }}>
                    <img src="assets/import.png" alt="Import" width="16px" height="16px"/>
                </ControlButton>
                <ControlButton title="Save current layout" onClick={() => {
                    const flow = reactFlowInstance.toObject();
                    const json = JSON.stringify(flow, null, 2);
                    const blob = new Blob([json], { type: 'application/json' });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = 'node-placement.json';
                    a.click();
                    URL.revokeObjectURL(url);
                }}>
                    <img src="assets/save.png" alt="Save" width="16px" height="16px"/>
                </ControlButton>
            </Controls>
            <Background />
        </ReactFlow>
    </div>
}

export default GraphViewer