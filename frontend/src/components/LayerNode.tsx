import React from 'react';
import { Handle, Position } from '@xyflow/react';
import { Node } from '@types';
import { NodeColors } from '@utils/utils';
import { Card, Accordion } from 'react-bootstrap';
import ActivationChannels from '@components/LayerViews/ActivationChannels';
import LazyAccordionItem from '@components/LazyAccordionItem';
import ScatterplotView from '@components/LayerViews/ScatterplotView';
import ActivationHeatmapView from '@components/LayerViews/ActivationHeatmapView';
import JaccardSimilarityView from '@components/LayerViews/JaccardSimilarityView';
import DenseArgmax from '@components/LayerViews/DenseArgmax';
import { useAppSelector } from '@hooks';
import { selectAnalysisResult } from '@features/analyzeSlice';
import HierarchyTree from '@components/LayerViews/HierarchyTree';


const LayerNode = React.memo(function LayerNode({ id: _id, data }: { id: string, data: Node }) {
    const analysisResult = useAppSelector(selectAnalysisResult)
    return (
        <div style={{
            display: 'flex',
            flexDirection: 'column',
            backgroundColor: NodeColors[data.layer_type],
            borderRadius: 10,
            border: '1px solid #aaa',
        }}>
            <Handle type="target" position={data.layout_horizontal?Position.Left:Position.Top} />
            <Card className={data.tutorial_node?'tutorial-cnn-layer':''}>
                <Card.Body style={{
                    display: 'flex',
                    flexDirection: 'column',
                }}>
                    <Card.Title>{data.name}</Card.Title>
                    <Accordion alwaysOpen flush>
                        <LazyAccordionItem className={data.tutorial_node?'tutorial-cnn-layer-details':''} header="Details" eventKey="2">
                            <ul>
                                <li> <b>Layer :</b> {data.layer_type} </li>
                                <li> <b>Input :</b> ({data.input_shape.toString()}) </li>
                                {data.kernel_size?<li> <b>Kernel Shape :</b> ({data.kernel_size.toString()}) </li>:null}
                                {data.out_edge_weight && <li> <b>Kernel # :</b> {data.out_edge_weight.length} </li>}
                                <li> <b>Output :</b> ({data.output_shape.toString()}) </li>
                            </ul>
                        </LazyAccordionItem>
                        {analysisResult.examplePerClass !== 0 && ['Conv2D', 'Concatenate', 'Conv2d', 'Cat'].includes(data.layer_type) && <LazyAccordionItem header="Activations" eventKey="3">
                            <ActivationChannels node={data} />
                        </LazyAccordionItem>}
                        {analysisResult.examplePerClass !== 0 && ['Dense', 'Linear'].includes(data.layer_type) && <LazyAccordionItem header="Predictions" eventKey="5">
                            <DenseArgmax node={data} />
                        </LazyAccordionItem>}
                        {/* {['Conv2D'].includes(data.layer_type) && <LazyAccordionItem header="Edges" eventKey="6">
                            <LayerOutEdges node={data} />
                        </LazyAccordionItem>} */}
                        {analysisResult.examplePerClass !== 0 && ['Conv2D', 'Dense', 'Concatenate', 'Conv2d', 'Linear', 'Cat', 'Add'].includes(data.layer_type) && <LazyAccordionItem header="Scatterplot View" headerColor='#2853c9' eventKey="4">
                            <ScatterplotView node={data} />
                        </LazyAccordionItem>}
                        {analysisResult.examplePerClass !== 0 && ['Conv2D', 'Dense', 'Concatenate', 'Conv2d', 'Linear', 'Cat', 'Add'].includes(data.layer_type) && <LazyAccordionItem header="Jaccard Similarity View" headerColor='#b2006d' eventKey="1">
                            <JaccardSimilarityView node={data} width={350} height={350} />
                        </LazyAccordionItem>}
                        {analysisResult.examplePerClass !== 0 && ['Conv2D', 'Dense', 'Concatenate', 'Conv2d', 'Linear', 'Cat', 'Add'].includes(data.layer_type) && <LazyAccordionItem header="Confusion Hierarchy" headerColor='black' eventKey="6">
                            <HierarchyTree node={data} />
                        </LazyAccordionItem>}
                        {analysisResult.examplePerClass !== 0 && [
                            // tensorflow
                            'Conv2D', 'Dense', 'Concatenate',
                            // pytorch
                            'Conv2d', 'Linear', 'Cat', 'Add',
                        ].includes(data.layer_type) && <LazyAccordionItem className={data.tutorial_node?'tutorial-cnn-layer-heatmap':''} header="Heatmap View" eventKey="0" headerColor='#00724a'>
                            <ActivationHeatmapView
                                node={data}
                                minWidth={350}
                                // height={data.output_shape[data.output_shape.length-1]!*HEATMAP_HEIGHT_FACTOR}
                                minHeight={300}
                                normalizeRow={true}
                            />
                        </LazyAccordionItem>}
                    </Accordion>
                </Card.Body>
            </Card>
            <Handle type="source" position={data.layout_horizontal?Position.Right:Position.Bottom} />
        </div>
    );
})

export default LayerNode;