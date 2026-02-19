'use client'

import React from 'react'
import Controls from '@components/Controls'
import GraphViewer from '@components/GraphViewer'
import { ReactFlowProvider } from '@xyflow/react'

function MainView() {
    return <div style={{
        display: "flex",
        flexDirection: "row",
    }}>
        <Controls />
        <ReactFlowProvider>
            <GraphViewer />
        </ReactFlowProvider>
    </div>
}

export default MainView
