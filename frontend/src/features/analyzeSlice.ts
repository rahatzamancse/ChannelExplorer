import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { RootState } from '@/app/store';

export interface AnalysisConfig {
    // selected class indices
    selectedClasses: number[]
    // examples per class
    examplePerClass: number
    // User selected Images indices
    selectedImages: number[]
    // Whether the images are shuffled
    shuffled: boolean
    // Prediction of current selected classes
    predictions: number[]
}

const initialState: AnalysisConfig = {
    selectedClasses: [],
    examplePerClass: 0,
    selectedImages: [],
    shuffled: false,
    predictions: [],
}

export const analysisResultSlice = createSlice({
    name: 'analysisResult',
    initialState,
    reducers: {
        setAnalysisResult: (state, action: PayloadAction<AnalysisConfig>) => {
            state.selectedClasses = action.payload.selectedClasses
            state.examplePerClass = action.payload.examplePerClass
            state.shuffled = action.payload.shuffled
            state.predictions = action.payload.predictions
        },
        setSelectedImgs: (state, action: PayloadAction<number[]>) => {
            state.selectedImages = action.payload;
        }
    },
});

export const {
    setAnalysisResult,
    setSelectedImgs
} = analysisResultSlice.actions;
export const selectAnalysisResult = (state: RootState) => state.analysisResult;

export default analysisResultSlice.reducer;
