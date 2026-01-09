import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { RootState } from '@/app/store';

export interface FeatureHuntState {
    uploadComplete: boolean
}

const initialState: FeatureHuntState = {
    uploadComplete: false
}

export const featureHuntSlice = createSlice({
    name: 'featureHunt',
    initialState,
    reducers: {
        setUploadComplete: (state, action: PayloadAction<boolean>) => {
            state.uploadComplete = action.payload;
        },
    },
});

export const {
    setUploadComplete,
} = featureHuntSlice.actions;
export const selectFeatureHunt = (state: RootState) => state.featureHunt;

export default featureHuntSlice.reducer;
