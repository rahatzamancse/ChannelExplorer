import React from 'react';
import Navigation from '@components/Navigation';
import { Route, Routes } from 'react-router-dom';
import About from '@components/About';
import MainView from '@components/MainView';
import FeatureHunt from '@components/featurehunt/FeatureHunt';
import { TourProvider } from '@reactour/tour'
import { tutorialSteps } from '@/tutorialSteps';

function App() {
  return <TourProvider steps={tutorialSteps}>
    <Navigation />
    <Routes>
      <Route path="/" element={<MainView />} />
      <Route path="/featurehunt" element={<FeatureHunt />} />
      <Route path="/about" element={<About />} />
    </Routes>
  </TourProvider>
}

export default App;
