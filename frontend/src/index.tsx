import React from 'react';
import { Provider } from 'react-redux';
import { store } from '@/app/store';
import ReactDOM from 'react-dom/client';
import App from '@/App';
import { BrowserRouter as Router } from "react-router-dom"

// Importing the Bootstrap CSS
import 'bootstrap/dist/css/bootstrap.min.css';
import '@styles/index.css'

// Suppress benign ResizeObserver error that occurs with accordion animations
// This error doesn't affect functionality - it just means the browser couldn't
// deliver all resize notifications within a single animation frame
const resizeObserverErr = (e: ErrorEvent) => {
  if (e.message === 'ResizeObserver loop completed with undelivered notifications.') {
    e.stopImmediatePropagation();
    e.preventDefault();
  }
};
window.addEventListener('error', resizeObserverErr, true);

const root = ReactDOM.createRoot(
  document.getElementById('root') as HTMLElement
);
root.render(
  <React.StrictMode>
    <Router>
      <Provider store={store}>
        <App />
      </Provider>
    </Router>
  </React.StrictMode>
);