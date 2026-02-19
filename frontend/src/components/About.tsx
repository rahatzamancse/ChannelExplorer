'use client'

import React from 'react'
import '@styles/about.css'

function About() {
  return (
    <div className="about-container">
      <section className="about-hero">
        <h1 className="about-title">ChannelExplorer</h1>
        <p className="about-subtitle">
          A visual analytics tool for exploring neural network activation channels
        </p>
      </section>

      <section className="about-section">
        <h2>About The Project</h2>
        <p>
          ChannelExplorer is an interactive visualization system designed to help
          researchers and practitioners understand what individual channels in a
          neural network have learned. It extracts per-channel activation
          summaries from convolutional and dense layers, then presents them
          through coordinated views — heatmaps, dimensionality-reduced
          embeddings, clustering, and overlay visualizations — so users can
          quickly identify patterns, outliers, and redundancies across classes.
        </p>
        <p>
          The tool supports both <strong>TensorFlow / Keras</strong> and{' '}
          <strong>PyTorch</strong> models out of the box. Simply point it at a
          trained model and a labeled dataset, and the built-in server will
          expose a rich set of REST endpoints consumed by this frontend.
        </p>
      </section>

      <section className="about-section">
        <h2>Key Features</h2>
        <ul className="about-features">
          <li>
            <strong>Model Graph View</strong> — Visualize the architecture of
            your model as a layered, interactive graph.
          </li>
          <li>
            <strong>Activation Heatmaps</strong> — Inspect per-channel
            activation magnitudes across all images in a sortable heatmap.
          </li>
          <li>
            <strong>Embedding Projections</strong> — Project activations into
            2-D with MDS, t-SNE, UMAP, PCA, or autoencoders to reveal class
            separability at each layer.
          </li>
          <li>
            <strong>Activation Overlays</strong> — Superimpose channel
            activations onto the original input images to see which spatial
            regions drive each filter.
          </li>
          <li>
            <strong>Clustering &amp; Outlier Detection</strong> — Automatically
            cluster activations with X-Means or K-Means and flag outlier
            images.
          </li>
          <li>
            <strong>Pluggable Summary Functions</strong> — Choose from L2 norm,
            percentile, Otsu threshold, and more, or provide your own.
          </li>
        </ul>
      </section>

      <section className="about-section">
        <h2>How It Works</h2>
        <ol className="about-steps">
          <li>
            Pass a trained model and dataset to{' '}
            <code>ChannelExplorer_TF</code> or <code>APAnalysisTorchModel</code>.
          </li>
          <li>
            Call <code>server.run()</code> to start the FastAPI backend.
          </li>
          <li>
            Open the frontend in your browser, select the classes and number of
            examples, then hit <em>Analyze</em>.
          </li>
          <li>
            Explore the results layer-by-layer through the interactive graph
            and coordinated views.
          </li>
        </ol>
      </section>

      <section className="about-section about-author">
        <h2>Author</h2>
        <p>
          <strong><a href="https://rahatzaman.me" target="_blank" rel="noopener noreferrer">Rahat Zaman</a></strong> started his Ph.D. program at the
          University of Utah with Dr. Paul Rosen.
        </p>
        <div className="about-links">
          <a
            href="https://github.com/rahatzamancse"
            target="_blank"
            rel="noopener noreferrer"
          >
            GitHub
          </a>
          <a
            href="https://rahatzaman.me"
            target="_blank"
            rel="noopener noreferrer"
          >
            Website
          </a>
        </div>
      </section>
    </div>
  )
}

export default About
