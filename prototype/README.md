# Casimir simplified interface prototype

Double-click `Open Prototype.cmd` on Windows, or open `index.html` in a modern browser. All assets are local; no installation is required.

## User journey

1. Start with the default scenario, or open Customize scenario for optional adjustments.
2. Select Initialize simulation and wait.
3. Read the brief result. Use View more results for details, or Save result to revisit it.

The only navigation entries are Simulation and Saved results. Data preparation, model selection, training, evaluation metrics, and configuration controls are not exposed to users. The guide explains the same three-step journey.

## Review boundaries

This is an interface prototype. Initialization demonstrates a short waiting state and returns a fixed illustrative value. No data generation, training, or inference takes place. Entered values do not change the output. The real engine remains the next development priority.

`simulationAdapter.initialize(inputs)` is the integration boundary for the future Python backend. That backend must independently validate inputs, manage reproducible data and saved pipelines internally, and provide computed results. The interface should require no manual model-development workflow.

Theme preferences, guide state, and examples use browser storage. Privacy settings and file URLs may affect persistence. Reports are exported as JSON and marked as illustrative examples.

`styles.css` contains the shared theme and shell; `simplified.css` contains the simplified user flow. No external fonts or icon downloads are required.
